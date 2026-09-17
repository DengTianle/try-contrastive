from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_contrastive import (
    StagedWarmupCosineScheduler,
    distributed_totals,
    evaluate,
    step_optimizer_and_scheduler,
    train_one_epoch,
)
from pretrain_melody_encoder import pretrain_step
from prosodia.melody_encoder import MELODY_FEATURE_DIM, MelodyMaskedProsodyModel
from prosodia.training import grouped_info_nce_loss


class EmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, melody_features, candidate_input_values, **kwargs):
        return melody_features * self.scale, candidate_input_values


class ProgressBatches(list):
    def __init__(self, batches):
        super().__init__(batches)
        self.updates = []

    def set_postfix(self, **values):
        self.updates.append(values)


class EpochMetricsTest(unittest.TestCase):
    def test_distributed_totals_reduce_before_host_transfer(self) -> None:
        totals = torch.arange(1, 9, dtype=torch.float64)
        with (
            patch("train_contrastive.dist.is_initialized", return_value=True),
            patch("train_contrastive.dist.all_reduce", side_effect=lambda tensor, op: tensor.mul_(2)) as reduce,
        ):
            result = distributed_totals(totals)
        self.assertEqual(result, [float(2 * n) for n in range(1, 9)])
        reduce.assert_called_once()
        self.assertIs(reduce.call_args.args[0], totals)

    def setUp(self) -> None:
        torch.manual_seed(14)
        self.batches = []
        for size in (2, 1, 3, 2):
            self.batches.append({
                "melody_features": torch.randn(size, 4),
                "candidate_input_values": torch.randn(size, 3, 4),
                "target": torch.arange(size) % 3,
                "candidate_mask": torch.ones(size, 3, dtype=torch.bool),
                "melody_attention_mask": None,
                "candidate_audio_attention_mask": torch.ones(size, 3, 4, dtype=torch.bool),
                "candidate_note_onsets": None,
                "candidate_note_durations": None,
                "candidate_note_attention_mask": None,
            })

    def reference(self, global_weight, symmetric):
        totals = dict(loss=0., hard_loss=0., global_loss=0., accuracy=0.,
                      global_accuracy=0., global_audio_accuracy=0.)
        validation = dict(loss=0., accuracy=0., mean_rank=0., mrr=0.,
                          recall_at_1=0., recall_at_2=0., recall_at_3=0., recall_at_5=0.)
        count = global_count = 0
        for batch in self.batches:
            melody, audio, targets = (
                batch["melody_features"], batch["candidate_input_values"], batch["target"],
            )
            size = len(targets)
            hard_loss, logits = grouped_info_nce_loss(melody, audio, targets=targets, temperature=1.)
            correct = int((logits.argmax(-1) == targets).sum())
            global_loss = 0.
            if global_weight > 0 and size > 1:
                global_logits = melody @ audio[torch.arange(size), targets].T
                labels = torch.arange(size)
                global_loss = float(torch.nn.functional.cross_entropy(global_logits, labels))
                if symmetric:
                    reverse = torch.nn.functional.cross_entropy(global_logits.T, labels)
                    global_loss = (global_loss + float(reverse)) / 2
                    totals["global_audio_accuracy"] += int((global_logits.argmax(0) == labels).sum())
                totals["global_accuracy"] += int((global_logits.argmax(1) == labels).sum())
                totals["global_loss"] += global_loss * size
                global_count += size
            totals["loss"] += float(hard_loss + global_weight * global_loss) * size
            totals["hard_loss"] += float(hard_loss) * size
            totals["accuracy"] += correct
            ranks = (logits >= logits.gather(1, targets[:, None])).sum(-1)
            validation["loss"] += float(hard_loss) * size
            validation["accuracy"] += correct
            validation["recall_at_1"] += correct
            validation["mean_rank"] += float(ranks.sum())
            validation["mrr"] += float((1 / ranks.float()).sum())
            for k in (2, 3, 5):
                validation[f"recall_at_{k}"] += int((ranks <= k).sum())
            count += size
        return (
            {key: value / max(global_count if key.startswith("global_") else count, 1)
             for key, value in totals.items()},
            {key: value / count for key, value in validation.items()},
        )

    def test_metrics_and_transfer_frequency(self) -> None:
        original_cpu = torch.Tensor.cpu
        for weight, symmetric in ((0., False), (.5, False), (.5, True)):
            expected_train, expected_val = self.reference(weight, symmetric)
            for progress, interval in ((True, 2), (True, 0), (False, 1)):
                with self.subTest(weight=weight, symmetric=symmetric, progress=progress, interval=interval):
                    model = EmbeddingModel()
                    # Keep embeddings fixed while exercising backward and optimizer updates.
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
                    scaler = torch.amp.GradScaler("cpu", enabled=False)
                    for training, expected in ((True, expected_train), (False, expected_val)):
                        transfers = []

                        def tracked_cpu(tensor, *args, **kwargs):
                            self.assertFalse(tensor.requires_grad)
                            transfers.append(tensor.numel())
                            return original_cpu(tensor, *args, **kwargs)

                        bar = ProgressBatches(self.batches)
                        common = dict(
                            model=model, loader=self.batches, device=torch.device("cpu"),
                            temperature=1., use_amp=False, progress=progress,
                            desc="test", log_every_steps=interval,
                        )
                        with (
                            patch("train_contrastive.maybe_progress", return_value=bar),
                            patch.object(torch.Tensor, "cpu", tracked_cpu),
                        ):
                            if training:
                                actual = train_one_epoch(
                                    **common, optimizer=optimizer, scaler=scaler,
                                    scheduler=SimpleNamespace(step=lambda: None),
                                    global_loss_weight=weight, symmetric_global_loss=symmetric,
                                    audio_noise_snr_db=None, audio_background_mix_prob=0.,
                                    audio_background_mix_snr_db=35., grad_clip_norm=None,
                                )
                            else:
                                actual = evaluate(**common)
                        for key, value in expected.items():
                            self.assertAlmostEqual(actual[key], value, places=6)
                        updates = 2 if progress and interval == 2 else 0
                        self.assertEqual(len(bar.updates), updates)
                        self.assertEqual(len(transfers), updates + 1)


class OptimizerUpdateTest(unittest.TestCase):
    def test_scheduler_advances_only_after_successful_optimizer_step(self) -> None:
        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [parameter],
                    "peak_lr": 0.1,
                    "schedule_name": "head",
                    "lr": 0.1,
                }
            ]
        )
        scheduler = StagedWarmupCosineScheduler(
            optimizer,
            total_steps=10,
            head_warmup_steps=0,
            hubert_unfreeze_step=None,
            hubert_warmup_steps=0,
            min_lr_ratio=0.01,
            enabled=True,
        )
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0)

        overflowing_loss = parameter * torch.tensor(float("inf"))
        scaler.scale(overflowing_loss).backward()
        skipped = step_optimizer_and_scheduler(
            scaler,
            optimizer,
            scheduler,
        )
        self.assertFalse(skipped)
        self.assertEqual(scheduler.update_step, 0)
        self.assertEqual(float(parameter.detach()), 1.0)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(parameter).backward()
        succeeded = step_optimizer_and_scheduler(
            scaler,
            optimizer,
            scheduler,
        )
        self.assertTrue(succeeded)
        self.assertEqual(scheduler.update_step, 1)
        self.assertLess(float(parameter.detach()), 1.0)


class MaskedNotePretrainingTest(unittest.TestCase):
    def test_pretraining_classifies_all_four_note_attributes(self) -> None:
        features = torch.zeros(1, 2, MELODY_FEATURE_DIM)
        features[0, :, 0] = 1.0
        features[0, :, 128] = 1.0
        features[0, :, 129] = 1.0
        features[0, :, 153] = 1.0
        batch = {
            "melody_features": features,
            "melody_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        model = MelodyMaskedProsodyModel(
            d_model=8,
            num_layers=1,
            num_heads=2,
            dim_feedforward=16,
            dropout=0.0,
        )

        loss, metrics = pretrain_step(
            model=model,
            batch=batch,
            mask_prob=0.5,
            mask_span_notes=1,
            mask_generator=torch.Generator().manual_seed(0),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["pitch_change_loss"], 0.0)
        self.assertGreater(metrics["duration_loss"], 0.0)
        self.assertGreater(metrics["onset_shift_loss"], 0.0)
        self.assertGreater(metrics["masked_notes"], 0.0)


if __name__ == "__main__":
    unittest.main()
