from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_contrastive import (
    StagedWarmupCosineScheduler,
    step_optimizer_and_scheduler,
)


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


if __name__ == "__main__":
    unittest.main()
