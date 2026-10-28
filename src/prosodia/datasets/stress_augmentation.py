"""CPU, worker-local word-consistent prominence reversal for hard negatives."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class StressNegativeConfig:
    # Public epochs are one-based: 5 means four complete unaugmented epochs.
    start_epoch: int = 5
    ramp_epochs: float = 2.0
    weight: float = 0.25
    probability: float = 0.5
    max_gain_db: float = 6.0
    min_changed_fraction: float = 0.6
    max_changed_fraction: float = 0.8
    min_changed_duration_fraction: float = 0.5
    min_annotated_fraction: float = 0.8
    min_change_db: float = 1.5
    min_note_seconds: float = 0.1
    fade_seconds: float = 0.02
    min_loudness_span_db: float = 3.0
    max_rank_correlation: float = -0.25

    def __post_init__(self) -> None:
        numeric = [value for value in vars(self).values()]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Stress-negative settings must be finite")
        if self.start_epoch < 1 or self.ramp_epochs < 0 or not 0 < self.weight <= 1:
            raise ValueError("Stress start epoch must be >= 1, ramp >= 0, and weight in (0, 1]")
        if not 0 <= self.probability <= 1:
            raise ValueError("Stress-negative probability must be in [0, 1]")
        if not 0 < self.min_changed_fraction <= self.max_changed_fraction <= 1:
            raise ValueError("Stress changed-note fractions must satisfy 0 < min <= max <= 1")
        if not 0 < self.min_changed_duration_fraction <= 1 or not 0 < self.min_annotated_fraction <= 1:
            raise ValueError("Stress duration/annotation coverage fractions must be in (0, 1]")
        if not 0 < self.min_change_db <= self.max_gain_db:
            raise ValueError("Stress gain limit must be >= the positive minimum change")
        if min(self.min_note_seconds, self.fade_seconds, self.min_loudness_span_db) <= 0:
            raise ValueError("Stress note duration, fade, and loudness span must be positive")
        if not -1 <= self.max_rank_correlation < 0:
            raise ValueError("Stress rank-correlation threshold must be in [-1, 0)")

    def active(self, epoch_index: int) -> bool:
        return epoch_index + 1 >= self.start_epoch and self.probability > 0

    def loss_weight(self, epoch_index: int, step: int, steps_per_epoch: int) -> float:
        if not self.active(epoch_index):
            return 0.0
        if self.ramp_epochs == 0:
            return self.weight
        elapsed = epoch_index - (self.start_epoch - 1) + step / max(steps_per_epoch, 1)
        return self.weight * min(max(elapsed / self.ramp_epochs, 0.0), 1.0)


def _note_loudness(audio: np.ndarray, start: int, end: int, sample_rate: int) -> float:
    # Robust active-frame RMS avoids silence, consonant edges, and single peaks.
    trim = min(round(0.02 * sample_rate), (end - start) // 5)
    interior = audio[start + trim:end - trim]
    frame = max(1, round(0.02 * sample_rate))
    count = len(interior) // frame
    if count == 0:
        return float("-inf")
    powers = np.mean(interior[:count * frame].reshape(count, frame).astype(np.float64) ** 2, axis=1)
    active = powers[powers >= max(float(np.max(powers)) * 0.01, 1e-10)]
    return 10.0 * math.log10(float(np.median(active))) if active.size else float("-inf")


def _ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    average = np.cumsum(counts) - (counts + 1) / 2
    return average[inverse]


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a, b = _ranks(first), _ranks(second)
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / denominator if denominator > 0 else 1.0


def reverse_word_prominence(
    audio: torch.Tensor,
    note_onsets: torch.Tensor,
    note_durations: torch.Tensor,
    metadata: dict[str, np.ndarray],
    sample_rate: int,
    config: StressNegativeConfig,
    rng: random.Random,
) -> tuple[torch.Tensor | None, dict[str, float | str]]:
    """Return one accepted edit or a rejection reason; never mutate the source.

    Metadata intervals and note IDs must already be shifted/clipped to this
    candidate's window. Whole-word gain protects relative levels inside words.
    """
    if audio.device.type != "cpu" or audio.ndim != 1:
        raise ValueError("Stress-negative generation expects a mono CPU waveform")
    waveform = audio.numpy()
    starts = note_onsets.numpy()
    durations = note_durations.numpy()
    word_ids = metadata["note_word_ids"]
    if word_ids.shape != starts.shape or durations.shape != starts.shape:
        raise ValueError("Stress metadata and candidate notes must have matching lengths")
    if len(starts) and np.mean(word_ids >= 0) < config.min_annotated_fraction:
        return None, {"reason": "incomplete_word_links"}
    if not np.isfinite(waveform).all():
        return None, {"reason": "nonfinite_audio"}
    length_seconds = len(waveform) / sample_rate
    word_starts = metadata["word_onset_seconds"]
    word_ends = word_starts + metadata["word_duration_seconds"]
    lines = metadata["word_line_ids"]
    eligible = np.zeros(len(starts), dtype=bool)
    levels = np.full(len(word_starts), np.nan)
    word_note_indices: dict[int, np.ndarray] = {}
    for word in range(len(word_starts)):
        # Never change a truncated word or overlap a neighboring word.
        if word_starts[word] < -1e-5 or word_ends[word] > length_seconds + 1e-5:
            continue
        overlaps = (word_starts < word_ends[word] - 0.001) & (word_ends > word_starts[word] + 0.001)
        overlaps[word] = False
        if np.any(overlaps):
            continue
        indices = np.flatnonzero(
            (word_ids == word) & (durations >= config.min_note_seconds)
            & (starts >= word_starts[word] - 0.001)
            & (starts + durations <= word_ends[word] + 0.001)
        )
        loudness = []
        valid_indices = []
        for index in indices:
            begin = max(0, round(float(starts[index]) * sample_rate))
            end = min(len(waveform), round(float(starts[index] + durations[index]) * sample_rate))
            level = _note_loudness(waveform, begin, end, sample_rate)
            if math.isfinite(level):
                loudness.append(level)
                valid_indices.append(index)
        if valid_indices:
            levels[word] = np.median(loudness)
            word_note_indices[word] = np.asarray(valid_indices)
            eligible[valid_indices] = True
    if eligible.sum() < 3:
        return None, {"reason": "too_few_eligible_notes"}
    usable_lines = []
    for line in np.unique(lines):
        words = np.flatnonzero((lines == line) & np.isfinite(levels))
        if len(words) >= 3 and np.ptp(levels[words]) >= config.min_loudness_span_db:
            usable_lines.append(words[np.argsort(levels[words], kind="stable")])
    if not usable_lines:
        return None, {"reason": "insufficient_word_contrast"}

    # A small bounded search handles unequal note counts per word. Pairs exchange
    # the line's existing loudness ranks; no synthesized stress labels are needed.
    pairs = [
        (int(words[index]), int(words[-index - 1]))
        for words in usable_lines for index in range(len(words) // 2)
    ]
    for attempt in range(8):
        order = list(pairs)
        rng.shuffle(order)
        wanted_fraction = rng.uniform(config.min_changed_fraction, config.max_changed_fraction)
        gains = np.zeros(len(word_starts), dtype=np.float64)
        for quiet, loud in order:
            difference = min(float(levels[loud] - levels[quiet]), config.max_gain_db)
            if difference < config.min_change_db:
                continue
            gains[quiet], gains[loud] = difference, -difference
            changed = np.zeros_like(eligible)
            for word in np.flatnonzero(np.abs(gains) >= config.min_change_db):
                changed[word_note_indices[int(word)]] = True
            if changed.sum() / eligible.sum() >= wanted_fraction:
                break
        if attempt >= 4:
            # With four one-note words, pair swaps can change only 50% or 100%.
            # A three-word permutation can change 75% while retaining the same
            # loudness values. The same post-edit reversal gates still apply.
            words = rng.choice(usable_lines)
            middle = int(rng.choice(words[1:-1]))
            selected = [int(words[0]), middle, int(words[-1])]
            gains[words] = 0
            targets = np.roll(levels[selected], rng.choice((-1, 1)))
            gains[selected] = np.clip(targets - levels[selected], -config.max_gain_db, config.max_gain_db)
        envelope_db = np.zeros(len(waveform), dtype=np.float32)
        for word in np.flatnonzero(gains):
            begin = max(0, round(float(word_starts[word]) * sample_rate))
            end = min(len(waveform), round(float(word_ends[word]) * sample_rate))
            fade = min(round(config.fade_seconds * sample_rate), (end - begin) // 4)
            shape = np.ones(end - begin, dtype=np.float32)
            if fade > 1:
                ramp = (1 - np.cos(np.linspace(0, math.pi, fade))) / 2
                shape[:fade], shape[-fade:] = ramp, ramp[::-1]
            envelope_db[begin:end] = gains[word] * shape
        edited = waveform * np.power(10.0, envelope_db / 20.0)
        # Re-measure the actual edit, including ramps and gain caps.
        new_levels = levels.copy()
        actual_changes = np.zeros(len(starts), dtype=np.float64)
        for word, indices in word_note_indices.items():
            new_notes = []
            for index in indices:
                begin = max(0, round(float(starts[index]) * sample_rate))
                end = min(len(waveform), round(float(starts[index] + durations[index]) * sample_rate))
                new = _note_loudness(edited, begin, end, sample_rate)
                old = _note_loudness(waveform, begin, end, sample_rate)
                actual_changes[index] = abs(new - old)
                new_notes.append(new)
            new_levels[word] = np.median(new_notes)
        changed = eligible & (actual_changes >= config.min_change_db)
        fraction = float(changed.sum() / eligible.sum())
        duration_fraction = float(durations[changed].sum() / durations[eligible].sum())
        correlations = [
            _rank_correlation(levels[words], new_levels[words]) for words in usable_lines
            if np.any(gains[words])
        ]
        correlation = float(np.mean(correlations)) if correlations else 1.0
        if not (
            config.min_changed_fraction <= fraction <= config.max_changed_fraction
            and duration_fraction >= config.min_changed_duration_fraction
            and correlation <= config.max_rank_correlation
        ):
            continue
        original_rms = math.sqrt(float(np.mean(waveform.astype(np.float64) ** 2)))
        edited_rms = math.sqrt(float(np.mean(edited.astype(np.float64) ** 2)))
        if original_rms < 1e-6 or edited_rms < 1e-6:
            return None, {"reason": "silent_audio"}
        edited *= original_rms / edited_rms
        # Reject instead of clipping or changing the positive's level.
        if float(np.max(np.abs(edited))) > 0.99:
            return None, {"reason": "insufficient_headroom"}
        return torch.from_numpy(edited.astype(np.float32)), {
            "reason": "accepted", "changed_note_fraction": fraction,
            "changed_duration_fraction": duration_fraction, "rank_correlation": correlation,
            "eligible_note_count": int(eligible.sum()), "total_note_count": len(starts),
            "changed_note_count": int(changed.sum()),
            "changed_all_note_fraction": float(changed.sum() / len(starts)),
        }
    return None, {"reason": "weak_or_excessive_edit"}
