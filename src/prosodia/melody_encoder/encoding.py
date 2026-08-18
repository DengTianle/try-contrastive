from __future__ import annotations

import numpy as np


PITCH_CHANGE_BINS = 128
DURATION_BINS = 24
ONSET_SHIFT_BINS = 24
MELODY_FEATURE_DIM = PITCH_CHANGE_BINS + 1 + DURATION_BINS + ONSET_SHIFT_BINS
MELODY_REPRESENTATION = "mlm_note_177d_v1"

PITCH_CHANGE_SLICE = slice(0, PITCH_CHANGE_BINS)
PITCH_SIGN_INDEX = PITCH_CHANGE_BINS
DURATION_SLICE = slice(PITCH_SIGN_INDEX + 1, PITCH_SIGN_INDEX + 1 + DURATION_BINS)
ONSET_SHIFT_SLICE = slice(DURATION_SLICE.stop, DURATION_SLICE.stop + ONSET_SHIFT_BINS)


def hz_to_midi_pitch(frequency_hz: float) -> int:
    """Convert Hz to the integer MIDI pitch used by the MLM reference code."""
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError(f"Expected a positive finite frequency, got {frequency_hz}")
    return int(69.0 + 12.0 * np.log2(float(frequency_hz) / 440.0))


def _quantize_log_values(values: np.ndarray, num_bins: int) -> np.ndarray:
    logged = np.log2(values.astype(np.float32) + np.float32(1e-6))
    minimum = np.min(logged)
    maximum = np.max(logged)
    normalized = (logged - minimum) / (maximum - minimum + np.float32(1e-6))
    normalized = np.clip(normalized, 0.0, 1.0)
    indices = (normalized * (num_bins - 1)).astype(np.int64)
    return np.eye(num_bins, dtype=np.float32)[indices]


def encode_note_sequence(
    midi_pitches: np.ndarray,
    onset_seconds: np.ndarray,
    duration_seconds: np.ndarray,
) -> np.ndarray:
    """Encode notes with the 177-D representation from Wang et al. (2026).

    The feature blocks are absolute pitch displacement from the first note (128-D
    one-hot plus a sign bit), log-duration (24-D one-hot), and log inter-onset
    interval (24-D one-hot). Duration and onset shift are min-max normalized per
    sequence, matching the authors' implementation.
    """
    pitches = np.asarray(midi_pitches)
    onsets = np.asarray(onset_seconds, dtype=np.float32)
    durations = np.asarray(duration_seconds, dtype=np.float32)
    if pitches.ndim != 1 or onsets.ndim != 1 or durations.ndim != 1:
        raise ValueError("midi_pitches, onset_seconds, and duration_seconds must be 1-D")
    if not (pitches.shape == onsets.shape == durations.shape):
        raise ValueError("Pitch, onset, and duration arrays must have identical shapes")
    if pitches.size == 0:
        raise ValueError("Cannot encode an empty note sequence")
    if not np.all(np.isfinite(pitches)) or not np.all(np.equal(pitches, np.trunc(pitches))):
        raise ValueError("MIDI pitches must be finite integers")
    if not np.all(np.isfinite(onsets)) or not np.all(np.diff(onsets) >= 0.0):
        raise ValueError("Note onsets must be finite and nondecreasing")
    if not np.all(np.isfinite(durations)) or not np.all(durations > 0.0):
        raise ValueError("Note durations must be positive and finite")

    pitch_changes = pitches.astype(np.int64) - int(pitches[0])
    magnitudes = np.abs(pitch_changes)
    if np.any(magnitudes >= PITCH_CHANGE_BINS):
        raise ValueError("Pitch displacement from the first note must be in [-127, 127]")
    pitch_one_hot = np.eye(PITCH_CHANGE_BINS, dtype=np.float32)[magnitudes]
    pitch_sign = (pitch_changes >= 0).astype(np.float32)[:, None]

    onset_shifts = np.zeros_like(onsets)
    onset_shifts[1:] = np.diff(onsets)
    duration_one_hot = _quantize_log_values(durations, DURATION_BINS)
    onset_one_hot = _quantize_log_values(onset_shifts, ONSET_SHIFT_BINS)
    encoded = np.concatenate(
        [pitch_one_hot, pitch_sign, duration_one_hot, onset_one_hot],
        axis=1,
    )
    if encoded.shape[1] != MELODY_FEATURE_DIM:
        raise AssertionError(f"Unexpected melody feature dimension: {encoded.shape}")
    return encoded.astype(np.float32, copy=False)
