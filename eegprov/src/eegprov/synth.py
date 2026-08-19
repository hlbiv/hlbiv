"""Synthetic EEG generation.

Exists so the toolchain is testable with zero downloads and zero credentials.
Every quality problem the QC pass is meant to catch can be injected here on
purpose, which is the only way to know the detector actually detects it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

# 10-20 subset. Pz/Cz/Fz carry the P300; frontals carry blink artifact.
DEFAULT_MONTAGE = [
    "Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4",
    "P3", "Pz", "P4", "O1", "O2", "T7", "T8", "M1",
]


@dataclass
class SynthSpec:
    """What to generate, including which defects to inject."""

    n_seconds: float = 120.0
    sfreq: float = 250.0
    ch_names: list[str] = field(default_factory=lambda: list(DEFAULT_MONTAGE))
    line_freq: float = 60.0
    line_amp_uv: float = 2.0
    # Injected defects — each maps to a QC check we expect to fire.
    flat_channels: list[str] = field(default_factory=list)
    noisy_channels: list[str] = field(default_factory=list)
    drift_channels: list[str] = field(default_factory=list)
    blink_rate_per_min: float = 12.0
    # P300 oddball paradigm; None disables events entirely.
    oddball: bool = True
    oddball_isi_s: float = 1.0
    oddball_target_prob: float = 0.2
    p300_amp_uv: float = 6.0
    seed: int = 0


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f background, which is what real EEG spectra actually look like."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    out = np.fft.irfft(spec * scale, n=n)
    return out / (np.std(out) or 1.0)


def _alpha_bump(n: int, sfreq: float, rng: np.random.Generator) -> np.ndarray:
    """Waxing/waning ~10 Hz rhythm, strongest posteriorly in real data."""
    t = np.arange(n) / sfreq
    freq = 10.0 + 0.4 * rng.standard_normal()
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2 * np.pi))
    return envelope * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))


def _blink_train(n: int, sfreq: float, rate_per_min: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Slow high-amplitude deflections, the dominant frontal artifact."""
    out = np.zeros(n)
    if rate_per_min <= 0:
        return out
    n_blinks = int(round(rate_per_min * (n / sfreq) / 60.0))
    width = int(0.25 * sfreq)
    if width < 3:
        return out
    shape = np.hanning(width)
    for _ in range(n_blinks):
        start = rng.integers(0, max(1, n - width))
        out[start:start + width] += shape * rng.uniform(0.7, 1.3)
    return out


def _p300_wave(sfreq: float, amp_uv: float) -> np.ndarray:
    """Positive deflection peaking ~300 ms post-stimulus."""
    dur = 0.8
    t = np.arange(int(dur * sfreq)) / sfreq
    return amp_uv * np.exp(-((t - 0.30) ** 2) / (2 * 0.06 ** 2))


def make_raw(spec: SynthSpec) -> mne.io.RawArray:
    """Build a synthetic Raw with the defects named in `spec` injected."""
    rng = np.random.default_rng(spec.seed)
    n = int(spec.n_seconds * spec.sfreq)
    t = np.arange(n) / spec.sfreq
    data = np.zeros((len(spec.ch_names), n))

    posterior = {"P3", "Pz", "P4", "O1", "O2"}
    frontal = {"Fp1", "Fp2", "F3", "Fz", "F4"}

    for i, ch in enumerate(spec.ch_names):
        sig = 12.0 * _pink_noise(n, rng)
        sig += (8.0 if ch in posterior else 3.0) * _alpha_bump(n, spec.sfreq, rng)
        sig += spec.line_amp_uv * np.sin(2 * np.pi * spec.line_freq * t)
        if ch in frontal:
            sig += 60.0 * _blink_train(n, spec.sfreq, spec.blink_rate_per_min, rng)
        if ch in spec.noisy_channels:
            sig += 90.0 * rng.standard_normal(n)
        if ch in spec.drift_channels:
            sig += 70.0 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6.28))
        data[i] = sig

    events, event_desc = _oddball_events(spec, n, data)

    # Flatten last. A dead electrode records nothing, including the evoked
    # response — zeroing before event injection would leave the P300 on top
    # of an otherwise flat channel and defeat the detector.
    for ch in spec.flat_channels:
        if ch in spec.ch_names:
            data[spec.ch_names.index(ch)] = 1e-3 * rng.standard_normal(n)

    info = mne.create_info(spec.ch_names, spec.sfreq, ch_types="eeg")
    info["line_freq"] = spec.line_freq
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")  # MNE wants volts

    if events:
        onsets = [e[0] / spec.sfreq for e in events]
        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=[0.0] * len(onsets),
                description=[event_desc[e[1]] for e in events],
            )
        )
    return raw


def _oddball_events(spec: SynthSpec, n: int,
                    data: np.ndarray) -> tuple[list[tuple[int, int]], dict[int, str]]:
    """Stamp an oddball sequence and add the P300 response to targets."""
    if not spec.oddball:
        return [], {}
    rng = np.random.default_rng(spec.seed + 1)
    step = int(spec.oddball_isi_s * spec.sfreq)
    wave = _p300_wave(spec.sfreq, spec.p300_amp_uv)
    # Midline sites carry the response most strongly.
    weights = {"Pz": 1.0, "Cz": 0.8, "Fz": 0.45, "P3": 0.6, "P4": 0.6}

    events: list[tuple[int, int]] = []
    for onset in range(step, n - len(wave), step):
        is_target = rng.random() < spec.oddball_target_prob
        events.append((onset, 1 if is_target else 0))
        if not is_target:
            continue
        for ch, w in weights.items():
            if ch in spec.ch_names:
                idx = spec.ch_names.index(ch)
                data[idx, onset:onset + len(wave)] += w * wave
    return events, {0: "standard", 1: "target"}


def write_session(out_path: Path, spec: SynthSpec) -> Path:
    """Write a synthetic session to disk, preferring EDF for realism."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = make_raw(spec)
    if out_path.suffix.lower() == ".edf":
        try:
            raw.export(str(out_path), fmt="edf", overwrite=True, verbose="ERROR")
            return out_path
        except Exception:
            # edfio missing — fall back rather than fail the whole fixture.
            out_path = out_path.with_suffix(".raw.fif")
    raw.save(str(out_path), overwrite=True, verbose="ERROR")
    return out_path
