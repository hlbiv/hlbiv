"""Signal-quality metrics for a single EEG recording.

These are the numbers that go on the provenance card. Every metric here is
one a human would otherwise eyeball by scrolling the recording, which is the
step this whole toolchain exists to remove from the critical path.

Thresholds are deliberately conservative: this pass FLAGS, it does not
silently drop data. A human resolves anything ambiguous.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import signal as sps

# Channels whose std is below this are dead, not quiet. Real scalp EEG with
# a working electrode does not sit under 0.5 uV RMS.
FLAT_STD_UV = 0.5
# Robust z-score above which a channel's amplitude is an outlier vs its peers.
NOISY_ROBUST_Z = 4.0
# ...but a statistical outlier is not automatically a large one. On a montage
# where most channels are near-identical, MAD collapses and ordinary
# scalp-topography variation (frontal blink pickup, posterior alpha) scores
# z > 4 while sitting within 25% of the median. A channel must ALSO be
# genuinely high-amplitude before it is called noisy.
NOISY_ABS_RATIO = 2.0
# A sample above this is almost certainly artifact, not brain.
ARTIFACT_AMP_UV = 150.0
# Drift is measured below 0.5 Hz, not below 1 Hz: real EEG is 1/f, so a
# 0.1-1 Hz band is dominated by legitimate delta and slow-cortical activity
# and flags healthy channels. Clean 1/f sits near 0.18 in this band; injected
# electrode drift sits near 0.49.
DRIFT_LO_HZ, DRIFT_HI_HZ = 0.05, 0.5
DRIFT_RATIO = 0.30
# Line power this many times the neighbouring baseline is contamination.
LINE_RATIO_WARN = 3.0


@dataclass
class ChannelReport:
    name: str
    std_uv: float
    ptp_uv: float
    robust_z: float
    drift_ratio: float
    flags: list[str] = field(default_factory=list)


@dataclass
class QCReport:
    n_channels: int
    sfreq_hz: float
    duration_s: float
    line_freq_detected: float | None
    line_ratio: float | None
    blink_rate_per_min: float | None
    artifact_fraction: float
    retained_fraction: float
    channels: list[ChannelReport]
    flags: list[str] = field(default_factory=list)

    @property
    def bad_channels(self) -> list[str]:
        return [c.name for c in self.channels if c.flags]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bad_channels"] = self.bad_channels
        return d


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD z-score. Mean and std are useless here — one blown channel
    drags them both, and then nothing looks like an outlier."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def _band_power(data_uv: np.ndarray, sfreq: float,
                lo: float, hi: float) -> np.ndarray:
    nperseg = int(min(len(data_uv[0]), sfreq * 4))
    if nperseg < 16:
        return np.zeros(data_uv.shape[0])
    freqs, psd = sps.welch(data_uv, fs=sfreq, nperseg=nperseg, axis=-1)
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return np.zeros(data_uv.shape[0])
    return psd[:, mask].sum(axis=-1)


def _detect_line(data_uv: np.ndarray, sfreq: float) -> tuple[float | None, float | None]:
    """Return (line frequency, contamination ratio) for whichever mains
    frequency dominates. Knowing 50 vs 60 Hz is itself provenance — it tells
    you what continent the recording came from when the notes don't."""
    if sfreq <= 120:
        candidates = [f for f in (50.0, 60.0) if f < sfreq / 2]
    else:
        candidates = [50.0, 60.0]
    if not candidates:
        return None, None

    best_f, best_ratio = None, 0.0
    for f in candidates:
        peak = _band_power(data_uv, sfreq, f - 1.0, f + 1.0).mean()
        side = (
            _band_power(data_uv, sfreq, f - 5.0, f - 2.0).mean()
            + _band_power(data_uv, sfreq, f + 2.0, f + 5.0).mean()
        ) / 2.0
        ratio = float(peak / side) if side > 0 else 0.0
        if ratio > best_ratio:
            best_f, best_ratio = f, ratio
    return best_f, round(best_ratio, 2)


def _blink_rate(data_uv: np.ndarray, ch_names: list[str],
                sfreq: float) -> float | None:
    """Count large slow frontal deflections. Blink rate is a cheap proxy for
    subject state — an unusually low rate on a long recording often means the
    frontal electrodes were not actually attached."""
    frontal = [i for i, n in enumerate(ch_names)
               if n.upper().startswith(("FP", "AF", "F7", "F8"))]
    if not frontal or data_uv.shape[1] < sfreq * 5:
        return None
    sig = data_uv[frontal].mean(axis=0)
    sos = sps.butter(4, [1.0, 10.0], btype="bandpass", fs=sfreq, output="sos")
    filt = sps.sosfiltfilt(sos, sig)
    thresh = 4.0 * np.median(np.abs(filt - np.median(filt))) * 1.4826
    if thresh <= 0:
        return None
    peaks, _ = sps.find_peaks(np.abs(filt), height=thresh,
                              distance=int(0.3 * sfreq))
    minutes = data_uv.shape[1] / sfreq / 60.0
    return round(len(peaks) / minutes, 1) if minutes > 0 else None


def analyze(data_uv: np.ndarray, ch_names: list[str], sfreq: float) -> QCReport:
    """Run every check against one recording. `data_uv` is (n_channels, n_times)."""
    n_ch, n_times = data_uv.shape
    duration = n_times / sfreq

    stds = data_uv.std(axis=1)
    ptps = data_uv.max(axis=1) - data_uv.min(axis=1)
    # Amplitude varies multiplicatively across the scalp, so score it in log
    # space; a channel twice the median should read the same whether the
    # montage sits at 10 uV or 100 uV.
    zs = _robust_z(np.log10(np.maximum(stds, 1e-6)))
    median_std = float(np.median(stds))

    total_power = _band_power(data_uv, sfreq, 0.1, min(45.0, sfreq / 2 - 1))
    low_power = _band_power(data_uv, sfreq, DRIFT_LO_HZ, DRIFT_HI_HZ)
    with np.errstate(divide="ignore", invalid="ignore"):
        drift = np.where(total_power > 0, low_power / total_power, 0.0)

    channels: list[ChannelReport] = []
    for i, name in enumerate(ch_names):
        flags: list[str] = []
        if stds[i] < FLAT_STD_UV:
            flags.append("flat")
        elif zs[i] > NOISY_ROBUST_Z and stds[i] > NOISY_ABS_RATIO * median_std:
            flags.append("noisy")
        if drift[i] > DRIFT_RATIO:
            flags.append("drift")
        channels.append(ChannelReport(
            name=name,
            std_uv=round(float(stds[i]), 2),
            ptp_uv=round(float(ptps[i]), 1),
            robust_z=round(float(zs[i]), 2),
            drift_ratio=round(float(drift[i]), 3),
            flags=flags,
        ))

    over = np.abs(data_uv) > ARTIFACT_AMP_UV
    artifact_fraction = float(over.any(axis=0).mean())

    line_freq, line_ratio = _detect_line(data_uv, sfreq)

    report = QCReport(
        n_channels=n_ch,
        sfreq_hz=float(sfreq),
        duration_s=round(duration, 2),
        line_freq_detected=line_freq,
        line_ratio=line_ratio,
        blink_rate_per_min=_blink_rate(data_uv, ch_names, sfreq),
        artifact_fraction=round(artifact_fraction, 4),
        retained_fraction=round(1.0 - artifact_fraction, 4),
        channels=channels,
    )
    report.flags = _session_flags(report)
    return report


def _session_flags(r: QCReport) -> list[str]:
    """Session-level problems worth a human's attention."""
    flags: list[str] = []
    if r.sfreq_hz < 250:
        flags.append(f"low_sampling_rate({r.sfreq_hz:g}Hz) — ERP timing marginal")
    if r.line_ratio and r.line_ratio > LINE_RATIO_WARN:
        flags.append(f"line_noise({r.line_freq_detected:g}Hz, {r.line_ratio}x baseline)")
    if r.artifact_fraction > 0.30:
        flags.append(f"high_artifact_load({r.artifact_fraction:.0%})")
    if r.blink_rate_per_min is not None and r.blink_rate_per_min < 2:
        flags.append("blink_rate_near_zero — check frontal electrode contact")
    bad = r.bad_channels
    if len(bad) > max(1, r.n_channels // 4):
        flags.append(f"many_bad_channels({len(bad)}/{r.n_channels})")
    return flags
