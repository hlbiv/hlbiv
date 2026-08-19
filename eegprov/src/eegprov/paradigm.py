"""Recover the experimental paradigm from event timing, and check that the
event labels are actually supported by the signal.

Two distinct jobs, and the second is the one that matters most for a legacy
archive:

1. **What paradigm was this?** Event onsets and labels carry the answer even
   when the protocol document is gone. A two-class sequence at a fixed
   interval with one class at ~20% is an oddball run; nothing else looks like
   that.

2. **Do the labels mean what they say?** An events file can survive while its
   code->condition mapping is lost, mis-mapped, or offset by a trigger delay.
   Averaging the rare class and looking for a time-locked response tests the
   labels against the data instead of trusting them. A "target" class with no
   evoked response is the single most expensive thing to discover late.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import signal as sps

# Below this, an ERP average is too noisy to trust. 20 is the common floor
# quoted for P300 work; 30-40 is where estimates actually stabilise.
MIN_TRIALS_FOR_ERP = 20
STABLE_TRIALS_FOR_ERP = 30
# A class rarer than this in a two-class design is the "oddball".
ODDBALL_MAX_RARE_PROB = 0.35
# Coefficient of variation of the inter-stimulus interval below which the
# presentation counts as fixed-rate rather than self-paced.
ISI_REGULAR_CV = 0.15

# Where a P300 lives, in seconds post-stimulus.
P300_WINDOW = (0.25, 0.50)
EPOCH = (-0.2, 0.8)
BASELINE = (-0.2, 0.0)

# Low-pass before averaging. This is standard ERP practice, and skipping it is
# actively dangerous for exactly the designs this tool targets: a fixed
# inter-stimulus interval that is a whole multiple of the mains period (a 1.0 s
# ISI against 60 Hz, say) phase-locks line noise to every single onset, so the
# hum survives averaging while genuine background cancels. The result is a
# clean-looking "evoked response" made entirely of mains interference — and it
# beats a random-onset null, so the permutation test alone will not catch it.
# The P300 is under 10 Hz, so 30 Hz costs nothing.
ERP_LOWPASS_HZ = 30.0
# Midline sites carry the P300 most strongly; fall back to all channels.
P300_SITES = ("Pz", "Cz", "CPz", "POz")

# Detection is a permutation test, NOT a threshold on SNR.
#
# Taking the maximum over a search window is a biased statistic: the peak of a
# noisy average is always positive, and on ~30 trials of 1/f background it
# routinely clears 2x the baseline standard deviation. An SNR threshold
# therefore "finds" a P300 in data containing none — which is the exact failure
# this check exists to prevent.
#
# The null is built by re-running the identical procedure — same trial count,
# same epoching, same peak-picking — at random onsets. Because the bias is
# present in the null too, it cancels.
N_PERMUTATIONS = 200
PERMUTATION_ALPHA = 0.05
PERMUTATION_SEED = 0


@dataclass
class EvokedCheck:
    channel_basis: list[str]
    n_trials: int
    peak_amplitude_uv: float | None
    peak_latency_s: float | None
    baseline_noise_uv: float | None
    snr: float | None
    p_value: float | None
    response_detected: bool
    note: str


@dataclass
class ParadigmReport:
    paradigm: str
    confidence: str
    n_events: int
    classes: dict[str, int]
    class_proportions: dict[str, float]
    median_isi_s: float | None
    isi_cv: float | None
    rare_class: str | None
    evoked: EvokedCheck | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _isi_stats(onsets: np.ndarray) -> tuple[float | None, float | None]:
    if len(onsets) < 3:
        return None, None
    isis = np.diff(np.sort(onsets))
    isis = isis[isis > 0]
    if len(isis) < 2:
        return None, None
    median = float(np.median(isis))
    cv = float(np.std(isis) / median) if median > 0 else None
    return round(median, 4), (round(cv, 3) if cv is not None else None)


def _classify(classes: dict[str, int], median_isi: float | None,
              isi_cv: float | None) -> tuple[str, str, str | None]:
    """Name the paradigm from event structure alone."""
    total = sum(classes.values())
    if total == 0:
        return ("resting_or_unmarked",
                "high — no events present, so nothing is time-locked", None)

    props = {k: v / total for k, v in classes.items()}
    rare = min(props, key=props.get) if len(props) > 1 else None
    regular = isi_cv is not None and isi_cv < ISI_REGULAR_CV

    if len(classes) == 2 and rare and props[rare] <= ODDBALL_MAX_RARE_PROB:
        if median_isi and median_isi < 0.4 and regular:
            return ("p300_speller",
                    "medium — two-class, fast fixed-rate presentation is "
                    "consistent with a P300 speller matrix", rare)
        return ("oddball",
                "high — two classes at a fixed interval with a rare class at "
                f"{props[rare]:.0%} is the oddball signature", rare)

    if median_isi and median_isi < 0.1 and regular:
        return ("ssvep_or_high_rate_stimulation",
                "medium — sub-100 ms fixed-rate stimulation", rare)

    if len(classes) == 1:
        return ("single_condition_or_lost_mapping",
                "low — one event label only; if the study had conditions, the "
                "code->condition mapping did not survive", None)

    if not regular:
        return ("self_paced_or_response_locked",
                "low — irregular intervals suggest events follow participant "
                "responses rather than a fixed schedule", rare)

    return ("multi_condition", f"low — {len(classes)} classes at a fixed "
            "interval; paradigm not identifiable from timing alone", rare)


def _epoch(data_uv: np.ndarray, onsets_s: np.ndarray,
           sfreq: float) -> np.ndarray | None:
    """Cut baseline-corrected epochs. Returns (n_trials, n_channels, n_times)."""
    pre = int(round(EPOCH[0] * sfreq))
    post = int(round(EPOCH[1] * sfreq))
    n_times = post - pre
    base_n = int(round((BASELINE[1] - BASELINE[0]) * sfreq))

    epochs = []
    for onset in onsets_s:
        start = int(round(onset * sfreq)) + pre
        stop = start + n_times
        if start < 0 or stop > data_uv.shape[1]:
            continue
        seg = data_uv[:, start:stop]
        if base_n > 0:
            seg = seg - seg[:, :base_n].mean(axis=1, keepdims=True)
        epochs.append(seg)
    return np.stack(epochs) if epochs else None


def _lowpass(data_uv: np.ndarray, sfreq: float) -> np.ndarray:
    """Zero-phase low-pass ahead of averaging. See ERP_LOWPASS_HZ."""
    nyquist = sfreq / 2.0
    cutoff = min(ERP_LOWPASS_HZ, nyquist * 0.9)
    if cutoff <= 0 or cutoff >= nyquist:
        return data_uv
    sos = sps.butter(4, cutoff, btype="lowpass", fs=sfreq, output="sos")
    return sps.sosfiltfilt(sos, data_uv, axis=-1)


def _window_peak(epochs: np.ndarray, times: np.ndarray,
                 win: np.ndarray) -> tuple[float, int]:
    """Average epochs across trials and channels, then take the windowed peak."""
    evoked = epochs.mean(axis=(0, 1))
    i = int(np.argmax(evoked[win]))
    return float(evoked[win][i]), i


def check_evoked(data_uv: np.ndarray, ch_names: list[str], sfreq: float,
                 onsets_s: np.ndarray, label: str,
                 n_permutations: int = N_PERMUTATIONS) -> EvokedCheck:
    """Average one event class and test whether a time-locked P300-range
    response is present at those onsets specifically.

    Significance comes from a permutation null over random onsets, so the
    peak-picking bias applies equally to observation and null and cancels."""
    sites = [c for c in P300_SITES if c in ch_names]
    basis = sites or list(ch_names)
    idx = [ch_names.index(c) for c in basis]
    picked = _lowpass(data_uv[idx], sfreq)

    epochs = _epoch(picked, onsets_s, sfreq)
    if epochs is None or len(epochs) == 0:
        return EvokedCheck(basis, 0, None, None, None, None, None, False,
                           "no epochs fit inside the recording")

    n_times = epochs.shape[-1]
    times = np.arange(n_times) / sfreq + EPOCH[0]
    win = (times >= P300_WINDOW[0]) & (times <= P300_WINDOW[1])
    if not win.any():
        return EvokedCheck(basis, len(epochs), None, None, None, None, None,
                           False, "epoch too short to contain the P300 window")

    peak_amp, peak_i = _window_peak(epochs, times, win)
    peak_lat = float(times[win][peak_i])

    evoked = epochs.mean(axis=(0, 1))
    base_mask = times < BASELINE[1]
    noise = float(np.std(evoked[base_mask])) if base_mask.any() else 0.0
    snr = float(peak_amp / noise) if noise > 0 else None

    p_value = _permutation_p(picked, sfreq, len(epochs), times, win,
                             peak_amp, n_permutations)
    detected = bool(p_value is not None and p_value < PERMUTATION_ALPHA
                    and peak_amp > 0)

    if p_value is None:
        note = (f"response for '{label}' not assessable — recording too short "
                "to build a permutation null")
    elif detected:
        note = (f"time-locked positivity at {peak_lat * 1000:.0f} ms "
                f"({peak_amp:.1f} uV, p={p_value:.3f} vs random onsets) — "
                f"consistent with the '{label}' label")
    else:
        note = (f"NO time-locked response for '{label}' (peak {peak_amp:.1f} uV "
                f"is within chance, p={p_value:.3f}). Either the label does not "
                "mean what it says, the code->condition mapping is wrong, or the "
                "trigger timing is offset.")

    return EvokedCheck(basis, len(epochs), round(peak_amp, 2), round(peak_lat, 3),
                       round(noise, 3), round(snr, 2) if snr else None,
                       round(p_value, 4) if p_value is not None else None,
                       detected, note)


def _permutation_p(picked: np.ndarray, sfreq: float, n_trials: int,
                   times: np.ndarray, win: np.ndarray, observed: float,
                   n_permutations: int) -> float | None:
    """Fraction of random-onset averages whose windowed peak matches or beats
    the observed one."""
    pre = int(round(EPOCH[0] * sfreq))
    post = int(round(EPOCH[1] * sfreq))
    lo, hi = -pre, picked.shape[1] - post
    if hi <= lo or n_permutations < 1:
        return None

    rng = np.random.default_rng(PERMUTATION_SEED)
    n_ge = 0
    for _ in range(n_permutations):
        starts = rng.integers(lo, hi, size=n_trials) / sfreq
        null_epochs = _epoch(picked, starts, sfreq)
        if null_epochs is None or len(null_epochs) == 0:
            continue
        null_peak, _ = _window_peak(null_epochs, times, win)
        if null_peak >= observed:
            n_ge += 1
    # +1 in both terms: an observed value can never be reported as impossible
    # under a finite null.
    return (n_ge + 1) / (n_permutations + 1)


def analyze(descriptions: list[str], onsets_s: np.ndarray,
            data_uv: np.ndarray | None = None,
            ch_names: list[str] | None = None,
            sfreq: float | None = None) -> ParadigmReport:
    """Infer the paradigm, and verify the rare class against the signal."""
    onsets_s = np.asarray(onsets_s, dtype=float)
    classes: dict[str, int] = {}
    for d in descriptions:
        classes[d] = classes.get(d, 0) + 1

    median_isi, isi_cv = _isi_stats(onsets_s)
    paradigm, confidence, rare = _classify(classes, median_isi, isi_cv)
    total = sum(classes.values()) or 1

    report = ParadigmReport(
        paradigm=paradigm,
        confidence=confidence,
        n_events=len(descriptions),
        classes=classes,
        class_proportions={k: round(v / total, 3) for k, v in classes.items()},
        median_isi_s=median_isi,
        isi_cv=isi_cv,
        rare_class=rare,
    )

    if rare and data_uv is not None and ch_names and sfreq:
        rare_onsets = onsets_s[np.array(descriptions) == rare]
        report.evoked = check_evoked(data_uv, ch_names, sfreq, rare_onsets, rare)

    report.flags = _flags(report)
    return report


def _flags(r: ParadigmReport) -> list[str]:
    flags: list[str] = []
    if r.rare_class:
        n = r.classes[r.rare_class]
        if n < MIN_TRIALS_FOR_ERP:
            flags.append(
                f"only {n} '{r.rare_class}' trials — below the ~{MIN_TRIALS_FOR_ERP} "
                "minimum for a usable ERP average; this session cannot carry an "
                "ERP analysis on its own"
            )
        elif n < STABLE_TRIALS_FOR_ERP:
            flags.append(
                f"{n} '{r.rare_class}' trials — usable but below the "
                f"~{STABLE_TRIALS_FOR_ERP} where amplitude estimates stabilise"
            )
    if r.paradigm == "single_condition_or_lost_mapping":
        flags.append(
            "one event label only — recover the code->condition mapping from "
            "the acquisition script or the paradigm cannot be reconstructed"
        )
    if r.evoked and not r.evoked.response_detected and r.evoked.n_trials >= MIN_TRIALS_FOR_ERP:
        flags.append(
            f"labels unverified by signal — {r.evoked.n_trials} '{r.rare_class}' "
            "trials averaged with no time-locked response"
        )
    return flags
