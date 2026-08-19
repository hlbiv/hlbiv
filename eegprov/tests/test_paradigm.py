"""Paradigm-recovery tests.

The load-bearing case is `test_labels_without_a_response_are_flagged`: a
correct-looking events file whose labels the signal does not support. Trusting
those labels is how an archive produces a confident, wrong result.
"""

import numpy as np
import pytest

from eegprov import paradigm
from eegprov.synth import SynthSpec, make_raw


def _from_spec(spec: SynthSpec, with_signal: bool = True):
    raw = make_raw(spec)
    ann = raw.annotations
    data = raw.get_data() * 1e6
    return paradigm.analyze(
        list(ann.description),
        np.asarray(ann.onset),
        data if with_signal else None,
        list(raw.ch_names) if with_signal else None,
        float(raw.info["sfreq"]) if with_signal else None,
    )


def test_oddball_is_identified_from_event_structure():
    r = _from_spec(SynthSpec(n_seconds=120, seed=51, oddball_target_prob=0.2))
    assert r.paradigm == "oddball"
    assert r.rare_class == "target"
    assert r.class_proportions["target"] == pytest.approx(0.2, abs=0.08)
    assert r.median_isi_s == pytest.approx(1.0, abs=0.01)


def test_fixed_rate_presentation_is_recognised_as_regular():
    r = _from_spec(SynthSpec(n_seconds=60, seed=52), with_signal=False)
    assert r.isi_cv < paradigm.ISI_REGULAR_CV


def test_absent_events_read_as_resting_not_as_a_paradigm():
    r = _from_spec(SynthSpec(n_seconds=30, seed=53, oddball=False))
    assert r.paradigm == "resting_or_unmarked"
    assert r.n_events == 0
    assert r.rare_class is None


def test_fast_two_class_presentation_reads_as_a_speller():
    r = _from_spec(SynthSpec(n_seconds=60, seed=54, oddball_isi_s=0.125,
                             oddball_target_prob=0.16), with_signal=False)
    assert r.paradigm == "p300_speller"


def test_evoked_check_finds_the_p300_at_the_right_latency():
    r = _from_spec(SynthSpec(n_seconds=180, seed=55, p300_amp_uv=8.0))
    assert r.evoked.response_detected
    assert r.evoked.peak_latency_s == pytest.approx(0.30, abs=0.06)
    assert r.evoked.p_value < paradigm.PERMUTATION_ALPHA
    # It should average over midline sites, not the whole montage.
    assert "Pz" in r.evoked.channel_basis


def test_labels_without_a_response_are_flagged():
    """Events present and well-formed, but no evoked response behind them —
    a lost or mis-mapped code->condition table looks exactly like this."""
    r = _from_spec(SynthSpec(n_seconds=180, seed=56, p300_amp_uv=0.0))
    assert not r.evoked.response_detected
    assert "NO time-locked response" in r.evoked.note
    assert any("labels unverified by signal" in f for f in r.flags)


def test_lowpass_removes_mains_phase_locking_from_the_average():
    """A 1.0 s ISI is a whole multiple of the 60 Hz mains period, so line noise
    lands at identical phase on every onset and survives averaging while real
    background cancels. Unfiltered, that alone produces an "evoked response"
    that also beats a random-onset null — the permutation test cannot rescue
    it, only the low-pass can. This test pins the reason the filter is there."""
    spec = SynthSpec(n_seconds=180, seed=56, p300_amp_uv=0.0,
                     oddball_isi_s=1.0, line_freq=60.0, line_amp_uv=8.0)
    raw = make_raw(spec)
    data = raw.get_data() * 1e6
    ch_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])
    desc = np.array(raw.annotations.description)
    onsets = np.asarray(raw.annotations.onset)[desc == "target"]

    picked = data[[ch_names.index("Pz")]]
    n_times = int(round((paradigm.EPOCH[1] - paradigm.EPOCH[0]) * sfreq))
    times = np.arange(n_times) / sfreq + paradigm.EPOCH[0]
    win = ((times >= paradigm.P300_WINDOW[0]) & (times <= paradigm.P300_WINDOW[1]))

    unfiltered, _ = paradigm._window_peak(
        paradigm._epoch(picked, onsets, sfreq), times, win)
    filtered, _ = paradigm._window_peak(
        paradigm._epoch(paradigm._lowpass(picked, sfreq), onsets, sfreq), times, win)

    assert unfiltered > 2 * filtered


def test_permutation_test_rejects_a_response_on_signal_free_data():
    """With the filter in place, background alone must not read as a P300 —
    even though peak-picking over a window is a positively biased statistic."""
    r = _from_spec(SynthSpec(n_seconds=180, seed=56, p300_amp_uv=0.0))
    assert not r.evoked.response_detected
    assert r.evoked.p_value > paradigm.PERMUTATION_ALPHA


def test_a_dead_p300_electrode_dilutes_label_verification():
    """Why the paradigm and QC sections have to be read together: averaging a
    dead Pz into the basis drags the evoked estimate toward zero even though
    the response is present in the data. A weak verification result is not
    evidence about the labels until the channel flags are checked."""
    intact = _from_spec(SynthSpec(n_seconds=180, seed=61, p300_amp_uv=8.0))
    dead = _from_spec(SynthSpec(n_seconds=180, seed=61, p300_amp_uv=8.0,
                                flat_channels=["Pz"]))
    assert intact.evoked.peak_amplitude_uv > dead.evoked.peak_amplitude_uv


def test_permutation_p_value_is_deterministic():
    """A seeded null keeps the report reproducible across runs."""
    a = _from_spec(SynthSpec(n_seconds=120, seed=59, p300_amp_uv=6.0))
    b = _from_spec(SynthSpec(n_seconds=120, seed=59, p300_amp_uv=6.0))
    assert a.evoked.p_value == b.evoked.p_value


def test_p_value_is_never_reported_as_zero():
    """A finite null cannot prove impossibility; (n+1)/(N+1) keeps it honest."""
    r = _from_spec(SynthSpec(n_seconds=180, seed=60, p300_amp_uv=25.0))
    assert r.evoked.response_detected
    assert r.evoked.p_value > 0


def test_too_few_trials_for_an_erp_is_flagged():
    r = _from_spec(SynthSpec(n_seconds=40, seed=57, oddball_isi_s=2.0,
                             oddball_target_prob=0.2))
    assert r.classes.get("target", 0) < paradigm.MIN_TRIALS_FOR_ERP
    assert any("below the ~20 minimum" in f for f in r.flags)


def test_single_label_reports_a_lost_mapping():
    r = paradigm.analyze(["stim"] * 50, np.arange(50) * 1.0)
    assert r.paradigm == "single_condition_or_lost_mapping"
    assert any("code->condition mapping" in f for f in r.flags)


def test_irregular_intervals_read_as_self_paced():
    rng = np.random.default_rng(0)
    onsets = np.cumsum(rng.uniform(0.4, 3.0, size=60))
    labels = ["a" if i % 2 else "b" for i in range(60)]
    r = paradigm.analyze(labels, onsets)
    assert r.paradigm == "self_paced_or_response_locked"
    assert r.isi_cv > paradigm.ISI_REGULAR_CV


def test_epochs_running_past_the_recording_end_are_dropped_not_padded():
    raw = make_raw(SynthSpec(n_seconds=20, seed=58))
    data = raw.get_data() * 1e6
    sfreq = float(raw.info["sfreq"])
    # An onset one second before the end cannot hold a 0.8 s post-stimulus epoch.
    onsets = np.array([5.0, 10.0, 19.9])
    check = paradigm.check_evoked(data, list(raw.ch_names), sfreq, onsets, "target")
    assert check.n_trials == 2
