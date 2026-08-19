"""QC tests inject a known defect and assert the detector fires on it.

A quality checker that has never been shown a bad channel is not a quality
checker, so every test here builds the failure on purpose.
"""

import numpy as np
import pytest

from eegprov import qc
from eegprov.synth import SynthSpec, make_raw


def _analyze(spec: SynthSpec) -> qc.QCReport:
    raw = make_raw(spec)
    return qc.analyze(raw.get_data() * 1e6, list(raw.ch_names),
                      float(raw.info["sfreq"]))


def _flags_for(report: qc.QCReport, channel: str) -> list[str]:
    return next(c.flags for c in report.channels if c.name == channel)


def test_clean_recording_has_no_bad_channels():
    report = _analyze(SynthSpec(n_seconds=30, seed=11))
    assert report.bad_channels == []
    assert report.n_channels == 16
    assert report.duration_s == pytest.approx(30.0, abs=0.1)


def test_flat_channel_is_flagged():
    report = _analyze(SynthSpec(n_seconds=30, seed=12, flat_channels=["Pz"]))
    assert "flat" in _flags_for(report, "Pz")
    assert "Pz" in report.bad_channels
    # Its neighbours must not be dragged down with it.
    assert _flags_for(report, "P3") == []


def test_noisy_channel_is_flagged_and_neighbours_are_not():
    report = _analyze(SynthSpec(n_seconds=30, seed=13, noisy_channels=["T8"]))
    assert "noisy" in _flags_for(report, "T8")
    assert _flags_for(report, "T7") == []


def test_drifting_channel_is_flagged():
    report = _analyze(SynthSpec(n_seconds=60, seed=14, drift_channels=["M1"]))
    assert "drift" in _flags_for(report, "M1")


def test_line_frequency_is_detected_at_60hz():
    report = _analyze(SynthSpec(n_seconds=30, seed=15, line_freq=60.0,
                                line_amp_uv=12.0))
    assert report.line_freq_detected == 60.0
    assert report.line_ratio > qc.LINE_RATIO_WARN


def test_line_frequency_is_detected_at_50hz():
    """50 vs 60 Hz is itself provenance — it narrows down the recording site."""
    report = _analyze(SynthSpec(n_seconds=30, seed=16, line_freq=50.0,
                                line_amp_uv=12.0))
    assert report.line_freq_detected == 50.0


def test_blink_rate_tracks_the_injected_rate():
    busy = _analyze(SynthSpec(n_seconds=60, seed=17, blink_rate_per_min=20))
    quiet = _analyze(SynthSpec(n_seconds=60, seed=17, blink_rate_per_min=0))
    assert busy.blink_rate_per_min > quiet.blink_rate_per_min


def test_absent_blinks_flag_electrode_contact():
    report = _analyze(SynthSpec(n_seconds=60, seed=18, blink_rate_per_min=0))
    assert any("blink_rate_near_zero" in f for f in report.flags)


def test_low_sampling_rate_is_flagged_for_erp_work():
    report = _analyze(SynthSpec(n_seconds=30, seed=19, sfreq=128.0))
    assert any("low_sampling_rate" in f for f in report.flags)


def test_retained_fraction_falls_when_artifact_load_rises():
    clean = _analyze(SynthSpec(n_seconds=30, seed=20))
    dirty = _analyze(SynthSpec(n_seconds=30, seed=20,
                               noisy_channels=["T7", "T8", "O1", "O2"]))
    assert dirty.retained_fraction < clean.retained_fraction


def test_robust_z_survives_one_blown_channel():
    """Mean/std would be dragged by the outlier and hide everything else."""
    values = np.array([10.0, 11.0, 10.5, 9.8, 400.0])
    z = qc._robust_z(values)
    assert z[-1] > qc.NOISY_ROBUST_Z
    assert all(abs(v) < qc.NOISY_ROBUST_Z for v in z[:-1])
