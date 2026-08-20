"""EDF is the format legacy archives are actually stored in.

The rest of the suite runs on FIF because MNE writes it natively, but a tool
that reports on old EEG archives and has never opened an EDF file is untested
where it counts. EDF also encodes differently — int16 samples with per-channel
physical/digital scaling, header fields fixed to 16 and 80 characters, events
in a separate annotation channel — so the read path can lose things FIF never
would.

These tests assert the conclusions survive that round trip, not just the bytes.
"""

import pytest

from eegprov import bids, inventory, paradigm, qc
from eegprov.synth import SynthSpec, write_session

pytest.importorskip("edfio", reason="EDF export needs edfio")


def _analyze(path):
    rec = inventory.read_header(path)
    data, ch_names, sfreq, desc, onsets = inventory.load_recording(path)
    return (rec,
            qc.analyze(data, ch_names, sfreq),
            paradigm.analyze(desc, onsets, data, ch_names, sfreq))


def _write(tmp_path, name, spec):
    path = write_session(tmp_path / name, spec)
    assert path.suffix == ".edf", "edfio present, so EDF must not fall back to FIF"
    return path


def test_edf_round_trip_preserves_header_and_events(tmp_path):
    spec = SynthSpec(n_seconds=60, seed=71)
    rec, _, _ = _analyze(_write(tmp_path, "sub-01_task-oddball.edf", spec))

    assert rec.readable
    assert rec.sfreq_hz == 250.0
    assert rec.n_channels == 16
    assert rec.duration_s == pytest.approx(60.0, abs=0.1)
    # Electrode names must survive EDF's 16-character label field intact —
    # a truncated "Pz" is a channel the P300 checks would silently miss.
    assert "Pz" in rec.ch_names and "Cz" in rec.ch_names
    assert set(rec.annotation_labels) == {"standard", "target"}


def test_edf_and_fif_reach_identical_conclusions(tmp_path):
    """int16 quantisation must not change any reported finding."""
    spec = dict(n_seconds=180, seed=1, flat_channels=["Pz"])

    edf = _analyze(_write(tmp_path, "cmp.edf", SynthSpec(**spec)))
    fif = _analyze(write_session(tmp_path / "cmp.raw.fif", SynthSpec(**spec)))

    for (_, q_a, p_a), (_, q_b, p_b) in [(edf, fif)]:
        assert q_a.bad_channels == q_b.bad_channels
        assert q_a.line_freq_detected == q_b.line_freq_detected
        assert q_a.retained_fraction == q_b.retained_fraction
        assert q_a.blink_rate_per_min == q_b.blink_rate_per_min
        assert p_a.paradigm == p_b.paradigm
        assert p_a.classes == p_b.classes
        assert p_a.evoked.response_detected == p_b.evoked.response_detected
        assert p_a.evoked.p_value == p_b.evoked.p_value


def test_dead_electrode_survives_edf_quantisation(tmp_path):
    """A flat channel is ~0.001 uV — far below one EDF quantisation step.
    It must still read as flat rather than as rounded-to-zero noise."""
    spec = SynthSpec(n_seconds=60, seed=72, flat_channels=["Pz"])
    _, q, _ = _analyze(_write(tmp_path, "sub-02_task-oddball.edf", spec))

    pz = next(c for c in q.channels if c.name == "Pz")
    assert "flat" in pz.flags
    assert q.bad_channels == ["Pz"]


def test_event_labels_are_plain_strings_not_numpy_scalars(tmp_path):
    """Readers hand back numpy string scalars, which serialise as
    np.str_('target') and break JSON round-tripping in the report."""
    spec = SynthSpec(n_seconds=60, seed=73)
    _, _, pd = _analyze(_write(tmp_path, "sub-03_task-oddball.edf", spec))

    for key in pd.classes:
        assert type(key) is str
    assert type(pd.rare_class) is str


def test_bids_draft_keeps_the_edf_extension(tmp_path):
    spec = SynthSpec(n_seconds=30, seed=74)
    path = _write(tmp_path, "sub-04_task-oddball_run-01.edf", spec)
    rec, q, _ = _analyze(path)
    bd = bids.draft(rec, q)
    assert bd.proposed_path.endswith("_eeg.edf")
