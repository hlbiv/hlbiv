"""Entity inference tests, drawn from the filename styles real archives use."""

import pytest

from eegprov.inventory import infer_entities, read_header
from eegprov.synth import SynthSpec, write_session


@pytest.mark.parametrize("filename,subject,session,task,run", [
    ("sub-01_ses-02_task-oddball_run-03_eeg.edf", "01", "02", "oddball", "03"),
    ("s04_run2_p300_final.edf", "04", None, "p300", "2"),
    ("subject_12_day3_resting.edf", "12", "3", "rest", None),
    ("P07_speller_block4.edf", "07", None, "p300speller", "4"),
    ("s02_session1_ssvep.edf", "02", "1", "ssvep", None),
    ("recording_copy_2.edf", None, None, None, None),
])
def test_entity_inference(filename, subject, session, task, run):
    ent = infer_entities(filename)
    assert ent.subject == subject
    assert ent.session == session
    assert ent.task == task
    assert ent.run == run


def test_every_inferred_value_records_its_source():
    """An inference with no stated basis is indistinguishable from a guess."""
    ent = infer_entities("s04_run2_p300_final.edf")
    for name in ("subject", "task", "run"):
        assert name in ent.sources
        assert ent.sources[name].startswith("filename:")


def test_unresolved_entities_are_listed_not_filled_in():
    ent = infer_entities("recording_copy_2.edf")
    assert set(ent.unresolved) == {"subject", "session", "task", "run"}


def test_bids_names_are_not_reparsed_by_loose_patterns():
    """`sub-01` must not also match the bare `s(\\d+)` fallback."""
    ent = infer_entities("sub-01_ses-02_task-rest_run-01_eeg.edf")
    assert ent.subject == "01"
    assert ent.sources["subject"] == "filename:bids"


def test_provenance_labels_are_human_readable():
    """The source string lands on a card a researcher reads."""
    ent = infer_entities("s04_run2_p300_final.edf")
    assert ent.sources["subject"] == "filename:s-prefix"
    assert ent.sources["run"] == "filename:run-prefix"
    assert ent.sources["task"] == "filename:keyword(p300)"


def test_parent_directory_is_a_fallback_only_for_subject_and_session():
    """Ambient directory text must not manufacture a task or run."""
    ent = infer_entities("recording.edf", parent="sub-09")
    assert ent.subject == "09"
    assert ent.sources["subject"] == "parentdir:bids"
    assert ent.task is None and ent.run is None


def test_read_header_extracts_facts_and_flags_missing_events(tmp_path):
    path = write_session(tmp_path / "sub-01_task-rest_eeg.raw.fif",
                         SynthSpec(n_seconds=20, seed=31, oddball=False))
    rec = read_header(path)
    assert rec.readable
    assert rec.sfreq_hz == 250.0
    assert rec.n_channels == 16
    assert rec.duration_s == pytest.approx(20.0, abs=0.1)
    assert rec.n_annotations == 0
    assert any("no event annotations" in w for w in rec.warnings)


def test_read_header_reports_annotations_when_present(tmp_path):
    path = write_session(tmp_path / "sub-02_task-oddball_eeg.raw.fif",
                         SynthSpec(n_seconds=30, seed=32))
    rec = read_header(path)
    assert rec.n_annotations > 0
    assert set(rec.annotation_labels) == {"standard", "target"}


def test_suspect_filename_tokens_raise_a_provenance_warning(tmp_path):
    path = write_session(tmp_path / "s03_task-rest_final_FIXED.raw.fif",
                         SynthSpec(n_seconds=10, seed=33))
    rec = read_header(path)
    assert any("provenance uncertain" in w for w in rec.warnings)


def test_unreadable_format_is_reported_not_skipped(tmp_path):
    path = tmp_path / "s06_speller_block1.dat"
    path.write_bytes(b"placeholder")
    rec = read_header(path)
    assert not rec.readable
    assert "BCI2000" in rec.reason
    # Entities are still inferred — an unreadable file is still inventory.
    assert rec.entities.subject == "06"
