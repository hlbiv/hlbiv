"""BIDS drafting and end-to-end CLI tests.

The property that matters most: the drafter never invents an entity to make a
file convert. A confidently-wrong BIDS tree is worse than an honest blocker.
"""

import json

import pytest

from eegprov import bids, inventory, qc
from eegprov.cli import main
from eegprov.synth import SynthSpec, write_session


def _draft_for(tmp_path, filename, spec=None):
    path = write_session(tmp_path / filename, spec or SynthSpec(n_seconds=15, seed=41))
    rec = inventory.read_header(path)
    data, ch_names, sfreq = inventory.load_data_uv(path)
    return rec, bids.draft(rec, qc.analyze(data, ch_names, sfreq))


def test_well_named_recording_gets_a_proposed_path(tmp_path):
    _, bd = _draft_for(tmp_path, "sub-01_ses-02_task-oddball_run-03.raw.fif")
    assert bd.proposed_path == (
        "sub-01/ses-02/eeg/sub-01_ses-02_task-oddball_run-03_eeg.raw.fif"
    )


def test_legacy_name_still_resolves_when_task_is_a_keyword(tmp_path):
    _, bd = _draft_for(tmp_path, "s04_run2_p300_final.raw.fif")
    assert bd.proposed_path is not None
    assert bd.proposed_path.startswith("sub-04/eeg/sub-04_task-p300_run-02")


def test_unresolvable_name_is_blocked_not_guessed(tmp_path):
    _, bd = _draft_for(tmp_path, "recording_copy_2.raw.fif")
    assert bd.proposed_path is None
    assert not bd.ready
    assert any("subject label unrecoverable" in b for b in bd.blockers)
    assert any("task label unrecoverable" in b for b in bd.blockers)


def test_human_only_fields_become_questions_not_defaults(tmp_path):
    """EEGReference is required by BIDS and absent from every raw header.
    Inventing 'average' here would silently corrupt every downstream analysis."""
    _, bd = _draft_for(tmp_path, "sub-01_task-oddball.raw.fif")
    assert bd.sidecar["EEGReference"] is None
    assert any(q.startswith("EEGReference") for q in bd.questions_for_human)
    assert any("EEGReference" in b for b in bd.blockers)


def test_detected_line_frequency_is_marked_as_inferred(tmp_path):
    _, bd = _draft_for(tmp_path, "sub-01_task-oddball.raw.fif",
                       SynthSpec(n_seconds=20, seed=42, line_amp_uv=12.0))
    assert bd.sidecar["PowerLineFrequency"] == 60.0
    assert bd.sidecar["_PowerLineFrequency_provenance"] == "inferred from signal spectrum"


def test_channels_tsv_carries_qc_status(tmp_path):
    _, bd = _draft_for(tmp_path, "sub-01_task-oddball.raw.fif",
                       SynthSpec(n_seconds=20, seed=43, flat_channels=["Pz"]))
    rows = {r["name"]: r for r in bd.channels_tsv_rows}
    assert rows["Pz"]["status"] == "bad"
    assert "flat" in rows["Pz"]["status_description"]
    assert rows["Cz"]["status"] == "good"


def test_missing_events_become_a_question(tmp_path):
    _, bd = _draft_for(tmp_path, "sub-01_task-rest.raw.fif",
                       SynthSpec(n_seconds=15, seed=44, oddball=False))
    assert any(q.startswith("events") for q in bd.questions_for_human)


def test_summarize_counts_ready_and_blocked(tmp_path):
    drafts = [
        _draft_for(tmp_path, "sub-01_task-oddball.raw.fif")[1],
        _draft_for(tmp_path, "recording_copy_2.raw.fif")[1],
    ]
    summary = bids.summarize(drafts)
    assert summary["total"] == 2
    assert summary["blocked"] >= 1
    assert summary["open_questions"]


def test_demo_then_archive_runs_end_to_end(tmp_path, capsys):
    archive_dir = tmp_path / "archive"
    out_dir = tmp_path / "report"

    assert main(["demo", str(archive_dir)]) == 0
    assert main(["scan", str(archive_dir)]) == 0
    assert main(["archive", str(archive_dir), "-o", str(out_dir)]) == 0

    inventory_json = json.loads((out_dir / "inventory.json").read_text())
    assert inventory_json["summary"]["total"] == 6  # 5 readable + 1 .dat
    assert (out_dir / "inventory.md").exists()
    assert len(list((out_dir / "cards").glob("*.md"))) == 6

    text = capsys.readouterr().out
    assert "Archive inventory" in text
    assert "Questions to put to whoever ran these studies" in text


def test_archive_surfaces_heterogeneous_acquisition(tmp_path, capsys):
    """Mixed sampling rates across an archive block pooling — say so up front."""
    archive_dir = tmp_path / "archive"
    main(["demo", str(archive_dir)])
    capsys.readouterr()
    main(["archive", str(archive_dir)])
    assert "Heterogeneous acquisition" in capsys.readouterr().out


def test_scan_on_empty_directory_exits_nonzero(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["scan", str(empty)]) == 1
