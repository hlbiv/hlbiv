"""Draft BIDS-EEG structure for discovered recordings.

This produces a PLAN, not a converted dataset. Entities that could not be
recovered stay visibly empty, because a BIDS tree built on invented subject
IDs is worse than no BIDS tree at all — it looks authoritative and is wrong.

BIDS-EEG requires four sidecar fields that a raw file header usually cannot
supply on its own (notably TaskName and EEGReference). Naming them as
blockers is the useful output: it turns "someone should organise this
archive" into a specific, short list of questions for whoever ran the study.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .inventory import Recording
from .qc import QCReport

# Required by the BIDS-EEG spec for every recording.
REQUIRED_SIDECAR = ("TaskName", "SamplingFrequency", "PowerLineFrequency",
                    "EEGReference", "SoftwareFilters")

# Fields no file header carries — a human who was there has to answer these.
HUMAN_ONLY_FIELDS = {
    "EEGReference": "Which electrode(s) served as reference? (e.g. 'mastoids', "
                    "'Cz', 'average'). Not stored in most raw formats.",
    "EEGPlacementScheme": "Which montage/cap layout? (e.g. '10-20', '10-10')",
    "Manufacturer": "Amplifier manufacturer and model.",
}


@dataclass
class BidsDraft:
    source_path: str
    proposed_path: str | None
    sidecar: dict
    channels_tsv_rows: list[dict] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    questions_for_human: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.proposed_path is not None and not self.blockers

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ready"] = self.ready
        return d


def _bids_label(value: str | None) -> str | None:
    """BIDS labels are alphanumeric only."""
    if value is None:
        return None
    cleaned = "".join(ch for ch in value if ch.isalnum())
    return cleaned or None


def _proposed_path(rec: Recording, datatype: str = "eeg") -> tuple[str | None, list[str]]:
    """Build the BIDS relative path, or explain why it cannot be built."""
    blockers: list[str] = []
    sub = _bids_label(rec.entities.subject)
    task = _bids_label(rec.entities.task)
    if sub is None:
        blockers.append("subject label unrecoverable from filename or header")
    if task is None:
        blockers.append("task label unrecoverable — BIDS requires TaskName")
    if blockers:
        return None, blockers

    ses = _bids_label(rec.entities.session)
    run = _bids_label(rec.entities.run)

    parts = [f"sub-{sub}"]
    if ses:
        parts.append(f"ses-{ses}")
    parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{int(run):02d}" if run.isdigit() else f"run-{run}")
    stem = "_".join(parts) + f"_{datatype}"

    dirs = [f"sub-{sub}"]
    if ses:
        dirs.append(f"ses-{ses}")
    dirs.append(datatype)
    return "/".join(dirs) + "/" + stem + _extension(rec.filename), []


def _extension(filename: str) -> str:
    """Preserve the full extension. MNE's FIF files end in a compound
    `.raw.fif`, and truncating that to `.fif` produces a name MNE will not
    write back out."""
    lower = filename.lower()
    for compound in (".raw.fif", ".raw.fif.gz", ".nii.gz"):
        if lower.endswith(compound):
            return filename[-len(compound):]
    dot = filename.rfind(".")
    return filename[dot:] if dot > 0 else ""


def _channels_rows(rec: Recording, qc: QCReport | None) -> list[dict]:
    """channels.tsv — carries the per-channel status QC discovered.

    'status' is where an artifact-rejection pass becomes durable metadata
    instead of a number in someone's notebook."""
    by_name = {c.name: c for c in qc.channels} if qc else {}
    rows = []
    for name in rec.ch_names:
        c = by_name.get(name)
        rows.append({
            "name": name,
            "type": "EEG",
            "units": "uV",
            "sampling_frequency": rec.sfreq_hz,
            "low_cutoff": rec.highpass_hz if rec.highpass_hz else "n/a",
            "high_cutoff": rec.lowpass_hz if rec.lowpass_hz else "n/a",
            "status": ("bad" if c and c.flags else "good") if c else "n/a",
            "status_description": ",".join(c.flags) if c and c.flags else "n/a",
        })
    return rows


def draft(rec: Recording, qc: QCReport | None = None) -> BidsDraft:
    """Draft the BIDS entry for one recording."""
    if not rec.readable:
        return BidsDraft(
            source_path=rec.path,
            proposed_path=None,
            sidecar={},
            blockers=[f"unreadable: {rec.reason}"],
        )

    path, blockers = _proposed_path(rec)

    sidecar: dict = {
        "TaskName": _bids_label(rec.entities.task),
        "SamplingFrequency": rec.sfreq_hz,
        "EEGChannelCount": rec.n_channels,
        "RecordingDuration": rec.duration_s,
        "RecordingType": "continuous",
        "SoftwareFilters": _software_filters(rec),
        "PowerLineFrequency": (qc.line_freq_detected if qc and qc.line_freq_detected
                               else None),
        "EEGReference": None,
    }

    questions: list[str] = []
    for field_name, prompt in HUMAN_ONLY_FIELDS.items():
        if not sidecar.get(field_name):
            questions.append(f"{field_name}: {prompt}")

    if sidecar["PowerLineFrequency"] is None:
        questions.append(
            "PowerLineFrequency: not detectable from signal (run the QC pass, "
            "or supply 50/60 from the recording site)."
        )
    else:
        sidecar["_PowerLineFrequency_provenance"] = "inferred from signal spectrum"

    missing_required = [f for f in REQUIRED_SIDECAR if not sidecar.get(f)]
    if missing_required:
        blockers = blockers + [
            f"missing BIDS-required sidecar field(s): {', '.join(missing_required)}"
        ]

    if rec.n_annotations == 0:
        questions.append(
            "events: no annotations in file — supply the event/marker file or "
            "the code->condition mapping, or the paradigm cannot be reconstructed."
        )

    return BidsDraft(
        source_path=rec.path,
        proposed_path=path,
        sidecar=sidecar,
        channels_tsv_rows=_channels_rows(rec, qc),
        blockers=blockers,
        questions_for_human=questions,
    )


def _software_filters(rec: Recording) -> dict | str:
    """Report the acquisition filters the header knows about."""
    filters: dict = {}
    if rec.highpass_hz:
        filters["HighPass"] = {"CutoffFrequency": rec.highpass_hz}
    if rec.lowpass_hz:
        filters["LowPass"] = {"CutoffFrequency": rec.lowpass_hz}
    return filters or "n/a"


def summarize(drafts: list[BidsDraft]) -> dict:
    """Archive-level view: what converts cleanly, what needs a human."""
    ready = [d for d in drafts if d.ready]
    blocked = [d for d in drafts if not d.ready]

    reasons: dict[str, int] = {}
    for d in blocked:
        for b in d.blockers:
            key = b.split(":")[0].split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1

    all_questions: dict[str, int] = {}
    for d in drafts:
        for q in d.questions_for_human:
            key = q.split(":")[0]
            all_questions[key] = all_questions.get(key, 0) + 1

    return {
        "total": len(drafts),
        "ready": len(ready),
        "blocked": len(blocked),
        "blocker_counts": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "open_questions": dict(sorted(all_questions.items(), key=lambda kv: -kv[1])),
    }
