"""Discover EEG recordings on disk and recover what can be recovered.

The premise: a legacy archive is a folder of files like
`s04_run2_final_FIXED.edf` whose headers say sampling rate and channel names
but say nothing about subject, session, paradigm, or condition. This module
reads what the header knows and infers what the filename implies, keeping the
two strictly separate — inferred values are labelled as inferred, and anything
neither source supports is reported as unresolved rather than invented.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mne
import numpy as np

# Extension -> MNE reader. Formats we can detect but not read are listed in
# UNREADABLE so an inventory still reports them instead of pretending the
# files are not there.
READERS = {
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".gdf": mne.io.read_raw_gdf,
    ".fif": mne.io.read_raw_fif,
    ".set": mne.io.read_raw_eeglab,
    ".vhdr": mne.io.read_raw_brainvision,
    ".cnt": mne.io.read_raw_cnt,
    ".nxe": mne.io.read_raw_nicolet,
}

UNREADABLE = {
    ".dat": "BCI2000 .dat — no MNE reader; convert with BCI2000 tools first",
    ".mat": "MATLAB container — structure is lab-specific, needs a custom loader",
    ".txt": "OpenBCI text export — needs column mapping before it can be read",
    ".csv": "delimited export — needs column mapping before it can be read",
}

# `\b` is useless here: underscore is a word character, so `\bs(\d+)\b` never
# matches `s04_run2` — the boundary after "04" fails against "_". Filenames are
# mostly underscore-delimited, so token starts are matched explicitly instead.
_START = r"(?:^|[^A-Za-z0-9])"

# Each pattern carries a plain-language label, because the label lands on the
# provenance card a human reads. "s-prefix" tells them how the value was
# derived; a raw regex tells them nothing and looks like a defect.
#
# Ordered most-specific first: a real BIDS name should never be re-parsed by
# the loose patterns below it.
SUBJECT_PATTERNS = [
    ("bids", re.compile(r"sub-([A-Za-z0-9]+)")),
    ("subject-prefix", re.compile(_START + r"subj(?:ect)?[_-]?(\d+)", re.I)),
    ("s-prefix", re.compile(_START + r"s(\d{1,3})(?![0-9])", re.I)),
    # Two digits max: an unanchored `p(\d{1,3})` would read the task name
    # "p300" as subject 300.
    ("p-prefix", re.compile(_START + r"p(\d{1,2})(?![0-9])", re.I)),
]
SESSION_PATTERNS = [
    ("bids", re.compile(r"ses-([A-Za-z0-9]+)")),
    ("session-prefix", re.compile(_START + r"ses(?:sion)?[_-]?(\d+)", re.I)),
    ("day-prefix", re.compile(_START + r"day[_-]?(\d+)", re.I)),
    ("visit-prefix", re.compile(_START + r"visit[_-]?(\d+)", re.I)),
]
RUN_PATTERNS = [
    ("bids", re.compile(r"run-(\d+)")),
    ("run-prefix", re.compile(_START + r"run[_-]?(\d+)", re.I)),
    # `block` before the bare `r` fallback, or "block4" loses to a stray "r".
    ("block-prefix", re.compile(_START + r"block[_-]?(\d+)", re.I)),
    ("r-prefix", re.compile(_START + r"r(\d{1,2})(?![0-9])", re.I)),
]
TASK_PATTERNS = [("bids", re.compile(r"task-([A-Za-z0-9]+)"))]

# Substring -> canonical task label. Filenames are the only place paradigm
# survives in most legacy archives.
TASK_KEYWORDS = {
    "p300": "p300", "oddball": "oddball", "speller": "p300speller",
    "rest": "rest", "resting": "rest", "eyesopen": "resteyesopen",
    "eyesclosed": "resteyesclosed", "ssvep": "ssvep", "mi": "motorimagery",
    "motor": "motorimagery", "flanker": "flanker", "stroop": "stroop",
    "nback": "nback", "auditory": "auditory", "visual": "visual",
}

# Filename fragments that mean a human already knew something was wrong.
SUSPECT_TOKENS = ("final", "fixed", "copy", "old", "new", "test", "bad",
                  "redo", "temp", "backup", "dontuse", "corrupt")


@dataclass
class Entities:
    """BIDS entities, each tagged with where the value came from."""

    subject: str | None = None
    session: str | None = None
    task: str | None = None
    run: str | None = None
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def unresolved(self) -> list[str]:
        return [k for k in ("subject", "session", "task", "run")
                if getattr(self, k) is None]


@dataclass
class Recording:
    path: str
    filename: str
    size_bytes: int
    readable: bool
    reason: str | None = None
    # Header facts — trustworthy.
    sfreq_hz: float | None = None
    n_channels: int | None = None
    ch_names: list[str] = field(default_factory=list)
    duration_s: float | None = None
    meas_date: str | None = None
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    n_annotations: int = 0
    annotation_labels: list[str] = field(default_factory=list)
    # Filename inferences — labelled, never merged with header facts.
    entities: Entities = field(default_factory=Entities)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entities"]["unresolved"] = self.entities.unresolved
        return d


def _first_match(patterns: list[tuple[str, re.Pattern]],
                 text: str) -> tuple[str | None, str | None]:
    for label, pattern in patterns:
        m = pattern.search(text)
        if m:
            return m.group(1), label
    return None, None


def infer_entities(filename: str, parent: str = "") -> Entities:
    """Recover subject/session/task/run from a path. Never guesses silently —
    every value carries the pattern that produced it.

    The filename is always searched first. The parent directory is consulted
    only for subject and session, and only as a fallback: BIDS-style trees put
    those in directory names, but any other directory name is just ambient text
    that would otherwise manufacture entities out of an unrelated path."""
    ent = Entities()
    stem = Path(filename).stem

    for attr, patterns in (
        ("subject", SUBJECT_PATTERNS),
        ("session", SESSION_PATTERNS),
        ("run", RUN_PATTERNS),
        ("task", TASK_PATTERNS),
    ):
        value, label = _first_match(patterns, filename)
        source = "filename"
        if value is None and parent and attr in ("subject", "session"):
            value, label = _first_match(patterns, parent)
            source = "parentdir"
        if value is not None:
            setattr(ent, attr, value)
            ent.sources[attr] = f"{source}:{label}"

    if ent.task is None:
        low = re.sub(r"[^a-z0-9]", "", stem.lower())
        for kw, canonical in TASK_KEYWORDS.items():
            if kw in low:
                ent.task = canonical
                ent.sources["task"] = f"filename:keyword({kw})"
                break
    return ent


def _suspect_warnings(filename: str) -> list[str]:
    low = filename.lower()
    hits = [t for t in SUSPECT_TOKENS if t in low]
    if not hits:
        return []
    return [f"filename contains {hits} — provenance uncertain, "
            f"a human previously flagged or duplicated this file"]


def read_header(path: Path) -> Recording:
    """Open one file, take header facts, infer entities. Never loads full data."""
    rec = Recording(
        path=str(path),
        filename=path.name,
        size_bytes=path.stat().st_size,
        readable=False,
        entities=infer_entities(path.name, path.parent.name),
    )
    rec.warnings.extend(_suspect_warnings(path.name))

    ext = path.suffix.lower()
    if ext in UNREADABLE:
        rec.reason = UNREADABLE[ext]
        return rec
    reader = READERS.get(ext)
    if reader is None:
        rec.reason = f"no reader registered for '{ext}'"
        return rec

    try:
        raw = reader(str(path), preload=False, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001 — any failure is a reportable fact
        rec.reason = f"{type(exc).__name__}: {exc}"
        return rec

    info = raw.info
    rec.readable = True
    rec.sfreq_hz = float(info["sfreq"])
    rec.n_channels = len(raw.ch_names)
    rec.ch_names = list(raw.ch_names)
    rec.duration_s = round(float(raw.n_times) / float(info["sfreq"]), 2)
    rec.meas_date = info["meas_date"].isoformat() if info["meas_date"] else None
    rec.highpass_hz = float(info["highpass"]) if info["highpass"] else None
    rec.lowpass_hz = float(info["lowpass"]) if info["lowpass"] else None

    ann = raw.annotations
    rec.n_annotations = len(ann)
    if len(ann):
        rec.annotation_labels = sorted(set(ann.description.tolist()))

    if rec.n_annotations == 0:
        rec.warnings.append(
            "no event annotations — paradigm timing is unrecoverable from this "
            "file alone; check for a companion event/marker file"
        )
    if rec.sfreq_hz < 250:
        rec.warnings.append(
            f"sampling rate {rec.sfreq_hz:g} Hz is marginal for ERP work "
            f"(<=4 ms resolution)"
        )
    return rec


def load_data_uv(path: Path) -> tuple[np.ndarray, list[str], float]:
    """Load full signal in microvolts, for the QC pass."""
    reader = READERS[Path(path).suffix.lower()]
    raw = reader(str(path), preload=True, verbose="ERROR")
    raw.pick("eeg")
    return raw.get_data() * 1e6, list(raw.ch_names), float(raw.info["sfreq"])


def scan(root: Path, recursive: bool = True) -> list[Recording]:
    """Inventory every candidate recording under `root`."""
    root = Path(root)
    known = set(READERS) | set(UNREADABLE)
    pattern = "**/*" if recursive else "*"
    files = sorted(p for p in root.glob(pattern)
                   if p.is_file() and p.suffix.lower() in known)
    return [read_header(p) for p in files]
