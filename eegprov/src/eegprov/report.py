"""Render provenance cards.

The card is the deliverable: one page per recording saying what this file is,
how good the signal is, and what nobody can tell from the file alone. It is
the durable artifact that survives after the person who ran the session has
graduated.
"""

from __future__ import annotations

from .bids import BidsDraft
from .inventory import Recording
from .qc import QCReport


def _fmt(value, suffix: str = "", dash: str = "unknown") -> str:
    if value is None:
        return dash
    return f"{value}{suffix}"


def card(rec: Recording, qc: QCReport | None = None,
         bd: BidsDraft | None = None) -> str:
    """One recording, as markdown."""
    lines = [f"# {rec.filename}", "", f"`{rec.path}`", ""]

    lines += ["## Identity", ""]
    ent = rec.entities
    for name in ("subject", "session", "task", "run"):
        value = getattr(ent, name)
        if value is None:
            lines.append(f"- **{name}**: *unresolved*")
        else:
            lines.append(f"- **{name}**: `{value}`  ({ent.sources.get(name, 'header')})")
    lines.append("")

    lines += ["## Header facts", ""]
    lines += [
        f"- sampling rate: {_fmt(rec.sfreq_hz, ' Hz')}",
        f"- channels: {_fmt(rec.n_channels)}",
        f"- duration: {_fmt(rec.duration_s, ' s')}",
        f"- recorded: {_fmt(rec.meas_date, dash='not stored')}",
        f"- acquisition filters: "
        f"{_fmt(rec.highpass_hz, ' Hz', 'none')} – {_fmt(rec.lowpass_hz, ' Hz', 'none')}",
        f"- annotations: {rec.n_annotations}"
        + (f" ({', '.join(rec.annotation_labels)})" if rec.annotation_labels else ""),
        "",
    ]

    if qc is not None:
        lines += _qc_section(qc)
    if bd is not None:
        lines += _bids_section(bd)
    if rec.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in rec.warnings] + [""]
    return "\n".join(lines)


def _qc_section(qc: QCReport) -> list[str]:
    lines = ["## Signal quality", ""]
    lines += [
        f"- data retained after amplitude rejection: **{qc.retained_fraction:.1%}**",
        f"- line noise: {_fmt(qc.line_freq_detected, ' Hz', 'not detected')}"
        + (f" at {qc.line_ratio}x baseline" if qc.line_ratio else ""),
        f"- blink rate: {_fmt(qc.blink_rate_per_min, '/min', 'not estimable')}",
        "",
    ]

    bad = [c for c in qc.channels if c.flags]
    if bad:
        lines += ["### Flagged channels", "",
                  "| channel | std (uV) | p-p (uV) | robust z | drift | flags |",
                  "|---|---|---|---|---|---|"]
        for c in bad:
            lines.append(
                f"| {c.name} | {c.std_uv} | {c.ptp_uv} | {c.robust_z} | "
                f"{c.drift_ratio} | {', '.join(c.flags)} |"
            )
        lines.append("")
    else:
        lines += ["All channels within tolerance.", ""]

    if qc.flags:
        lines += ["### Session flags", ""] + [f"- {f}" for f in qc.flags] + [""]
    return lines


def _bids_section(bd: BidsDraft) -> list[str]:
    lines = ["## BIDS draft", ""]
    if bd.proposed_path:
        lines += [f"Proposed path: `{bd.proposed_path}`", ""]
    else:
        lines += ["*No path proposed — entities unresolved.*", ""]

    if bd.blockers:
        lines += ["### Blockers", ""] + [f"- {b}" for b in bd.blockers] + [""]
    if bd.questions_for_human:
        lines += ["### Questions only a human can answer", ""]
        lines += [f"- {q}" for q in bd.questions_for_human] + [""]
    return lines


def archive_summary(records: list[Recording], summary: dict) -> str:
    """Archive-level view — the thing you show someone to explain the state
    of a folder nobody has opened in five years."""
    readable = [r for r in records if r.readable]
    unreadable = [r for r in records if not r.readable]

    lines = ["# Archive inventory", ""]
    lines += [
        f"- files found: **{len(records)}**",
        f"- readable: **{len(readable)}**",
        f"- unreadable: **{len(unreadable)}**",
        f"- BIDS-ready: **{summary['ready']}** / blocked: **{summary['blocked']}**",
        "",
    ]

    if readable:
        total_hours = sum(r.duration_s or 0 for r in readable) / 3600.0
        rates = sorted({r.sfreq_hz for r in readable if r.sfreq_hz})
        counts = sorted({r.n_channels for r in readable if r.n_channels})
        lines += [
            f"- total signal: **{total_hours:.2f} hours**",
            f"- sampling rates present: {', '.join(f'{r:g} Hz' for r in rates)}",
            f"- channel counts present: {', '.join(str(c) for c in counts)}",
            "",
        ]
        if len(rates) > 1 or len(counts) > 1:
            lines += ["> Heterogeneous acquisition across the archive — recordings "
                      "cannot be pooled without resampling/channel reconciliation.", ""]

    if summary["blocker_counts"]:
        lines += ["## What blocks conversion", ""]
        lines += [f"- {k} — {v} file(s)" for k, v in summary["blocker_counts"].items()]
        lines.append("")

    if summary["open_questions"]:
        lines += ["## Questions to put to whoever ran these studies", ""]
        lines += [f"- **{k}** — affects {v} file(s)"
                  for k, v in summary["open_questions"].items()]
        lines.append("")

    if unreadable:
        lines += ["## Unreadable files", ""]
        for r in unreadable:
            lines.append(f"- `{r.filename}` — {r.reason}")
        lines.append("")
    return "\n".join(lines)
