"""Command line entry point.

    eegprov demo      <dir>   generate a messy synthetic archive to work against
    eegprov scan      <dir>   inventory recordings, headers and inferred entities
    eegprov card      <file>  full provenance card for one recording
    eegprov archive   <dir>   scan + QC + BIDS draft for everything, written out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bids, inventory, paradigm, qc, report, synth


def _demo(args: argparse.Namespace) -> int:
    """Write a synthetic archive that reproduces real archive problems:
    inconsistent naming, mixed sampling rates, dead and drifting electrodes,
    a session with no event markers, and a file no reader can open."""
    out = Path(args.directory)
    out.mkdir(parents=True, exist_ok=True)
    ext = ".edf" if args.edf else ".raw.fif"

    sessions = [
        # Clean, well-named — should convert with only human-only fields open.
        ("sub-01_ses-01_task-oddball_run-01" + ext,
         synth.SynthSpec(seed=1)),
        # Legacy naming, dead electrode at Pz — the P300 site.
        ("s02_run2_p300_final" + ext,
         synth.SynthSpec(seed=2, flat_channels=["Pz"])),
        # Drifting reference and a noisy temporal channel.
        ("s03_day2_oddball_FIXED" + ext,
         synth.SynthSpec(seed=3, drift_channels=["M1"], noisy_channels=["T8"])),
        # European site: 50 Hz mains, low sampling rate, no events at all.
        ("P04_resting_eyesclosed" + ext,
         synth.SynthSpec(seed=4, sfreq=128.0, line_freq=50.0,
                         line_amp_uv=9.0, oddball=False)),
        # Nothing recoverable from the name.
        ("recording_copy_2" + ext,
         synth.SynthSpec(seed=5, blink_rate_per_min=0.0)),
    ]

    written = []
    for name, spec in sessions:
        path = synth.write_session(out / name, spec)
        written.append(path.name)

    # A format detectable but unreadable — every real archive has these.
    (out / "s06_speller_block1.dat").write_bytes(b"BCI2000-ish placeholder\n")
    written.append("s06_speller_block1.dat")

    print(f"Wrote {len(written)} files to {out}/")
    for name in written:
        print(f"  {name}")
    print(f"\nNext:  eegprov archive {out} -o {out}_report")
    return 0


def _scan(args: argparse.Namespace) -> int:
    records = inventory.scan(Path(args.directory))
    if args.json:
        print(json.dumps([r.to_dict() for r in records], indent=2))
        return 0

    if not records:
        print(f"No EEG files found under {args.directory}")
        return 1

    print(f"{'file':<40} {'sfreq':>7} {'ch':>4} {'dur':>8}  entities")
    print("-" * 88)
    for r in records:
        if not r.readable:
            print(f"{r.filename:<40} {'—':>7} {'—':>4} {'—':>8}  UNREADABLE")
            continue
        ent = r.entities
        bits = [f"{k[:3]}={getattr(ent, k)}" for k in
                ("subject", "session", "task", "run") if getattr(ent, k)]
        missing = ent.unresolved
        label = " ".join(bits) or "—"
        if missing:
            label += f"  [unresolved: {','.join(missing)}]"
        print(f"{r.filename:<40} {r.sfreq_hz:>7.0f} {r.n_channels:>4} "
              f"{r.duration_s:>7.0f}s  {label}")

    readable = [r for r in records if r.readable]
    hours = sum(r.duration_s or 0 for r in readable) / 3600
    print("-" * 88)
    print(f"{len(records)} file(s), {len(readable)} readable, {hours:.2f} h of signal")
    return 0


def _analyze_one(rec: inventory.Recording, no_qc: bool = False):
    """Signal-level passes for one readable recording. Reads the file once."""
    data, ch_names, sfreq, desc, onsets = inventory.load_recording(Path(rec.path))
    q = None if no_qc else qc.analyze(data, ch_names, sfreq)
    pd = paradigm.analyze(desc, onsets, data, ch_names, sfreq)
    return q, pd


def _card(args: argparse.Namespace) -> int:
    path = Path(args.file)
    rec = inventory.read_header(path)
    if not rec.readable:
        print(report.card(rec))
        return 1
    q, pd = _analyze_one(rec)
    print(report.card(rec, q, bids.draft(rec, q), pd))
    return 0


def _archive(args: argparse.Namespace) -> int:
    root = Path(args.directory)
    out = Path(args.out) if args.out else None
    records = inventory.scan(root)
    if not records:
        print(f"No EEG files found under {root}")
        return 1

    drafts, cards = [], []
    for rec in records:
        q = pd = None
        if rec.readable:
            try:
                q, pd = _analyze_one(rec, no_qc=args.no_qc)
            except Exception as exc:  # noqa: BLE001 — a failed pass is reportable
                rec.warnings.append(f"analysis failed: {type(exc).__name__}: {exc}")
        bd = bids.draft(rec, q)
        drafts.append(bd)
        cards.append((rec, report.card(rec, q, bd, pd)))

    summary = bids.summarize(drafts)
    overview = report.archive_summary(records, summary)
    print(overview)

    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "inventory.md").write_text(overview)
        (out / "inventory.json").write_text(json.dumps(
            {
                "summary": summary,
                "recordings": [r.to_dict() for r in records],
                "bids_drafts": [d.to_dict() for d in drafts],
            },
            indent=2, default=str,
        ))
        cards_dir = out / "cards"
        cards_dir.mkdir(exist_ok=True)
        for rec, text in cards:
            (cards_dir / f"{Path(rec.filename).stem}.md").write_text(text)
        print(f"\nWrote {len(cards)} provenance cards + inventory to {out}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eegprov",
        description="Inventory, quality-check, and draft BIDS metadata for EEG archives.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="generate a synthetic messy archive")
    d.add_argument("directory")
    d.add_argument("--edf", action="store_true",
                   help="write EDF instead of FIF (needs the edfio package)")
    d.set_defaults(func=_demo)

    s = sub.add_parser("scan", help="inventory recordings and infer entities")
    s.add_argument("directory")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_scan)

    c = sub.add_parser("card", help="full provenance card for one recording")
    c.add_argument("file")
    c.set_defaults(func=_card)

    a = sub.add_parser("archive", help="scan + QC + BIDS draft for a whole archive")
    a.add_argument("directory")
    a.add_argument("-o", "--out", help="directory to write cards and inventory into")
    a.add_argument("--no-qc", action="store_true",
                   help="skip signal QC (headers only, much faster)")
    a.set_defaults(func=_archive)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
