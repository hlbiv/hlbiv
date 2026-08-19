# eegprov

Provenance and quality tooling for EEG archives.

Point it at a folder of undocumented recordings and it returns an inventory, a
per-session quality record, and a draft BIDS-EEG layout — with everything it
could *not* determine stated explicitly instead of guessed.

## Why

The bottleneck in EEG is not volume, it is provenance. Recordings survive;
the context does not. Six months after a study, a folder looks like this:

```
s04_run2_p300_final.edf
s04_run2_p300_final_FIXED.edf
recording_copy_2.edf
s06_speller_block1.dat
```

The file headers know the sampling rate and the channel names. They do not
know the subject, the session, the paradigm, the reference electrode, or which
of those two `s04` files anyone should actually use. That missing context is
what makes an archive unusable, and reconstructing it by hand is the reason
most archives are never reused at all.

This tool does the recoverable part automatically and turns the rest into a
short, specific list of questions for whoever ran the study.

## Install

```bash
pip install -e ".[dev]"      # add ".[edf]" for EDF export in the demo
```

Requires Python 3.10+. Depends on MNE, NumPy, and SciPy.

## Use

```bash
eegprov demo    ./archive                  # synthetic archive to work against
eegprov scan    ./archive                  # inventory + inferred entities
eegprov card    ./archive/s02_run2.edf     # full provenance card, one file
eegprov archive ./archive -o ./report      # everything, written to disk
```

`archive` writes `inventory.md`, `inventory.json`, and one markdown provenance
card per recording.

## What it reports

**Inventory** — every candidate file, readable or not. Formats it cannot open
(BCI2000 `.dat`, MATLAB containers, OpenBCI text exports) are still listed,
with the reason. A file missing from the inventory is a file nobody will
remember to convert.

**Identity** — subject, session, task, and run recovered from the filename,
each labelled with how it was derived (`filename:s-prefix`, `parentdir:bids`,
`filename:keyword(p300)`). Anything unrecoverable is reported as *unresolved*.
The tool never fills in an entity to make a file convert.

**Signal quality** — per channel: amplitude, peak-to-peak, robust z against
its peers, low-frequency drift ratio, and flags for `flat` / `noisy` /
`drift`. Per session: detected mains frequency and contamination ratio, blink
rate, and the fraction of data surviving amplitude rejection.

**BIDS draft** — a proposed path and sidecar, plus two lists that matter more
than the draft itself: *blockers* (what prevents conversion) and *questions
only a human can answer*. `EEGReference` is required by BIDS and is absent
from essentially every raw format — inventing a plausible value there would
silently corrupt every downstream re-reference.

## What it will not do

- It flags; it does not delete, interpolate, or silently drop data.
- It does not write a BIDS tree. It produces a plan you review first. A BIDS
  dataset built on invented subject IDs is worse than no BIDS dataset — it
  looks authoritative and is wrong.
- It does not replace a human reading the flagged sessions. It replaces the
  human reading *all* of them.

## Design notes

Two decisions are worth knowing about, because both were wrong on the first
attempt and the tests caught them:

**Drift is measured below 0.5 Hz, not below 1 Hz.** Real EEG is 1/f, so a
0.1–1 Hz band is dominated by legitimate delta activity and flags healthy
channels. Clean pink-noise background sits near 0.18 in the 0.05–0.5 Hz band;
injected electrode drift sits near 0.49.

**A statistical outlier is not automatically a large one.** On a montage where
most channels are near-identical, the median absolute deviation collapses and
ordinary scalp topography — frontal blink pickup, posterior alpha — scores
z > 4 while sitting within 25% of the median. A channel must be *both* a
robust-z outlier *and* genuinely high-amplitude before it is called noisy.

## Tests

```bash
pytest
```

Every quality check is tested by injecting the defect it is meant to catch —
a dead electrode, a noisy temporal channel, a drifting reference, 50 Hz versus
60 Hz mains, absent blinks. `synth.py` generates all of it, so the suite needs
no downloads, no credentials, and no real recordings.

## Status

Working, tested, and validated only against synthetic data. The next step is a
pass over a public corpus — Temple TUH or OpenNeuro — which is what the
thresholds actually need to be calibrated against.
