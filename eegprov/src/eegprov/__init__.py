"""eegprov — provenance and quality tooling for EEG archives.

Turns a folder of undocumented recordings into an inventory, a per-session
quality record, and a draft BIDS-EEG layout — with everything it could not
determine stated explicitly rather than guessed.
"""

from . import bids, inventory, paradigm, qc, report, synth

__version__ = "0.1.0"
__all__ = ["bids", "inventory", "paradigm", "qc", "report", "synth"]
