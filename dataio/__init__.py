"""Dataset loading, splitting, and streaming-protocol utilities."""

from . import finance, splits, synthetic, ts
from .eeg import load_subject as load_eeg_subject

__all__ = ["finance", "splits", "synthetic", "ts", "load_eeg_subject"]
