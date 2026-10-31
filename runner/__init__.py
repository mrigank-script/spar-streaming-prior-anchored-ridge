"""Streaming AAL runner package."""

from . import classifier
from .aal import RunnerConfig, make_stream_fn, run_method, run_stream, warm_start
from .align import (
    align_windows,
    cov_subspace,
    feature_subspace,
    moment_drift_blocks,
    pad_blocks,
    pad_blocks_cov,
    trigger_metrics,
)
from .live import TriggerConfig, run_live

__all__ = [
    "RunnerConfig",
    "TriggerConfig",
    "classifier",
    "make_stream_fn",
    "run_method",
    "run_live",
    "run_stream",
    "warm_start",
    "align_windows",
    "cov_subspace",
    "feature_subspace",
    "moment_drift_blocks",
    "pad_blocks",
    "pad_blocks_cov",
    "trigger_metrics",
]
