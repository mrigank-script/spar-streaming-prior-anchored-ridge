"""Synthetic drifting-teacher loader (files produced by scripts/generate_synthetic.py)."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "synthetic"


def load_config(name: str, root: Path = DEFAULT_ROOT) -> dict:
    """Load one npz config: x, y, c, U, w0, onsets, meta."""
    with np.load(Path(root) / name) as data:
        meta = json.loads(str(data["meta"]))
        out = {key: data[key] for key in ("x", "y", "c", "U", "w0", "onsets")}
    out["meta"] = meta
    return out


def split_indices(d: dict, train_frac: float = 0.2):
    """Chronological warm-up/stream split indices."""
    n = len(d["y"])
    cut = int(n * train_frac)
    return slice(0, cut), slice(cut, n)


def config_name(dim: int, length: int, drift: str, seed: int) -> str:
    return f"d{dim}_T{length}_{drift}_s{seed}.npz"


def list_configs(root: Path = DEFAULT_ROOT) -> list[str]:
    return sorted(path.name for path in Path(root).glob("d*_T*_*.npz"))
