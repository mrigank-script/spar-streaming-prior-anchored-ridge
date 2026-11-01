"""General TS loaders (DSOF/Autoformer files placed by the user in data/ts/)."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .splits import ETT_BORDERS, chronological_split, ett_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "ts"

FILE_CANDIDATES = {
    "ETTh1": ("ETTh1.csv", "etth1.csv"),
    "ETTh2": ("ETTh2.csv", "etth2.csv"),
    "ETTm1": ("ETTm1.csv", "ettm1.csv"),
    "ETTm2": ("ETTm2.csv", "ettm2.csv"),
    "Exchange": ("exchange_rate.csv", "exchange.txt"),
    "Weather": ("weather.csv", "Weather.csv"),
    "Traffic": ("traffic.csv", "traffic.txt"),
    "Electricity": ("electricity.csv", "electricity.txt"),
}


def resolve_path(name: str, root: Path = DEFAULT_ROOT) -> Path:
    root = Path(root)
    for candidate in FILE_CANDIDATES[name]:
        path = root / candidate
        if path.exists():
            return path
    lower = {path.name.lower(): path for path in root.glob("*") if path.is_file()}
    for candidate in FILE_CANDIDATES[name]:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise FileNotFoundError(
        f"no file for {name} in {root}; expected one of {FILE_CANDIDATES[name]}"
    )


def available(root: Path = DEFAULT_ROOT) -> dict:
    result = {}
    for name in FILE_CANDIDATES:
        try:
            resolve_path(name, root)
            result[name] = True
        except FileNotFoundError:
            result[name] = False
    return result


def _read_frame(path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_csv(path, header=None)

    def is_number(value) -> bool:
        try:
            float(str(value).replace(",", ""))
            return True
        except Exception:
            return False

    first_row = [is_number(v) for v in raw.iloc[0].iloc[1:]]
    header_like = len(first_row) > 1 and (sum(first_row) / len(first_row) < 0.5)
    if header_like:
        frame = pd.read_csv(path)
        first = frame.columns[0]
        frame[first] = pd.to_datetime(frame[first], errors="coerce")
    else:
        frame = raw
        if pd.api.types.is_numeric_dtype(frame.iloc[:, 0]):
            frame.columns = [f"var{i}" for i in range(frame.shape[1])]
        else:
            frame.columns = ["date"] + [f"var{i}" for i in range(frame.shape[1] - 1)]
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def load_raw(name: str, root: Path = DEFAULT_ROOT) -> dict:
    """Load a TS file: numeric values (T, C), target index, column names, dates."""
    path = resolve_path(name, root)
    frame = _read_frame(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    numeric_cols = [
        c for c in frame.columns
        if np.issubdtype(frame[c].dtype, np.number) and not c.lower().startswith("unnamed")
    ]
    first = frame.columns[0]
    if first in numeric_cols:
        dates = np.full(len(frame), np.datetime64("NaT"))
    else:
        dates = pd.to_datetime(frame[first], errors="coerce").to_numpy()
    values = frame[numeric_cols].to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    target = numeric_cols.index("OT") if "OT" in numeric_cols else len(numeric_cols) - 1
    return {"values": values, "target": target, "columns": numeric_cols, "dates": dates}


def split_indices(name: str, n: int):
    if name in ETT_BORDERS:
        return ett_split(n, name)
    return chronological_split(n, (0.2, 0.05, 0.75))


def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray):
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0) + 1e-8
    return ((train - mu) / sd, (val - mu) / sd, (test - mu) / sd, mu, sd)
