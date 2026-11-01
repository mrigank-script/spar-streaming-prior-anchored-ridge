"""Finance loaders: Ken French daily factors and the ETF panel; walk-forward panel builder."""

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "finance"


def parse_french(path: Path) -> pd.DataFrame:
    """Parse a Ken French daily CSV (skips copyright block, stops at first blank/annual section)."""
    lines = Path(path).read_text().splitlines()
    header = None
    for i, line in enumerate(lines):
        if not line.startswith(","):
            continue
        nxt = next((s.strip() for s in lines[i + 1:] if s.strip()), "")
        if nxt[:1].isdigit():
            header = i
            break
    if header is None:
        raise ValueError(f"Ken French header not found in {path}")
    body = []
    for line in lines[header + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        body.append(stripped)
    frame = pd.read_csv(io.StringIO("\n".join([lines[header]] + body)), index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    return frame.astype(float)


def _first_csv(folder: Path) -> Path:
    files = sorted(list(folder.glob("*.CSV")) + list(folder.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV in {folder}")
    return files[0]


def load_ff5(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    return parse_french(_first_csv(Path(root) / "FF5_daily"))


def load_momentum(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    return parse_french(_first_csv(Path(root) / "MOM_daily"))


def load_etf_panel(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    frame = pd.read_csv(Path(root) / "etf_close_daily.csv", index_col=0, parse_dates=True)
    return frame.sort_index()


def build_panel(close: pd.DataFrame, lags: int = 5, horizon: int = 1):
    """Feature panel (T, N, lags) of lagged log returns and next-period target (T, N)."""
    log_ret = np.log(close).diff()
    features = [log_ret.shift(lag).to_numpy(dtype=np.float64) for lag in range(1, lags + 1)]
    X = np.stack(features, axis=-1)
    y = log_ret.shift(-horizon).to_numpy(dtype=np.float64)
    return X, y


def panel_split(dates: pd.DatetimeIndex, train_end: str = "2014-12-31"):
    """Expanding train / streaming test masks by date."""
    train = np.asarray(dates <= pd.Timestamp(train_end))
    return train, ~train
