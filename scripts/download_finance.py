"""Download small finance datasets into data/finance/ (no GB-scale sets).

Sources: Ken French Data Library (daily factors) and a daily ETF panel via yfinance.
GB-scale sets (OpenAP, CRSP) are intentionally skipped until small-set results justify them.
"""

from pathlib import Path
import io
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "finance"

FRENCH = {
    "FF5_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "MOM_daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}

TICKERS = [
    "SPY", "QQQ", "IWM", "EEM", "EFA",
    "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
    "TLT", "GLD", "HYG", "LQD",
]


def download_french() -> None:
    for name, url in FRENCH.items():
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(response.content)).extractall(folder)
        print(f"[finance] {name}: {len(response.content) / 1e6:.2f} MB -> {folder}")


def download_etfs() -> None:
    import yfinance as yf

    frame = yf.download(
        TICKERS,
        start="2004-01-01",
        end="2025-12-31",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = frame["Close"].dropna(how="all")
    path = OUT / "etf_close_daily.csv"
    close.to_csv(path)
    print(f"[finance] ETF panel: {close.shape[0]} days x {close.shape[1]} tickers -> {path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    download_french()
    try:
        download_etfs()
    except Exception as exc:
        print(f"[finance] ETF panel failed ({exc}); French factors are saved, rerun later.")
