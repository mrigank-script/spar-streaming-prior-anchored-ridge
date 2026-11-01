"""Download a daily US equity panel (S&P 500 names) for the quant limits test."""

import io
from pathlib import Path
import time

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "finance"
LIST_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

FALLBACK = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "JNJ", "V", "PG",
    "XOM", "UNH", "HD", "CVX", "MA", "ABBV", "PFE", "KO", "PEP", "COST",
    "MRK", "WMT", "BAC", "CSCO", "TMO", "MCD", "ABT", "CRM", "ADBE", "LIN",
    "NFLX", "AMD", "TXN", "DHR", "PM", "NEE", "WFC", "BMY", "RTX", "QCOM",
    "T", "AMGN", "UPS", "LOW", "SPGI", "INTU", "HON", "UNP", "IBM", "CAT",
    "BA", "GE", "SBUX", "DE", "GILD", "ADP", "MDLZ", "ISRG", "BKNG", "ADI",
    "VRTX", "AXP", "PLD", "MMC", "CB", "CI", "SYK", "ZTS", "MO", "SO",
    "DUK", "BDX", "CL", "NOC", "GD", "ITW", "MMM", "FISV", "APD", "SHW",
    "CSX", "NSC", "FDX", "EMR", "ETN", "AON", "PGR", "TRV", "ALL", "MET",
    "PRU", "AFL", "TGT", "KMB", "GIS", "K", "HSY", "SYY", "STZ", "ROST",
    "TJX", "ORLY", "AZO", "CMG", "YUM", "MAR", "HLT", "DAL", "UAL", "LUV",
    "NKE", "LULU", "TIF", "DIS", "CMCSA", "CHTR", "VZ", "TMUS", "INTC", "MU",
    "AMAT", "LRCX", "KLAC", "ASML", "AVGO", "NOW", "PANW", "SNPS", "CDNS", "ANSS",
]


def tickers() -> list[str]:
    try:
        response = requests.get(LIST_URL, timeout=30)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        symbols = frame["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        symbols = [s for s in symbols if s.isalpha() or "-" in s]
        return sorted(set(symbols))
    except Exception as exc:
        print(f"[panel] S&P list fetch failed ({exc}); using fallback list")
        return sorted(set(FALLBACK))


def main() -> None:
    symbols = tickers()
    print(f"[panel] {len(symbols)} tickers")
    close_parts, vol_parts = [], []
    for start in range(0, len(symbols), 40):
        chunk = symbols[start:start + 40]
        frame = yf.download(chunk, start="2003-06-01", end="2025-12-31",
                            auto_adjust=True, progress=False, threads=True)
        if frame.empty:
            continue
        close = frame["Close"]
        volume = frame["Volume"]
        close_parts.append(close if isinstance(close, pd.DataFrame) else close.to_frame())
        vol_parts.append(volume if isinstance(volume, pd.DataFrame) else volume.to_frame())
        print(f"[panel] chunk {start // 40 + 1}: {chunk[0]}..{chunk[-1]} -> {close.shape}")
        time.sleep(1.5)
    close = pd.concat(close_parts, axis=1).sort_index()
    volume = pd.concat(vol_parts, axis=1).sort_index()
    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]
    close.to_csv(OUT / "equity_close_daily.csv")
    volume.to_csv(OUT / "equity_volume_daily.csv")
    print(f"[panel] saved close={close.shape} volume={volume.shape} -> {OUT}")


if __name__ == "__main__":
    main()
