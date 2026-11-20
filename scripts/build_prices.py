"""Snapshot daily prices and split history from Yahoo Finance into data/ (fallback for the live app).

Usage: python scripts/build_prices.py
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from universe import BENCHMARKS, INTERNATIONAL, UNIVERSE  # noqa: E402

TICKERS = [c["ticker"] for c in UNIVERSE] + list(INTERNATIONAL.values()) + list(BENCHMARKS.values())


def main() -> None:
    raw = yf.download(TICKERS, start="2019-12-31", auto_adjust=False, progress=False, threads=False)
    raw["Adj Close"].round(4).to_csv(ROOT / "data" / "prices_total_return.csv")
    raw["Close"].round(4).to_csv(ROOT / "data" / "prices_close.csv")
    rows = []
    for t in TICKERS:
        for d, ratio in yf.Ticker(t).splits.items():
            rows.append({"ticker": t, "date": pd.Timestamp(d).tz_localize(None).date(), "ratio": ratio})
    pd.DataFrame(rows, columns=["ticker", "date", "ratio"]).to_csv(ROOT / "data" / "splits.csv", index=False)
    print(f"prices through {raw.index[-1].date()}, {len(rows)} splits", file=sys.stderr)


if __name__ == "__main__":
    main()
