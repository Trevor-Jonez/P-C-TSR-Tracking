"""Read-only access to data/pc_market.db (built by scripts/build_database.py)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB = Path(__file__).parent / "data" / "pc_market.db"


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        return pd.read_sql_query(sql, con, params=params)


def metric(name: str, dimension: str = "", include_forecast: bool = True, period_type: str | None = None) -> pd.DataFrame:
    """Latest-vintage values for one industry metric."""
    sql = "SELECT * FROM industry_latest WHERE metric = ? AND dimension = ?"
    args: list = [name, dimension]
    if not include_forecast:
        sql += " AND is_forecast = 0"
    if period_type:
        sql += " AND period_type = ?"
        args.append(period_type)
    return q(sql + " ORDER BY period", tuple(args)).merge(
        q("SELECT source_id, title AS source_title, url AS source_url, vintage_date AS source_vintage FROM sources"),
        on="source_id", how="left")


def fundamentals_raw() -> pd.DataFrame:
    return q("SELECT * FROM carrier_fundamentals_raw")


def splits() -> pd.DataFrame:
    return q("SELECT * FROM splits")


def prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = q("SELECT * FROM prices")
    p["date"] = pd.to_datetime(p["date"])
    tr = p.pivot(index="date", columns="ticker", values="total_return_close").sort_index()
    cl = p.pivot(index="date", columns="ticker", values="close").sort_index()
    return tr, cl


def fred(series_id: str) -> pd.Series:
    d = q("SELECT date, value FROM fred_observations WHERE series_id = ?", (series_id,))
    return pd.Series(d["value"].values, index=pd.to_datetime(d["date"]), name=series_id)


def table(name: str) -> pd.DataFrame:
    return q(f"SELECT * FROM {name}")  # noqa: S608 (internal fixed names only)
