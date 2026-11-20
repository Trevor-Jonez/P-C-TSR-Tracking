"""Analytics for the P&C value tracker: fundamentals, TSR decomposition, regimes, shocks, outliers.

All statistics are computed from SEC 10-K XBRL data and Yahoo Finance prices in data/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from universe import UNIVERSE

NAME = {c["ticker"]: c["name"] for c in UNIVERSE}
SEGMENT = {c["ticker"]: c["segment"] for c in UNIVERSE}

# Dated market events used only to *label* statistically detected breaks/shocks (never to find them).
EVENTS = [
    ("2020-11-09", "Vaccine efficacy news triggers value rotation"),
    ("2021-08-29", "Hurricane Ida landfall"),
    ("2022-03-16", "Fed begins rate-hiking cycle"),
    ("2022-06-15", "Fed 75 bp hike after CPI surprise"),
    ("2022-09-28", "Hurricane Ian landfall"),
    ("2022-11-10", "Cooler CPI print sparks broad risk-on rally"),
    ("2023-03-10", "Silicon Valley Bank failure"),
    ("2024-09-18", "Fed begins cutting rates"),
    ("2024-10-09", "Hurricane Milton landfall"),
    ("2025-01-07", "Los Angeles wildfires begin"),
    ("2025-01-27", "AI-driven tech sell-off rotates money to defensives"),
    ("2025-04-02", "US tariff announcement sell-off"),
]
EVENTS = [(pd.Timestamp(d), label) for d, label in EVENTS]


def nearest_event(date: pd.Timestamp, window_days: int) -> str | None:
    best = min(EVENTS, key=lambda e: abs((e[0] - date).days))
    return best[1] if abs((best[0] - date).days) <= window_days else None


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------
def robust_z(x: pd.Series) -> pd.Series:
    """Median/MAD z-score (1.4826 scales MAD to sigma under normality)."""
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / mad if mad else x * 0


def outliers(x: pd.Series, z: float = 2.5) -> list[str]:
    rz = robust_z(x.dropna())
    return list(rz[rz.abs() > z].index)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
def split_factor_after(splits: pd.DataFrame, ticker: str, date) -> float:
    s = splits[(splits["ticker"] == ticker) & (pd.to_datetime(splits["date"]) > pd.Timestamp(date))]
    return float(s["ratio"].prod()) if len(s) else 1.0


def year_end_price(close: pd.DataFrame, ticker: str, year: int) -> float:
    s = close[ticker].dropna()
    s = s[s.index <= pd.Timestamp(f"{year}-12-31")]
    return float(s.iloc[-1]) if len(s) else np.nan


def build_fundamentals(raw: pd.DataFrame, close: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Company-year metrics on today's split basis (consistent with Yahoo split-adjusted prices)."""
    f = raw.copy()
    for c in ["preferred", "goodwill", "intangibles"]:
        f[c] = f[c].fillna(0)
    f["tbv"] = f["equity"] - f["preferred"] - f["goodwill"] - f["intangibles"]
    f["split_factor"] = [split_factor_after(splits, t, d) for t, d in zip(f["ticker"], f["shares_as_of"])]
    f["shares_adj"] = f["shares_outstanding"] * f["split_factor"]
    f["tbvps"] = (f["tbv"] / f["shares_adj"]).where(f["tbv"] > 0)  # negative TBV: ratios not meaningful
    f["price"] = [year_end_price(close, t, y) if t in close else np.nan for t, y in zip(f["ticker"], f["fiscal_year"])]
    f["p_tbv"] = f["price"] / f["tbvps"]
    f = f.sort_values(["ticker", "fiscal_year"])
    g = f.groupby("ticker")
    f["avg_tbv"] = (f["tbv"] + g["tbv"].shift()) / 2
    f["rote_pct"] = (f["net_income"] / f["avg_tbv"] * 100).where((f["tbv"] > 0) & (g["tbv"].shift() > 0))
    lr = f["losses_incurred"] / f["premiums_earned"] * 100
    f["loss_ratio_pct"] = lr.where(lr.between(35, 120))  # drop partial/segment-only tags
    f["nii_to_premium_pct"] = f["net_investment_income"] / f["premiums_earned"] * 100
    f["aoci_to_tbv_pct"] = (f["aoci"] / f["tbv"] * 100).where(f["tbv"] > 0)
    f["tbvps_growth_pct"] = g["tbvps"].pct_change() * 100
    f["premium_growth_pct"] = g["premiums_earned"].pct_change() * 100
    f["loss_ratio_change_pp"] = f["loss_ratio_pct"] - g["loss_ratio_pct"].shift()
    f["name"] = f["ticker"].map(NAME)
    f["segment"] = f["ticker"].map(SEGMENT)
    return f


# ---------------------------------------------------------------------------
# TSR and decomposition
# ---------------------------------------------------------------------------
def ann(x0: float, x1: float, years: float) -> float:
    return ((x1 / x0) ** (1 / years) - 1) * 100 if x0 and x0 > 0 and x1 > 0 and years > 0 else np.nan


def price_on(s: pd.Series, when: pd.Timestamp) -> tuple:
    s = s.dropna()
    s = s[s.index <= when]
    return (s.index[-1], float(s.iloc[-1])) if len(s) else (None, np.nan)


def decompose_tsr(fund: pd.DataFrame, tr: pd.DataFrame, close: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """Annualized TSR split into TBV/share growth, P/TBV re-rating and cash yield (dividends).

    Log-additive split of price return = ln(TBVPS ratio) + ln(P/TBV ratio), scaled to the
    arithmetic annualized price return; cash yield = TSR - price return.
    """
    rows = []
    a, b = pd.Timestamp(f"{y0}-12-31"), pd.Timestamp(f"{y1}-12-31")
    yrs = (b - a).days / 365.25
    for t in fund["ticker"].unique():
        f = fund[fund["ticker"] == t].set_index("fiscal_year")
        if y0 not in f.index or y1 not in f.index or t not in tr:
            continue
        tsr = ann(price_on(tr[t], a)[1], price_on(tr[t], b)[1], yrs)
        pr = ann(price_on(close[t], a)[1], price_on(close[t], b)[1], yrs)
        l_tbv = np.log(f.loc[y1, "tbvps"] / f.loc[y0, "tbvps"]) if f.loc[y0, "tbvps"] > 0 and f.loc[y1, "tbvps"] > 0 else np.nan
        l_mult = np.log(f.loc[y1, "p_tbv"] / f.loc[y0, "p_tbv"]) if f.loc[y0, "p_tbv"] > 0 and f.loc[y1, "p_tbv"] > 0 else np.nan
        tot = l_tbv + l_mult
        share = (l_tbv / tot) if tot and np.isfinite(tot) and abs(tot) > 1e-9 else np.nan
        rows.append({
            "ticker": t, "name": NAME[t], "segment": SEGMENT[t],
            "tsr_pct": tsr, "price_return_pct": pr,
            "tbvps_growth_pp": pr * share if np.isfinite(share) else np.nan,
            "multiple_change_pp": pr * (1 - share) if np.isfinite(share) else np.nan,
            "cash_yield_pp": tsr - pr,
            "avg_rote_pct": fund[(fund["ticker"] == t) & fund["fiscal_year"].between(y0 + 1, y1)]["rote_pct"].mean(),
            "avg_loss_ratio_pct": fund[(fund["ticker"] == t) & fund["fiscal_year"].between(y0 + 1, y1)]["loss_ratio_pct"].mean(),
            "p_tbv_start": f.loc[y0, "p_tbv"], "p_tbv_end": f.loc[y1, "p_tbv"],
        })
    return pd.DataFrame(rows)


def trailing_tsr(tr: pd.DataFrame, years: float = 5) -> pd.Series:
    out = {}
    for t in tr.columns:
        s = tr[t].dropna()
        if s.empty:
            continue
        end = s.index[-1]
        d0, p0 = price_on(s, end - pd.DateOffset(years=int(years)))
        out[t] = ann(p0, s.iloc[-1], (end - d0).days / 365.25) if d0 is not None else np.nan
    return pd.Series(out)


# ---------------------------------------------------------------------------
# Baskets
# ---------------------------------------------------------------------------
def basket_returns(tr: pd.DataFrame, tickers: list[str]) -> pd.Series:
    r = tr[tickers].pct_change(fill_method=None)
    return r.mean(axis=1, skipna=True).dropna()


def relative_daily(tr: pd.DataFrame, tickers: list[str], bench: str = "^GSPC") -> pd.Series:
    b = basket_returns(tr, tickers)
    m = tr[bench].pct_change(fill_method=None)
    return (b - m).dropna()


# ---------------------------------------------------------------------------
# Regime (change-point) detection: binary segmentation, CUSUM statistic, permutation p-values
# ---------------------------------------------------------------------------
def _cusum_split(x: np.ndarray, min_size: int) -> tuple[int, float]:
    """Best mean-shift split point and its standardized CUSUM statistic."""
    n = len(x)
    c = np.cumsum(x - x.mean())
    k = np.arange(1, n)
    stat = np.abs(c[:-1]) / np.sqrt(k * (n - k) / n)
    stat[: min_size - 1] = 0
    stat[n - min_size:] = 0
    i = int(np.argmax(stat))
    return i + 1, float(stat[i])


def detect_regimes(weekly: pd.Series, min_weeks: int = 13, alpha: float = 0.05, n_perm: int = 999,
                   seed: int = 7) -> list[dict]:
    """Recursive binary segmentation; a split is kept only if its CUSUM beats the permutation null.

    Returns a list of breaks: {index, date, stat, p_value}.
    """
    rng = np.random.default_rng(seed)
    x_all = weekly.values.astype(float)
    found = []

    def rec(lo: int, hi: int) -> None:
        x = x_all[lo:hi]
        if len(x) < 2 * min_weeks:
            return
        k, stat = _cusum_split(x, min_weeks)
        null = np.empty(n_perm)
        for b in range(n_perm):
            null[b] = _cusum_split(rng.permutation(x), min_weeks)[1]
        p = (1 + (null >= stat).sum()) / (n_perm + 1)
        if p < alpha:
            found.append({"index": lo + k, "date": weekly.index[lo + k], "stat": stat, "p_value": p})
            rec(lo, lo + k)
            rec(lo + k, hi)

    rec(0, len(x_all))
    return sorted(found, key=lambda d: d["index"])


def best_break(weekly: pd.Series, min_weeks: int = 13, n_perm: int = 999, seed: int = 7) -> dict:
    """Strongest single mean-shift candidate in the full sample and its permutation p-value."""
    rng = np.random.default_rng(seed)
    x = weekly.values.astype(float)
    k, stat = _cusum_split(x, min_weeks)
    null = np.array([_cusum_split(rng.permutation(x), min_weeks)[1] for _ in range(n_perm)])
    return {"date": weekly.index[k], "stat": stat, "p_value": (1 + (null >= stat).sum()) / (n_perm + 1)}


def regime_table(weekly_rel: pd.Series, breaks: list[dict], event_window: int = 45) -> pd.DataFrame:
    """Per-regime annualized excess return, plus a Welch test vs the previous regime."""
    edges = [0] + [b["index"] for b in breaks] + [len(weekly_rel)]
    rows = []
    prev = None
    for k in range(len(edges) - 1):
        seg = weekly_rel.iloc[edges[k]:edges[k + 1]]
        mu = seg.mean() * 52 * 100
        t_zero = stats.ttest_1samp(seg, 0.0)
        row = {
            "regime": k + 1, "start": seg.index[0], "end": seg.index[-1], "weeks": len(seg),
            "excess_return_ann_pct": mu, "cumulative_excess_pct": ((1 + seg).prod() - 1) * 100,
            "p_vs_zero": t_zero.pvalue, "break_p_value": np.nan, "catalyst": None,
        }
        if prev is not None:
            row["break_p_value"] = breaks[k - 1]["p_value"]
            row["catalyst"] = nearest_event(seg.index[0], event_window)
        rows.append(row)
        prev = seg
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shock days
# ---------------------------------------------------------------------------
def market_residuals(basket: pd.Series, market: pd.Series, window: int = 252) -> pd.DataFrame:
    """Daily basket return net of its rolling (lagged) market beta: the insurance-specific move."""
    df = pd.concat({"b": basket, "m": market}, axis=1).dropna()
    cov = df["b"].rolling(window, min_periods=120).cov(df["m"])
    var = df["m"].rolling(window, min_periods=120).var()
    beta = (cov / var).shift(1)
    alpha = (df["b"] - beta * df["m"]).rolling(window, min_periods=120).mean().shift(1)
    resid = df["b"] - alpha - beta * df["m"]
    return pd.DataFrame({"basket": df["b"], "market": df["m"], "beta": beta, "residual": resid}).dropna()


def shock_days(res: pd.DataFrame, breadth: pd.Series, z_cut: float = 3.0, window: int = 63) -> pd.DataFrame:
    """Days whose insurance-specific residual exceeds z_cut rolling standard deviations."""
    vol = res["residual"].rolling(window).std().shift(1)
    d = pd.DataFrame({
        "residual_pct": res["residual"] * 100, "z": res["residual"] / vol,
        "basket_pct": res["basket"] * 100, "sp500_pct": res["market"] * 100,
        "carriers_down_pct": breadth.reindex(res.index),
    }).dropna()
    d = d[d["z"].abs() >= z_cut].copy()
    d["direction"] = np.where(d["z"] > 0, "Sector-specific rally", "Sector-specific sell-off")
    d["catalyst"] = [nearest_event(i, 3) for i in d.index]
    return d.sort_values("z", key=abs, ascending=False)


def influence(values: pd.Series) -> pd.DataFrame:
    """Leave-one-out influence of each company on the cross-sectional mean, with robust z."""
    v = values.dropna()
    full = v.mean()
    loo = pd.Series({k: v.drop(k).mean() for k in v.index})
    infl = full - loo  # how much the company lifts (+) or drags (-) the average
    return pd.DataFrame({"value": v, "influence_pp": infl, "robust_z": robust_z(v)})


# ---------------------------------------------------------------------------
# Year-by-year significance of fundamental shifts
# ---------------------------------------------------------------------------
def year_signals(fund: pd.DataFrame, tr: pd.DataFrame, tickers: list[str], years: range) -> pd.DataFrame:
    rows = []
    rel_w = None
    if "^GSPC" in tr:
        b = basket_returns(tr, tickers)
        m = tr["^GSPC"].pct_change(fill_method=None)
        rel_w = ((1 + (b - m).dropna()).resample("W-FRI").prod() - 1)
    for y in years:
        cur = fund[(fund["fiscal_year"] == y) & fund["ticker"].isin(tickers)].set_index("ticker")
        prv = fund[(fund["fiscal_year"] == y - 1) & fund["ticker"].isin(tickers)].set_index("ticker")
        both = cur.index.intersection(prv.index)

        def paired(col):
            a, b_ = cur.loc[both, col], prv.loc[both, col]
            ok = a.notna() & b_.notna()
            if ok.sum() < 6:
                return np.nan, np.nan, int(ok.sum())
            return float((a[ok] - b_[ok]).median()), float(stats.wilcoxon(a[ok] - b_[ok]).pvalue), int(ok.sum())

        def one(col):
            x = cur[col].dropna()
            if len(x) < 6:
                return np.nan, np.nan
            return float(x.median()), float(stats.wilcoxon(x).pvalue)

        lr_d, lr_p, lr_n = paired("loss_ratio_pct")
        rote_d, rote_p, _ = paired("rote_pct")
        tbv_m, tbv_p = one("tbvps_growth_pct")
        aoci_d, aoci_p, _ = paired("aoci_to_tbv_pct")
        prem_m, prem_p = one("premium_growth_pct")
        ex, ex_p = np.nan, np.nan
        if rel_w is not None:
            s = rel_w[rel_w.index.year == y]
            if len(s) > 10:
                ex = float(((1 + s).prod() - 1) * 100)
                ex_p = float(stats.ttest_1samp(s, 0).pvalue)
        rows.append({
            "year": y,
            "loss_ratio_median": cur["loss_ratio_pct"].median(), "loss_ratio_change_pp": lr_d, "loss_ratio_p": lr_p, "n": lr_n,
            "rote_median": cur["rote_pct"].median(), "rote_change_pp": rote_d, "rote_p": rote_p,
            "tbvps_growth_median": tbv_m, "tbvps_growth_p": tbv_p,
            "aoci_to_tbv_change_pp": aoci_d, "aoci_p": aoci_p,
            "premium_growth_median": prem_m, "premium_growth_p": prem_p,
            "excess_return_pct": ex, "excess_return_p": ex_p,
        })
    return pd.DataFrame(rows)


def year_theme(r: pd.Series, alpha: float = 0.05) -> tuple[str, list[str]]:
    """Name the year's dominant theme from its statistically significant shifts only."""
    sig = lambda p: pd.notna(p) and p < alpha  # noqa: E731
    notes = []
    if sig(r["aoci_p"]) and r["aoci_to_tbv_change_pp"] < 0:
        notes.append(f"unrealized bond losses cut tangible book (AOCI {r['aoci_to_tbv_change_pp']:+.0f} pp of TBV)")
    if sig(r["aoci_p"]) and r["aoci_to_tbv_change_pp"] > 0:
        notes.append(f"bond-portfolio marks recovered (AOCI {r['aoci_to_tbv_change_pp']:+.0f} pp of TBV)")
    if sig(r["loss_ratio_p"]):
        notes.append(f"loss ratios {'deteriorated' if r['loss_ratio_change_pp'] > 0 else 'improved'} "
                     f"({r['loss_ratio_change_pp']:+.1f} pp median)")
    if sig(r["rote_p"]):
        notes.append(f"returns on tangible equity {'rose' if r['rote_change_pp'] > 0 else 'fell'} ({r['rote_change_pp']:+.1f} pp)")
    if sig(r["premium_growth_p"]) and r["premium_growth_median"] >= 10:
        notes.append(f"double-digit premium growth ({r['premium_growth_median']:.0f}% median)")
    if sig(r["excess_return_p"]):
        notes.append(f"stocks {'beat' if r['excess_return_pct'] > 0 else 'lagged'} the S&P 500 by {abs(r['excess_return_pct']):.0f} pp")

    aoci_hit = sig(r["aoci_p"]) and r["aoci_to_tbv_change_pp"] <= -10  # material only
    lr_up = sig(r["loss_ratio_p"]) and r["loss_ratio_change_pp"] >= 1
    lr_down = sig(r["loss_ratio_p"]) and r["loss_ratio_change_pp"] <= -1
    if aoci_hit and lr_up:
        title = "Double squeeze: rate shock and loss inflation"
    elif aoci_hit:
        title = "Rate shock hits book value"
    elif lr_up:
        title = "Loss-cost inflation outruns pricing"
    elif lr_down and sig(r["rote_p"]) and r["rote_change_pp"] > 0:
        title = "Underwriting recovery lifts returns"
    elif lr_down:
        title = "Pricing catches up with loss trend"
    elif sig(r["rote_p"]) and r["rote_change_pp"] > 0 and sig(r["aoci_p"]) and r["aoci_to_tbv_change_pp"] > 0:
        title = "Rebound: earnings and book value recover"
    elif sig(r["rote_p"]) and r["rote_change_pp"] > 0:
        title = "Earnings power expands"
    elif sig(r["premium_growth_p"]) and r["premium_growth_median"] >= 10:
        title = "Hard-market growth"
    elif notes:
        title = "Market-led year"
    else:
        title = "No statistically significant shift"
    return title, notes
