"""Single-company scorecard: peer-ranked metrics, charts, sourced punchlist and a plain-English verdict."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import db
import theme as T
from current import implied_rote
from universe import UNIVERSE

U = pd.DataFrame(UNIVERSE).set_index("ticker")

# metric key, label, source frame, higher-is-better, format
METRICS = [
    ("return_ytd_pct", "Total return, YTD", "yt", True, "{:+.1f}%"),
    ("ret_1y", "Total return, 1 year", "ct", True, "{:+.1f}%"),
    ("ann_since_2020", "Total return, annualised since 2020", "ct", True, "{:+.1f}%"),
    ("tsr_pct", "5-yr TSR (decomposed)", "dec", True, "{:+.1f}%"),
    ("loss_ratio_ytd", "Loss ratio, YTD", "yt", False, "{:.1f}%"),
    ("loss_ratio_chg_pp", "Loss ratio change vs prior YTD", "yt", False, "{:+.1f} pp"),
    ("avg_lr", "Average loss ratio (5 yrs)", "fh", False, "{:.1f}%"),
    ("lr_vol", "Loss-ratio volatility (5 yrs)", "fh", False, "{:.1f} pp"),
    ("rote_ytd_ann", "RoTE, YTD annualised", "yt", True, "{:.1f}%"),
    ("avg_rote", "Average RoTE (5 yrs)", "fh", True, "{:.1f}%"),
    ("net_income_growth_pct", "Net income growth, YTD", "yt", True, "{:+.1f}%"),
    ("revenue_growth_pct", "Revenue growth, YTD", "yt", True, "{:+.1f}%"),
    ("tbv_ytd_pct", "Tangible book growth, YTD", "yt", True, "{:+.1f}%"),
    ("aoci_chg_pct_tbv", "AOCI change, YTD (% of TBV)", "yt", True, "{:+.1f} pp"),
    ("p_tbv_now", "Price / tangible book", "yt", None, "{:.2f}×"),
    ("pe", "Price / earnings (last FY)", "ct", None, "{:.1f}×"),
    ("beta_1y", "Beta to S&P 500 (1 yr)", "ct", None, "{:.2f}"),
    ("vol_1y", "Volatility (1 yr)", "ct", False, "{:.0f}%"),
]


def history(fund: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    fy = fund[fund["fiscal_year"].between(y0 + 1, y1)]
    g = fy.groupby("ticker")
    return pd.DataFrame({"avg_lr": g["loss_ratio_pct"].mean(), "lr_vol": g["loss_ratio_pct"].std(),
                         "avg_rote": g["rote_pct"].mean()})


def peers(ticker: str) -> list[str]:
    """Same sub-industry; small groups are widened to the closest broader peer set."""
    sub = U.loc[ticker, "sub_industry"]
    if sub == "Insurtech":  # compare against personal-lines underwriters
        return list(U.index[(U["sub_industry"] == "Insurtech") | (U["segment"] == "Personal")])
    grp = list(U.index[U["sub_industry"] == sub])
    if len(grp) < 4:
        grp = list(U.index[U["sub_industry"].isin([sub, "Carrier"])])
    return grp


def scorecard(ticker: str, yt, ct, dec, fh) -> pd.DataFrame:
    frames = {"yt": yt, "ct": ct.set_index("ticker"), "dec": dec.set_index("ticker") if len(dec) else pd.DataFrame(), "fh": fh}
    pr = peers(ticker)
    rows = []
    for key, label, src, hib, fmt in METRICS:
        df = frames[src]
        if key not in df or ticker not in df.index:
            continue
        v = df.loc[ticker, key]
        if pd.isna(v):
            continue
        p = df.loc[df.index.intersection(pr), key].dropna()
        if len(p) < 3:
            continue
        pct = (p < v).mean() * 100 + (p == v).mean() * 50
        if hib is False:
            pct = 100 - pct
        rows.append({"Metric": label, "Value": fmt.format(v), "Peer median": fmt.format(p.median()),
                     "Percentile vs peers": round(pct) if hib is not None else None, "_raw": v, "_pmed": p.median(), "_hib": hib, "n": len(p)})
    cols = ["Metric", "Value", "Peer median", "Percentile vs peers", "_raw", "_pmed", "_hib", "n"]
    return pd.DataFrame(rows, columns=cols)


def price_fig(ticker: str, tr: pd.DataFrame, sub_idx: pd.DataFrame) -> go.Figure:
    start = pd.Timestamp("2020-12-31")
    s = tr[ticker].dropna()
    s = s[s.index >= max(start, s.index[0])]
    sub = U.loc[ticker, "sub_industry"]
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s / s.iloc[0] * 100, name=U.loc[ticker, "name"], line=dict(color=T.BLUE, width=2.6))
    if sub in sub_idx:
        si = sub_idx[sub].reindex(s.index).ffill()
        fig.add_scatter(x=si.index, y=si / si.iloc[0] * 100, name=f"{sub} peers (equal weight)", line=dict(color=T.NAVY, width=1.8, dash="dot"))
    sp = tr["^GSPC"].reindex(s.index).ffill()
    fig.add_scatter(x=sp.index, y=sp / sp.iloc[0] * 100, name="S&P 500", line=dict(color=T.MID, width=1.8))
    ev = db.table("events")
    ev = ev[ev["tickers"].fillna("").str.contains(ticker)]
    for _, e in ev.iterrows():
        d = pd.Timestamp(e["date"])
        if d >= s.index[0]:
            y = (s / s.iloc[0] * 100).asof(d)
            fig.add_scatter(x=[d], y=[y], mode="markers", marker=dict(size=11, color=T.RED if "Catastrophe" in e["category"] or "reserves" in e["category"] else T.MAGENTA,
                                                                    symbol="diamond", line=dict(width=1, color="white")),
                            name=e["label"][:40], showlegend=False, hovertemplate=f"{e['date']}<br><b>{e['label']}</b><extra></extra>")
    fig.update_layout(title=f"Total return vs peers and the S&P 500 (rebased = 100)", height=380, margin=dict(t=56, b=80))
    return fig


def lr_fig(ticker: str, fund: pd.DataFrame, yt: pd.DataFrame) -> go.Figure | None:
    last_fy = int(yt.attrs.get("year", 9999)) - 1
    fund = fund[fund["fiscal_year"] <= last_fy]
    f = fund[fund["ticker"] == ticker].set_index("fiscal_year")["loss_ratio_pct"].dropna()
    if len(f) < 3:
        return None
    med = fund[fund["ticker"].isin(peers(ticker))].groupby("fiscal_year")["loss_ratio_pct"].median().dropna()
    fig = go.Figure()
    fig.add_scatter(x=[str(i) for i in f.index], y=f.values, name=U.loc[ticker, "name"], mode="lines+markers", line=dict(color=T.BLUE, width=2.6))
    fig.add_scatter(x=[str(i) for i in med.index], y=med.values, name="Carrier median", mode="lines", line=dict(color=T.NAVY, dash="dot"))
    if ticker in yt.index and pd.notna(yt.loc[ticker, "loss_ratio_ytd"]):
        lab = f"{yt.attrs.get('year', '')} YTD"
        fig.add_scatter(x=[lab], y=[yt.loc[ticker, "loss_ratio_ytd"]], mode="markers", name="Current YTD", marker=dict(size=12, color=T.MAGENTA, symbol="star"))
    fig.update_layout(title="Loss ratio vs carrier median (%)", height=380, xaxis=dict(type="category"), margin=dict(t=56, b=80))
    return fig


def punchlist(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev = db.table("company_evidence")
    ev = ev[ev["ticker"] == ticker].sort_values("date", ascending=False)
    ms = db.table("events")
    ms = ms[ms["tickers"].fillna("").str.contains(ticker)].sort_values("date", ascending=False)
    k8 = db.table("filings_8k")
    k8 = k8[(k8["ticker"] == ticker) & k8["labels"].str.contains("agreement|Acquisition|Restructuring|impairment|terminated", case=False)]
    return ev, ms, k8.sort_values("date", ascending=False).head(12)


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def verdict(ticker: str, sc: pd.DataFrame, yt: pd.DataFrame) -> dict:
    name = U.loc[ticker, "name"]
    strong = sc[(sc["Percentile vs peers"].notna()) & (sc["Percentile vs peers"] >= 75)]
    weak = sc[(sc["Percentile vs peers"].notna()) & (sc["Percentile vs peers"] <= 25)]
    out = {"strengths": [f"{r.Metric}: {r.Value} (peer median {r['Peer median']}, {ordinal(int(r['Percentile vs peers']))} pct)" for _, r in strong.iterrows()],
           "watch": [f"{r.Metric}: {r.Value} (peer median {r['Peer median']}, {ordinal(int(r['Percentile vs peers']))} pct)" for _, r in weak.iterrows()]}
    pt = yt.loc[ticker, "p_tbv_now"] if ticker in yt.index else np.nan
    ro = yt.loc[ticker, "rote_ytd_ann"] if ticker in yt.index else np.nan
    imp = implied_rote(pt)
    if pd.notna(imp) and pd.notna(ro):
        gap = imp - ro
        mkt = (f"At {pt:.2f}× tangible book the market is pricing a sustainable RoTE of about <b>{imp:.0f}%</b> (cost of equity 9%, growth 3%) "
               f"against <b>{ro:.0f}%</b> annualised this year. "
               + ("That's a vote of confidence: the price assumes today's returns are durable. The risk is any evidence they aren't."
                  if gap > -2 else
                  f"The {abs(gap):.0f}-point discount says investors expect this year's returns to fade as the cycle turns. "
                  "Proving a higher through-cycle floor is the re-rating opportunity."))
    else:
        pe = None
        mkt = "Tangible book is not meaningful here (distribution, goodwill-heavy or negative TBV), so read valuation through P/E and growth instead."
    sub = U.loc[ticker, "sub_industry"]
    n_s, n_w = len(out["strengths"]), len(out["watch"])
    tone = ("a franchise the numbers say is executing" if n_s > n_w + 2 else
            "a mixed picture: real strengths with specific gaps to close" if n_s >= n_w else
            "a company where the evidence says the work isn't finished")
    out["headline"] = f"{name}: {tone}."
    out["market"] = mkt
    out["sub"] = sub
    return out
