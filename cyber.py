"""Cyber & AI insurance tracker: sourced market facts + coverage timeline from the database."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import db
import theme as T


def facts() -> pd.DataFrame:
    return db.q("SELECT * FROM context_facts WHERE domain = 'Cyber'")


def series(f: pd.DataFrame, metric: str) -> pd.Series:
    d = f[f["metric"] == metric]
    return pd.Series(pd.to_numeric(d["value"]).values, index=d["period"].astype(str)).sort_index()


def ai_events() -> pd.DataFrame:
    e = db.q("SELECT * FROM events WHERE category IN ('AI coverage','AI exclusion','Cyber') ORDER BY date")
    e["date"] = pd.to_datetime(e["date"])
    return e


def market_fig(f: pd.DataFrame) -> go.Figure:
    prem = series(f, "US cyber direct written premium (incl. alien surplus lines)")
    dom = series(f, "US cyber DWP - US-domiciled insurers")
    lr = series(f, "US cyber industry loss ratio")
    yrs = sorted(set(prem.index) | set(dom.index) | set(lr.index))
    prem, dom, lr = prem.reindex(yrs), dom.reindex(yrs), lr.reindex(yrs)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=prem.index, y=prem.values, name="Total US cyber DWP ($B, incl. alien surplus lines)", marker_color=T.NAVY,
                text=[f"${v:.1f}B" if pd.notna(v) else "" for v in prem.values], textposition="outside", cliponaxis=False)
    fig.add_bar(x=dom.index, y=dom.values, name="US-domiciled insurers ($B)", marker_color=T.BLUE_TINT)
    fig.add_scatter(x=lr.index, y=lr.values, name="Loss ratio (%; 2021 = top-20 average)", mode="lines+markers+text",
                    line=dict(color=T.RED, width=2.4), connectgaps=True,
                    text=[f"{v:.0f}%" if pd.notna(v) else "" for v in lr.values], textposition="top center", textfont=dict(color=T.RED, size=11),
                    secondary_y=True)
    fig.update_layout(title="US cyber: premium has stalled while losses climb", height=380, barmode="group",
                      xaxis=dict(type="category"), margin=dict(t=56, b=80))
    fig.update_yaxes(title_text="$B", secondary_y=False, range=[0, 12])
    fig.update_yaxes(title_text="Loss ratio %", secondary_y=True, range=[0, 80], showgrid=False)
    return fig


def rates_fig(f: pd.DataFrame) -> go.Figure:
    us = series(f, "US cyber rate change (Marsh GIMI)")
    gl = series(f, "Global cyber rate change (Marsh GIMI)")
    allq = [f"{y}-Q{q}" for y in range(2021, 2027) for q in range(1, 5)]
    allq = [q for q in allq if q <= max(list(us.index) + list(gl.index))]
    fig = go.Figure()
    fig.add_bar(x=allq, y=[us.get(q, None) for q in allq], name="US cyber rate change (Marsh)",
                marker_color=[T.RED if (us.get(q) or 0) > 0 else T.BLUE for q in allq],
                text=[f"{us[q]:+.0f}%" if q in us else "" for q in allq], textposition="outside", cliponaxis=False)
    fig.add_scatter(x=list(gl.index), y=gl.values, mode="markers+lines", name="Global cyber (Marsh)",
                    line=dict(color=T.NAVY, dash="dot"), marker=dict(size=8))
    fig.add_vrect(x0="2022-Q4", x1="2025-Q3", fillcolor=T.LIGHT, opacity=0.6, line_width=0)
    fig.add_annotation(x="2024-Q1", y=60, text="Quarterly figures not captured<br>(declines since Q2 2023 per Marsh)",
                       showarrow=False, font=dict(size=11, color=T.SLATE))
    fig.add_hline(y=0, line_color=T.MID)
    fig.update_layout(title="Cyber pricing: from +130% to a dozen straight quarters of cuts", height=380,
                      xaxis=dict(type="category", tickangle=-45), yaxis=dict(range=[-20, 150]), margin=dict(t=56, b=90))
    return fig


def claims_fig(f: pd.DataFrame) -> go.Figure:
    m = {
        "Initial ransom demand": "Initial ransom demand change (Coalition)",
        "BEC frequency": "BEC claims frequency change (Coalition)",
        "Overall claims frequency": "Claims frequency change (Coalition)",
        "Funds transfer fraud frequency": "Funds transfer fraud frequency change (Coalition)",
        "Overall claims severity": "Claims severity change (Coalition)",
    }
    vals = {k: float(f.loc[f["metric"] == v, "value"].iloc[0]) for k, v in m.items() if (f["metric"] == v).any()}
    s = pd.Series(vals).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[T.RED if v > 0 else T.BLUE for v in s.values],
                           text=[f"{v:+.0f}%" for v in s.values], textposition="outside", cliponaxis=False))
    fig.add_vline(x=0, line_color=T.SLATE)
    fig.update_layout(title="Claims in 2025: attackers ask for more, pay out less (y/y %)", height=330, showlegend=False,
                      xaxis=dict(range=[-35, 60]), margin=dict(t=56, b=30))
    return fig


def ai_timeline_fig(e: pd.DataFrame) -> go.Figure:
    a = e[e["category"].isin(["AI coverage", "AI exclusion"])].copy()
    a["n"] = 1
    fig = go.Figure()
    for cat, col, sym in [("AI coverage", T.BLUE, "triangle-up"), ("AI exclusion", T.RED, "triangle-down")]:
        d = a[a["category"] == cat].sort_values("date")
        d["cum"] = d["n"].cumsum()
        fig.add_scatter(x=d["date"], y=d["cum"], mode="lines+markers", line=dict(shape="hv", color=col, width=2.4),
                        marker=dict(symbol=sym, size=11, color=col), name="Affirmative AI cover launched" if cat == "AI coverage" else "AI exclusion actions",
                        customdata=np.stack([d["label"], d["detail"]], axis=1),
                        hovertemplate="%{x|%b %Y}<br><b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>")
    fig.update_layout(title="AI coverage fork: affirmative cover vs exclusions", height=340,
                      yaxis=dict(title="Cumulative count", dtick=1), margin=dict(t=56, b=70))
    return fig


def specialist_fig(tr: pd.DataFrame) -> go.Figure | None:
    if "HSX.L" not in tr:
        return None
    from universe import CARRIERS
    start = "2020-12-31"
    h = tr["HSX.L"].dropna()
    h = h[h.index >= start]
    b = tr.loc[tr.index >= start, [c["ticker"] for c in CARRIERS]].pct_change(fill_method=None).mean(axis=1).fillna(0)
    bi = (1 + b).cumprod() * 100
    fig = go.Figure()
    fig.add_scatter(x=h.index, y=h / h.iloc[0] * 100, name="Hiscox (cyber specialist, LSE)", line=dict(color=T.MAGENTA, width=2))
    fig.add_scatter(x=bi.index, y=bi, name="US P&C carriers (equal weight)", line=dict(color=T.NAVY, width=2))
    fig.add_vline(x=pd.Timestamp("2026-10-01").timestamp() * 1000, line_dash="dot", line_color=T.SLATE)
    fig.add_annotation(x=pd.Timestamp("2026-10-01"), y=1, yref="paper", text="Zurich closes Beazley deal", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(size=11, color=T.NAVY))
    fig.update_layout(title="Cyber specialist equity vs US P&C carriers (end-2020 = 100)", height=340, margin=dict(t=56, b=70))
    return fig
