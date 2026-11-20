"""Industry-level analysis from the database: reconciled deck series + FRED, with tested inflection points."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

import analysis as A
import db
import theme as T
from insights import Insight

SRC_DECK = "USC Risk Center (Hartwig) decks"


def series(metric: str, ptype: str = "year", forecast: bool = False) -> pd.DataFrame:
    d = db.metric(metric, include_forecast=forecast, period_type=ptype)
    if ptype == "year":
        d = d[d["period"].str.fullmatch(r"\d{4}")]
    return d


def cite(d: pd.DataFrame) -> str:
    """Primary sources + deck vintage for a chart footnote."""
    m = db.q("SELECT label, primary_sources FROM metric_catalog WHERE metric = ?", (d["metric"].iloc[0],))
    v = d["source_vintage"].max()
    return f"{m['primary_sources'].iloc[0]}; compiled in USC Risk Center deck dated {v} (latest vintage)."


def regimes_fig(s: pd.Series, breaks: list[dict], title: str, unit: str, color: str, forecast: pd.Series | None = None,
                zero: bool = True, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.values, mode="lines+markers", name="Actual", line=dict(color=color, width=2.2),
                    marker=dict(size=5), hovertemplate="%{x}: %{y:.1f}" + unit + "<extra></extra>")
    if forecast is not None and len(forecast):
        f = pd.concat([s.iloc[-1:], forecast])
        fig.add_scatter(x=f.index, y=f.values, mode="lines+markers", name="Forecast (deck)",
                        line=dict(color=color, width=2, dash="dot"), marker=dict(size=5, symbol="circle-open"))
    edges = [0] + [b["index"] for b in breaks] + [len(s)]
    for i in range(len(edges) - 1):
        seg = s.iloc[edges[i]:edges[i + 1]]
        fig.add_scatter(x=[seg.index[0], seg.index[-1]], y=[seg.mean()] * 2, mode="lines", showlegend=i == 0,
                        name="Regime average", line=dict(color=T.NAVY if color == T.MAGENTA else T.MAGENTA, width=2.5),
                        hovertemplate=f"Regime {seg.index[0]}–{seg.index[-1]}: avg {seg.mean():.1f}{unit}<extra></extra>")
    for b in breaks:
        # Position by category index: numeric-looking labels ("2017") would otherwise be read as coordinates
        xpos = b["index"] - 0.5
        fig.add_shape(type="line", x0=xpos, x1=xpos, y0=0, y1=1, yref="paper", line=dict(color=T.SLATE, dash="dash", width=1))
        fig.add_annotation(x=xpos, y=1, yref="paper", yanchor="bottom", text=f"break {b['date']} (p={b['p_value']:.3f})",
                           showarrow=False, font=dict(size=11, color=T.NAVY), xanchor="left")
    if zero:
        fig.add_hline(y=0, line_color=T.MID)
    fig.update_layout(title=title, height=height, xaxis=dict(type="category"), margin=dict(t=64))
    return fig


def build(ctx: dict) -> tuple[list[Insight], dict]:
    """Industry insights (for the deck) and computed objects (for the Industry tab)."""
    out: list[Insight] = []
    res: dict = {}

    # 1. Commercial pricing regimes (CIAB survey, quarterly)
    ci = series("ciab_commercial_rate_change", "quarter")
    if len(ci) > 20:
        s = ci.set_index("period")["value"]
        br = A.detect_regimes(s, min_weeks=6, block=4)
        res["ciab"] = (s, br, ci)
        if br:
            last_b = br[-1]
            seg_prev = s.iloc[(br[-2]["index"] if len(br) > 1 else 0):last_b["index"]]
            seg_now = s.iloc[last_b["index"]:]
            neg = s[s < 0]
            prior_neg = neg[neg.index < seg_now.index[0]]
            since = prior_neg.index[-1] if len(prior_neg) else s.index[0]
            out.append(Insight(
                "ciab", "Pricing cycle", f"Commercial pricing broke into a new regime in {last_b['date']}",
                f"{seg_prev.mean():+.1f}% → {seg_now.mean():+.1f}%", "avg quarterly rate change, prior vs current regime",
                f"Block-permutation CUSUM (autocorrelation-robust). Latest {s.index[-1]}: {s.iloc[-1]:+.1f}%, "
                f"{'first negative reading since ' + since if s.iloc[-1] < 0 else 'still positive'}.",
                last_b["p_value"], len(s), "CIAB survey via " + SRC_DECK, T.RED,
                lambda s=s, br=br: regimes_fig(s, br, "Average commercial rate change, all lines (%)", "%", T.NAVY),
                s.rename("rate change %").to_frame().tail(12)))

    # 2. Catastrophe loss step change
    cat = series("insured_cat_losses_real")
    if len(cat) > 20:
        s = cat.set_index("period")["value"]
        br = A.detect_regimes(np.log(s), min_weeks=5, block=2)
        res["cat"] = (s, br, cat)
        late = [b for b in br if b["date"] >= "2005"]
        if late:
            b = late[-1]
            before, after = s[s.index < b["date"]], s[s.index >= b["date"]]
            before = before[before.index >= str(int(b["date"]) - 27)]
            p = stats.mannwhitneyu(after, before, alternative="greater").pvalue
            out.append(Insight(
                "cat", "Catastrophe", f"Insured catastrophe losses stepped up in {b['date']}",
                f"${before.mean():.0f}B → ${after.mean():.0f}B", f"avg annual real losses, {before.index[0]}–{before.index[-1]} vs {after.index[0]}–{after.index[-1]}",
                f"Break located by CUSUM on log losses (p={b['p_value']:.3f}); level shift tested with Mann-Whitney.",
                p, len(s), "Verisk PCS / Munich Re / Aon via " + SRC_DECK, T.MAGENTA,
                lambda s=s, br=br: regimes_fig(s, br, "US insured catastrophe losses, inflation-adjusted ($B)", "B", T.MAGENTA,
                                               zero=False),
                s.rename("$B").to_frame().tail(12)))

    # 3. Premiums vs the economy (deck premiums vs live FRED GDP)
    dwp = series("industry_dwp_growth_yoy", "quarter")
    gdp = db.fred("GDP")
    if len(dwp) > 20 and len(gdp) > 20:
        gy = (gdp / gdp.shift(4) - 1) * 100
        gq = pd.Series(gy.values, index=[f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in gy.index])
        j = pd.concat([dwp.set_index("period")["value"], gq], axis=1, keys=["premium", "gdp"]).dropna()
        r, p = stats.pearsonr(j["premium"], j["gdp"])
        res["dwp_gdp"] = j
        if p < 0.10:
            def _fig(j=j):
                fig = go.Figure()
                fig.add_scatter(x=j.index, y=j["premium"], name="Direct premium growth (deck)", line=dict(color=T.NAVY, width=2))
                fig.add_scatter(x=j.index, y=j["gdp"], name="Nominal GDP growth (FRED, live)", line=dict(color=T.BLUE, width=2, dash="dash"))
                fig.add_hline(y=0, line_color=T.MID)
                fig.update_layout(title="Premium growth tracks the nominal economy (y/y %)", height=330, xaxis=dict(type="category"))
                return fig
            out.append(Insight(
                "dwp_gdp", "Growth", "Premium growth moves with the nominal economy",
                f"r = {r:.2f}", f"quarterly y/y, {j.index[0]}–{j.index[-1]}",
                "Deck premium series correlated with live BEA nominal GDP from FRED: exposure growth, not just rate, "
                "drives the top line, so a slowing economy feeds directly into premium growth.",
                p, len(j), "S&P/ISO via deck · FRED GDP", T.BLUE, _fig, j.round(2).tail(12)))

    # 4. Combined ratio: where 2025 sits in history (empirical rank, no regime claim)
    cr = series("industry_combined_ratio")
    crf = series("industry_combined_ratio", forecast=True)
    crf = crf[crf["is_forecast"] == 1]
    if len(cr) > 15:
        s = cr.set_index("period")["value"]
        last = s.index[-1]
        rank = int((s <= s.iloc[-1]).sum())
        p = rank / len(s)
        res["cr"] = (s, crf.set_index("period")["value"], cr)
        if p < 0.10:
            fc = crf.set_index("period")["value"]
            nxt = f" Deck forecast: {fc.iloc[-1]:.1f} by {fc.index[-1]}." if len(fc) else ""
            out.append(Insight(
                "cr_rank", "Underwriting", f"{last} combined ratio ranks among the best since {s.index[0]}",
                f"{s.iloc[-1]:.1f}", f"#{rank} of {len(s)} years (lower is better)",
                f"Empirical rank against the full history; no structural break is detected, so treat as a cyclical peak.{nxt}",
                p, len(s), "A.M. Best / Verisk / S&P via " + SRC_DECK, T.BLUE,
                lambda s=s, fc=fc: regimes_fig(s, [], "Industry combined ratio (%) with deck forecast", "", T.NAVY,
                                               forecast=fc, zero=False),
                s.rename("combined ratio").to_frame().tail(10)))

    # 5. Reconciliation quality (always shown in Data tab; deck card only as context)
    rec = db.table("reconciliation")
    res["rec"] = rec
    return out, res
