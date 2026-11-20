"""Insight engine: every card is a hypothesis test on the project's own data, with evidence attached."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

import theme as T

SRC_SEC = "SEC EDGAR 10-K XBRL"
SRC_PX = "Yahoo Finance prices"


@dataclass
class Insight:
    key: str
    theme: str
    headline: str
    value: str
    value_label: str
    evidence: str
    p: float
    n: int
    source: str
    accent: str = T.BLUE
    fig: Callable[[], go.Figure] | None = None
    table: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def weight(self) -> float:
        return float(-np.log10(max(self.p, 1e-12)))


def _scatter_fit(df: pd.DataFrame, x: str, y: str, xt: str, yt: str, title: str) -> go.Figure:
    d = df.dropna(subset=[x, y])
    fig = go.Figure()
    # Label only points that stand apart (robust |z| > 0.8 on either axis); hover shows every name
    from analysis import robust_z
    far = (robust_z(d[x]).abs() > 0.8) | (robust_z(d[y]).abs() > 0.8)
    for seg, col in T.SEGMENT_COLORS.items():
        s = d[d["segment"] == seg]
        if len(s):
            fig.add_scatter(x=s[x], y=s[y], mode="markers+text", name=seg,
                            text=[str(n) if far[t] else "" for t, n in zip(s.index, s.get("name", s.index))],
                            hovertext=list(s.get("name", s.index)), hovertemplate="%{hovertext}<br>%{x:.1f}, %{y:.1f}<extra></extra>",
                            textposition="top center", marker=dict(size=11, color=col, line=dict(width=1, color="white")),
                            textfont=dict(size=10.5, color=T.INK))
    b, a = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 20)
    fig.add_scatter(x=xs, y=a + b * xs, mode="lines", name="Linear fit", line=dict(color=T.SLATE, dash="dash", width=1.5))
    fig.update_layout(title=title, height=380, xaxis_title=xt, yaxis_title=yt, showlegend=True, margin=dict(b=110))
    return fig


def _bars(s: pd.Series, title: str, unit: str, ref: float | None = None) -> go.Figure:
    s = s.dropna().sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[T.POS if v >= 0 else T.NEG for v in s.values],
                           text=[f"{v:+.1f}{unit}" for v in s.values], textposition="outside", cliponaxis=False))
    med = s.median()
    fig.add_vline(x=med, line_dash="dash", line_color=T.MAGENTA)
    fig.add_annotation(x=med, y=1, yref="paper", yanchor="bottom", text=f"median {med:+.1f}{unit}", showarrow=False,
                       font=dict(color=T.NAVY, size=11))
    lo, hi = min(0, s.min()), max(0, s.max())
    pad = (hi - lo) * 0.18 + 0.5
    fig.update_layout(title=title, height=max(300, 18 * len(s) + 90), showlegend=False,
                      xaxis=dict(range=[lo - (pad if lo < 0 else 0), hi + pad]), margin=dict(t=60, b=30))
    return fig


def beta_shift(res: pd.DataFrame, days: int = 252) -> tuple[float, float, float]:
    """OLS with an interaction term: is the basket's market beta different in the last year vs the first?"""
    first, last = res.iloc[:days], res.iloc[-days:]
    d = pd.concat([first.assign(late=0.0), last.assign(late=1.0)])
    X = np.column_stack([np.ones(len(d)), d["market"], d["late"], d["market"] * d["late"]])
    yv = d["basket"].values
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    sigma2 = resid @ resid / (len(yv) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = coef[3] / se[3]
    p = 2 * stats.t.sf(abs(t), len(yv) - X.shape[1])
    return float(coef[1]), float(coef[1] + coef[3]), float(p)


def build(ctx: dict) -> list[Insight]:
    fund, dec, res, shocks, ys = ctx["fund"], ctx["decomp"], ctx["res"], ctx["shocks"], ctx["years"]
    y0, y1 = ctx["y0"], ctx["y1"]
    out: list[Insight] = []
    d = dec.set_index("ticker")

    # 1. Re-rating vs book growth as the driver of TSR dispersion
    ok = d.dropna(subset=["multiple_change_pp", "tsr_pct", "tbvps_growth_pp"])
    if len(ok) >= 8:
        r_m, p_m = stats.spearmanr(ok["multiple_change_pp"], ok["tsr_pct"])
        r_b, p_b = stats.spearmanr(ok["tbvps_growth_pp"], ok["tsr_pct"])
        if p_m < 0.10:
            out.append(Insight(
                "rerating", "Value drivers", "Re-rating, not book growth, separated winners from laggards",
                f"ρ = {r_m:.2f}", "rank correlation: multiple change vs TSR",
                f"Across {len(ok)} carriers ({y0}–{y1}), P/TBV re-rating tracks TSR rank (p={p_m:.3f}); "
                f"TBV/share growth does not (ρ={r_b:.2f}, p={p_b:.2f}).",
                p_m, len(ok), f"{SRC_SEC} · {SRC_PX}", T.BLUE,
                lambda: _scatter_fit(ok, "multiple_change_pp", "tsr_pct", "P/TBV re-rating (pp/yr)", "TSR (%/yr)",
                                     "Re-rating vs total shareholder return"),
                ok[["name", "tsr_pct", "tbvps_growth_pp", "multiple_change_pp", "cash_yield_pp"]].round(1)))

    # 2. Valuation mean reversion
    ok = d.dropna(subset=["p_tbv_start", "multiple_change_pp"])
    if len(ok) >= 8:
        r, p = stats.spearmanr(ok["p_tbv_start"], ok["multiple_change_pp"])
        if p < 0.10:
            out.append(Insight(
                "meanrev", "Valuation", f"Cheap in {y0}, re-rated by {y1}: valuations mean-reverted",
                f"ρ = {r:.2f}", f"starting P/TBV vs subsequent re-rating",
                f"Carriers that started at the lowest price-to-tangible-book gained the most multiple "
                f"({'p<0.001' if p < 0.001 else f'p={p:.3f}'}); premium-multiple names compressed.",
                p, len(ok), f"{SRC_SEC} · {SRC_PX}", T.NAVY,
                lambda: _scatter_fit(ok, "p_tbv_start", "multiple_change_pp", f"P/TBV at end-{y0} (x)",
                                     "Re-rating contribution (pp/yr)", "Starting valuation vs re-rating"),
                ok[["name", "p_tbv_start", "p_tbv_end", "multiple_change_pp"]].round(2)))

    # 3. Year-level fundamental shifts (generated for every year; only material + significant ones kept)
    piv = lambda col: fund.pivot(index="ticker", columns="fiscal_year", values=col)  # noqa: E731
    specs = [
        ("tbvps_growth_pct", "level", "Book value", "%", "TBV/share growth"),
        ("loss_ratio_pct", "change", "Underwriting", " pp", "Loss ratio change"),
        ("rote_pct", "change", "Profitability", " pp", "RoTE change"),
        ("aoci_to_tbv_pct", "change", "Rates & book value", " pp", "AOCI as % of TBV, change"),
        ("p_tbv", "change", "Valuation", "x", "P/TBV change"),
        ("premium_growth_pct", "change", "Growth", " pp", "Premium growth, change"),
    ]
    for col, kind, theme, unit, label in specs:
        P = piv(col)
        for yr in range(y0 + 1, y1 + 1):
            if yr not in P or (kind == "change" and yr - 1 not in P):
                continue
            s = (P[yr] - P[yr - 1]) if kind == "change" else P[yr]
            s = s.dropna()
            if len(s) < 8:
                continue
            pv = stats.wilcoxon(s).pvalue
            med = s.median()
            material = {"tbvps_growth_pct": med < 0 or abs(med) >= 15, "loss_ratio_pct": abs(med) >= 1.5,
                        "rote_pct": abs(med) >= 4, "aoci_to_tbv_pct": abs(med) >= 8, "p_tbv": abs(med) >= 0.15,
                        "premium_growth_pct": abs(med) >= 3}[col]
            if pv >= 0.10 or not material:
                continue
            share = (s > 0).mean() if med > 0 else (s < 0).mean()
            fmt = f"{med:+.2f}{unit}" if unit == "x" else f"{med:+.1f}{unit}"
            head = {
                "tbvps_growth_pct": f"{yr}: tangible book per share {'fell' if med < 0 else 'jumped'} across the sector",
                "loss_ratio_pct": f"{yr}: loss ratios {'deteriorated' if med > 0 else 'improved'} broadly",
                "rote_pct": f"{yr}: returns on tangible equity {'compressed' if med < 0 else 'expanded'}",
                "aoci_to_tbv_pct": f"{yr}: bond marks {'eroded' if med < 0 else 'restored'} book value",
                "p_tbv": f"{yr}: valuation multiples {'compressed' if med < 0 else 'expanded'}",
                "premium_growth_pct": f"{yr}: premium growth {'decelerated' if med < 0 else 'accelerated'}",
            }[col]
            ctxline = ""
            if col == "p_tbv" and yr in piv("tbvps_growth_pct"):
                g = piv("tbvps_growth_pct")[yr].median()
                if med < 0 and g > 0:
                    ctxline = f" TBV/share grew {g:+.0f}% at the same time: the market is discounting the cycle ahead."
                elif med > 0 and g < 0:
                    head = f"{yr}: multiples expanded only because book value fell"
                    ctxline = (f" TBV/share fell {g:.0f}% (bond marks) while prices held, so P/TBV rose mechanically"
                               " rather than from re-rating optimism.")
            accent = T.RED if (col == "loss_ratio_pct" and med > 0) or (col != "loss_ratio_pct" and med < 0) else T.BLUE
            out.append(Insight(
                f"{col}-{yr}", theme, head, fmt, f"median · {share:.0%} of carriers moved this way",
                f"Wilcoxon signed-rank across {len(s)} carriers{'' if kind == 'level' else f', {yr} vs {yr - 1}'}."
                + ctxline,
                pv, len(s), SRC_SEC + (" · " + SRC_PX if col == "p_tbv" else ""), accent,
                (lambda s=s, yr=yr, label=label, unit=unit: _bars(s.rename(index=lambda t: t), f"{label}, {yr}",
                                                                   "" if unit == "x" else unit.strip())),
                s.rename(label).to_frame().round(2)))

    # 4. Market beta decoupling
    if len(res) > 600:
        b_first, b_last, p = beta_shift(res)
        if p < 0.10:
            def _beta_fig(res=res):
                rb = res["beta"].resample("W-FRI").last()
                fig = go.Figure(go.Scatter(x=rb.index, y=rb, mode="lines", line=dict(color=T.NAVY, width=2), name="Beta"))
                fig.add_hline(y=1, line_dash="dot", line_color=T.SLATE)
                fig.add_hline(y=0, line_color=T.MID)
                fig.update_layout(title="Rolling 1-year beta of the P&C basket to the S&P 500", height=300, showlegend=False, margin=dict(b=30))
                return fig
            out.append(Insight(
                "beta", "Market signal", "P&C stocks have decoupled from the broad market",
                f"β {b_first:.2f} → {b_last:.2f}", "market beta, first vs latest year",
                "Equal-weight basket regressed on S&P 500 daily returns; the interaction term tests the beta shift.",
                p, len(res), SRC_PX, T.MAGENTA, _beta_fig))

    # 5. Asymmetric sector-specific shocks
    if len(shocks) >= 8:
        neg = int((shocks["z"] < 0).sum())
        p = stats.binomtest(neg, len(shocks), 0.5).pvalue
        if p < 0.10:
            top = shocks.iloc[0]

            def _shock_fig(shocks=shocks):
                s = shocks.sort_index()
                fig = go.Figure(go.Bar(x=s.index, y=s["residual_pct"], marker_color=[T.POS if v > 0 else T.NEG for v in s["residual_pct"]],
                                       hovertext=s["catalyst"].fillna("Unattributed"), width=1000 * 3600 * 24 * 6))
                fig.update_layout(title="Sector-specific shock days (residual return, %)", height=300, showlegend=False, margin=dict(b=30))
                return fig
            out.append(Insight(
                "shocks", "Market signal", "Sector-specific shocks skew to the downside",
                f"{neg} of {len(shocks)}", "extreme sector-specific days were sell-offs",
                f"Market-model residuals (beta removed). Largest: {top.name:%b %d, %Y} "
                f"({top['residual_pct']:+.1f}%{', ' + top['catalyst'] if isinstance(top['catalyst'], str) else ''}).",
                p, len(shocks), SRC_PX, T.RED, _shock_fig,
                shocks[["residual_pct", "z", "carriers_down_pct", "catalyst"]].round(2)))

    return out


def outlier_cards(dec: pd.DataFrame, y0: int, y1: int) -> list[Insight]:
    """Statistical outliers that skew the universe average (shown with the data, not in the deck)."""
    from analysis import influence
    out: list[Insight] = []
    d = dec.set_index("ticker")
    inf = influence(d["tsr_pct"])
    flag = inf[inf["robust_z"].abs() > 2.5]
    for t, r in flag.iterrows():
        out.append(Insight(
            f"outlier-{t}", "Data quality", f"{d.loc[t, 'name']} is a statistical outlier",
            f"z = {r['robust_z']:.1f}", f"robust z-score of {y0}–{y1} TSR",
            f"TSR {r['value']:.1f}%/yr vs universe median {d['tsr_pct'].median():.1f}%; on its own it moves the "
            f"universe average by {r['influence_pp']:+.1f} pp. Toggle 'Exclude outliers' to remove it.",
            float(2 * stats.norm.sf(abs(r["robust_z"]))), len(inf), SRC_PX, T.SLATE,
            lambda inf=inf: _bars(inf["influence_pp"], "Leave-one-out influence on average TSR", " pp"),
            inf.round(2)))
    return out


def _priority(i: Insight, latest_year: int) -> float:
    w = min(i.weight, 4.0)  # cap so near-zero p-values don't all tie at the top
    if i.key.endswith(str(latest_year)):
        w += 2.5  # what changed most recently
    if i.key in ("rerating", "meanrev", "beta"):
        w += 2.0  # cross-sectional / market-structure findings
    return w


def deal(pool: list[Insight], k: int, seed: int, latest_year: int = 0) -> list[Insight]:
    """Seed 0 = editorial priority (latest year, structural findings); later seeds draw weighted by strength."""
    if not pool:
        return []
    if seed == 0:
        ranked = sorted(pool, key=lambda i: -_priority(i, latest_year))
    else:
        rng = np.random.default_rng(seed)
        w = np.array([i.weight for i in pool]) + 0.5
        idx = rng.choice(len(pool), size=len(pool), replace=False, p=w / w.sum())
        ranked = [pool[i] for i in idx]
    picked, themes = [], set()
    for ins in ranked:
        family = ins.theme.lower().split("& ")[-1]  # "Rates & book value" and "Book value" share a family
        if family not in themes:
            picked.append(ins)
            themes.add(family)
        if len(picked) == k:
            break
    return picked
