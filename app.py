from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import analysis as A
import insights as I
import theme as T
from universe import BENCHMARKS, UNIVERSE

st.set_page_config(page_title="US P&C Value Signals", layout="wide", page_icon="📈")
T.register_plotly_template()
T.inject_css()

DATA = Path(__file__).parent / "data"
CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
ALL_TICKERS = [c["ticker"] for c in UNIVERSE]
NAME = {c["ticker"]: c["name"] for c in UNIVERSE}
SEG = {c["ticker"]: c["segment"] for c in UNIVERSE}
PX_TICKERS = ALL_TICKERS + list(BENCHMARKS.values())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading live prices…")
def load_prices() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    try:
        raw = yf.download(PX_TICKERS, start="2019-12-31", auto_adjust=False, progress=False, threads=False)
        tr, cl = raw["Adj Close"], raw["Close"]
        if tr.dropna(how="all").shape[0] < 500 or tr.columns.size < len(PX_TICKERS) - 2:
            raise RuntimeError("incomplete download")
        return tr.sort_index(), cl.sort_index(), "live"
    except Exception:
        tr = pd.read_csv(DATA / "prices_total_return.csv", index_col=0, parse_dates=True)
        cl = pd.read_csv(DATA / "prices_close.csv", index_col=0, parse_dates=True)
        return tr, cl, "snapshot"


@st.cache_data
def load_static() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(DATA / "fundamentals_raw.csv"), pd.read_csv(DATA / "splits.csv")


@st.cache_data(ttl=3600, show_spinner="Running statistical tests…")
def compute(active: tuple[str, ...], z_cut: float, px_tag: str):
    tr, cl, _ = load_prices()
    raw, splits = load_static()
    fund = A.build_fundamentals(raw, cl, splits)
    cover = fund.groupby("fiscal_year")["equity"].apply(lambda s: s.notna().mean())
    y1 = int(cover[cover >= 0.8].index.max())
    y0 = y1 - 5
    dec_all = A.decompose_tsr(fund, tr, cl, y0, y1)
    fund_a = fund[fund["ticker"].isin(active)]
    dec = dec_all[dec_all["ticker"].isin(active)]
    act = [t for t in active if t in tr]
    trw = tr[tr.index >= f"{y0}-12-31"]
    res = A.market_residuals(A.basket_returns(tr, act), tr["^GSPC"].pct_change(fill_method=None))
    breadth = (tr[act].pct_change(fill_method=None) < 0).mean(axis=1) * 100
    shocks = A.shock_days(res[res.index >= f"{y0}-12-31"], breadth, z_cut=z_cut)
    years = A.year_signals(fund_a, tr, act, range(y0 + 1, y1 + 1))
    rel = A.relative_daily(trw, act)
    weekly = (1 + rel).resample("W-FRI").prod() - 1
    brk = A.best_break(weekly)
    regimes = A.detect_regimes(weekly)
    return dict(fund=fund_a, fund_all=fund, decomp=dec, decomp_all=dec_all, res=res, shocks=shocks, years=years,
                y0=y0, y1=y1, rel=rel, weekly=weekly, best_break=brk, regimes=regimes)


tr, cl, px_source = load_prices()
last_px = tr.dropna(how="all").index[-1]

# ---------------------------------------------------------------------------
# Sidebar: analysis settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Analysis settings")
    excl = st.toggle("Exclude statistical outliers", value=False,
                     help="Drops carriers whose 5-year TSR has a robust (median/MAD) z-score beyond ±2.5.")
    z_cut = st.slider("Shock threshold (σ)", 2.5, 4.0, 3.0, 0.25,
                      help="A day counts as a sector-specific shock when the beta-adjusted residual exceeds this many rolling standard deviations.")
    st.markdown("**Segments**")
    seg_pick = [s for s in T.SEGMENT_COLORS if st.checkbox(s, value=True, key=f"seg_{s}")]
    st.caption("Settings recalculate every test, chart and insight card.")

base = compute(tuple(t for t in ALL_TICKERS if SEG[t] in seg_pick) or tuple(ALL_TICKERS), z_cut, px_source)
outlier_list = A.outliers(base["decomp_all"].set_index("ticker")["tsr_pct"])
active = tuple(t for t in ALL_TICKERS if SEG[t] in seg_pick and not (excl and t in outlier_list)) or tuple(ALL_TICKERS)
ctx = compute(active, z_cut, px_source)
y0, y1 = ctx["y0"], ctx["y1"]

# ---------------------------------------------------------------------------
# Header + insight deck
# ---------------------------------------------------------------------------
src_note = "live" if px_source == "live" else "snapshot (live feed unavailable)"
st.markdown(
    f'<div class="topbar"><div class="t">US P&amp;C <span>Value Signals</span></div>'
    f'<div class="m">{len(active)} carriers · SEC 10-K data FY{y0}–FY{y1} · prices through {last_px:%b %d, %Y} ({src_note})</div></div>',
    unsafe_allow_html=True,
)

pool = I.build(ctx)
if "deal" not in st.session_state:
    st.session_state.deal = 0
h1, h2 = st.columns([5, 1])
h1.markdown(f"**Market insights** &nbsp;<span style='color:{T.SLATE};font-size:.85rem'>{len(pool)} statistically "
            f"supported findings (p&lt;0.10) · showing 3</span>", unsafe_allow_html=True)
if h2.button("↻ Shuffle insights", use_container_width=True):
    st.session_state.deal += 1
cards = I.deal(pool, 3, st.session_state.deal, latest_year=y1)

cols = st.columns(3)
for col, ins in zip(cols, cards):
    cls, label = T.strength(ins.p)
    with col:
        st.markdown(
            f'<div class="icard" style="--accent:{ins.accent}">'
            f'<div class="row"><span class="chip">{ins.theme}</span><span class="badge {cls}">{label}</span></div>'
            f'<div class="h">{ins.headline}</div>'
            f'<div class="num">{ins.value}</div><div class="lbl">{ins.value_label}</div>'
            f'<div class="ev">{ins.evidence}</div>'
            f'<div class="src">n={ins.n} · {ins.source}</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Supporting evidence"):
            if ins.fig:
                st.plotly_chart(ins.fig(), width="stretch", config=CFG, key=f"ev_{ins.key}_{st.session_state.deal}")
            if ins.table is not None:
                tbl = ins.table.copy()
                tbl.index = [A.NAME.get(i, i) for i in tbl.index]
                st.dataframe(tbl, width="stretch", height=min(240, 38 + 35 * len(tbl)))

st.write("")
tabs = st.tabs(["Market signals", "Value drivers", "Fundamentals", "Year by year", "Data & methods"])

# ---------------------------------------------------------------------------
# Market signals
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("## Market signals")
    st.markdown('<p class="lede">How the equal-weight P&amp;C basket has traded against the S&amp;P 500, which days '
                "moved it for insurance-specific reasons, and whether its trend has statistically changed.</p>",
                unsafe_allow_html=True)
    res, shocks, weekly = ctx["res"], ctx["shocks"], ctx["weekly"]
    rel = ctx["rel"]
    cum_ex = ((1 + weekly).cumprod() - 1) * 100
    k = st.columns(4)
    bk = A.basket_returns(tr[tr.index >= f"{y0}-12-31"], list(active))
    yrs = (bk.index[-1] - bk.index[0]).days / 365.25
    b_ann = ((1 + bk).prod() ** (1 / yrs) - 1) * 100
    s = tr["^GSPC"].dropna()
    s = s[s.index >= f"{y0}-12-31"]
    m_ann = ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
    k[0].markdown(T.kpi(f"{b_ann:.1f}%", "P&C basket TSR, annualized", f"since end-{y0} · equal weight"), unsafe_allow_html=True)
    k[1].markdown(T.kpi(f"{b_ann - m_ann:+.1f} pp", "vs S&P 500 per year", f"S&P 500 {m_ann:.1f}%/yr"), unsafe_allow_html=True)
    k[2].markdown(T.kpi(f"{res['beta'].iloc[-1]:.2f}", "Current market beta", "rolling 1-year, lagged"), unsafe_allow_html=True)
    k[3].markdown(T.kpi(f"{len(shocks)}", "Sector-specific shock days", f"|z| ≥ {z_cut:g} · {(shocks['z'] < 0).sum()} sell-offs"),
                  unsafe_allow_html=True)
    st.write("")

    ex_shock = st.toggle("Remove shock days from the trend", value=False,
                         help="Recomputes cumulative excess return after dropping the flagged shock days.")
    fig = go.Figure()
    fig.add_scatter(x=cum_ex.index, y=cum_ex, mode="lines", name="Cumulative excess return vs S&P 500",
                    line=dict(color=T.NAVY, width=2.2))
    if ex_shock and len(shocks):
        rel_x = rel.drop(index=shocks.index, errors="ignore")
        cx = ((1 + ((1 + rel_x).resample("W-FRI").prod() - 1)).cumprod() - 1) * 100
        fig.add_scatter(x=cx.index, y=cx, mode="lines", name="Excluding shock days", line=dict(color=T.BLUE, width=2, dash="dash"))
        shock_contrib = cum_ex.iloc[-1] - cx.iloc[-1]
    if len(shocks):
        sh = shocks.copy()
        sh["y"] = cum_ex.reindex(sh.index, method="ffill").values
        for direction, colr, sym in [("Sector-specific sell-off", T.RED, "triangle-down"), ("Sector-specific rally", T.BLUE, "triangle-up")]:
            d = sh[sh["direction"] == direction]
            fig.add_scatter(x=d.index, y=d["y"], mode="markers", name=direction,
                            marker=dict(color=colr, size=11, symbol=sym, line=dict(width=1, color="white")),
                            customdata=np.stack([d["residual_pct"], d["z"], d["catalyst"].fillna("Unattributed")], axis=1),
                            hovertemplate="%{x|%b %d, %Y}<br>Residual %{customdata[0]:.1f}% (z %{customdata[1]:.1f})<br>%{customdata[2]}<extra></extra>")
        for dt, r in shocks.head(4).iterrows():
            if isinstance(r["catalyst"], str):
                fig.add_annotation(x=dt, y=cum_ex.reindex([dt], method="ffill").iloc[0], text=r["catalyst"], showarrow=True,
                                   arrowhead=0, ax=0, ay=-38 if r["z"] < 0 else 38, font=dict(size=11, color=T.NAVY),
                                   bgcolor="rgba(255,255,255,.85)")
    fig.add_hline(y=0, line_color=T.MID)
    fig.update_layout(title="Equal-weight P&C basket: cumulative return in excess of the S&P 500 (%)", height=430,
                      hovermode="closest", margin=dict(b=70))
    st.plotly_chart(fig, width="stretch", config=CFG)
    bb = ctx["best_break"]
    msg = (f"Change-point test (CUSUM, 999 permutations): the strongest candidate break in weekly excess returns is "
           f"{bb['date']:%b %d, %Y} (p = {bb['p_value']:.2f}). "
           + ("It is statistically significant." if bb["p_value"] < 0.05 else
              "It is <b>not</b> significant, so there is no evidence of a structural regime change in relative performance; "
              "the excess return is consistent with a single noisy trend."))
    if ex_shock and len(shocks):
        msg += f" Shock days account for {shock_contrib:+.1f} pp of the cumulative excess return."
    T.note(msg)

    l, r = st.columns(2)
    with l:
        rb = res["beta"].resample("W-FRI").last()
        rb = rb[rb.index >= f"{y0}-12-31"]
        fig = go.Figure(go.Scatter(x=rb.index, y=rb, mode="lines", line=dict(color=T.NAVY, width=2), fill="tozeroy",
                                   fillcolor="rgba(0,71,255,0.08)", name="Beta"))
        fig.add_hline(y=1, line_dash="dot", line_color=T.SLATE)
        fig.add_annotation(x=1, xref="paper", y=1, text="market-like (β = 1)", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=11, color=T.SLATE))
        fig.update_layout(title="Rolling 1-year beta to the S&P 500", height=360, showlegend=False, margin=dict(b=30))
        st.plotly_chart(fig, width="stretch", config=CFG)
        b_first, b_last, p_beta = I.beta_shift(res)
        T.note(f"Beta first year {b_first:.2f} vs latest year {b_last:.2f}; interaction-term test p = {p_beta:.3g}. "
               "A falling beta means the group trades on its own fundamentals (rates, catastrophe, pricing) rather than the market.")
    with r:
        fig = go.Figure()
        for seg_name, colr in T.SEGMENT_COLORS.items():
            ts = [t for t in active if SEG[t] == seg_name]
            if not ts:
                continue
            br_ = A.basket_returns(tr[tr.index >= f"{y0}-12-31"], ts)
            c = ((1 + br_).cumprod()) * 100
            fig.add_scatter(x=c.index, y=c, mode="lines", name=f"{seg_name} ({len(ts)})", line=dict(color=colr, width=2))
        sp = tr["^GSPC"][tr.index >= f"{y0}-12-31"].dropna()
        fig.add_scatter(x=sp.index, y=sp / sp.iloc[0] * 100, mode="lines", name="S&P 500", line=dict(color=T.MID, width=2))
        fig.update_layout(title=f"Total return by segment (end-{y0} = 100)", height=360, margin=dict(b=90))
        st.plotly_chart(fig, width="stretch", config=CFG)

    st.markdown("#### Shock-day log")
    if len(shocks):
        show = shocks.copy()
        show.index = show.index.strftime("%Y-%m-%d")
        show["catalyst"] = show["catalyst"].fillna("Unattributed: no catalogued event within ±3 days")
        st.dataframe(show[["direction", "residual_pct", "z", "basket_pct", "sp500_pct", "carriers_down_pct", "catalyst"]]
                     .rename(columns={"residual_pct": "Residual %", "basket_pct": "P&C basket %", "sp500_pct": "S&P 500 %",
                                      "carriers_down_pct": "% carriers down", "z": "z-score"}).round(2),
                     width="stretch", height=300)
        T.note("Residual = basket return minus rolling-beta × S&P 500 return (and rolling alpha). Catalysts are matched only "
               "after detection, from a fixed list of dated events; unmatched shocks are left unattributed.")

# ---------------------------------------------------------------------------
# Value drivers
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("## Value drivers")
    st.markdown(f'<p class="lede">Where each carrier\'s {y0}–{y1} shareholder return came from: growth in tangible book '
                "per share, change in the price-to-tangible-book multiple, and cash returned as dividends.</p>",
                unsafe_allow_html=True)
    dec = ctx["decomp"].dropna(subset=["tsr_pct"]).sort_values("tsr_pct")
    fig = go.Figure()
    comp_cols = {"TBV/share growth": "tbvps_growth_pp", "P/TBV re-rating": "multiple_change_pp", "Cash yield": "cash_yield_pp"}
    labels = [f"{n}  <b>{v:.1f}%</b>" for n, v in zip(dec["name"], dec["tsr_pct"])]
    for comp, (colr, txt) in T.COMPONENTS.items():
        v = dec[comp_cols[comp]]
        fig.add_bar(y=labels, x=v, name=comp, orientation="h", marker_color=colr,
                    text=[f"{x:.0f}" if abs(x) >= 2.5 else "" for x in v.fillna(0)], textposition="inside",
                    insidetextanchor="middle", textangle=0, constraintext="none", textfont=dict(color=txt, size=11))
    fig.add_scatter(y=labels, x=dec["tsr_pct"], mode="markers", name="Total TSR (in label)",
                    marker=dict(symbol="line-ns-open", size=20, color=T.INK, line=dict(width=3)),
                    hovertemplate="TSR %{x:.1f}%<extra></extra>")
    med = dec["tsr_pct"].median()
    fig.add_vline(x=med, line_dash="dash", line_color=T.MAGENTA)
    fig.add_annotation(x=med, y=1, yref="paper", yanchor="bottom", text=f"median {med:.1f}%", showarrow=False,
                       font=dict(color=T.NAVY, size=11))
    fig.update_layout(barmode="relative", title=f"Annualized TSR decomposition, end-{y0} to end-{y1} (pp per year)",
                      height=max(420, 30 * len(dec) + 150), margin=dict(t=64, b=70, r=20), bargap=0.22)
    st.plotly_chart(fig, width="stretch", config=CFG)
    T.note("Price return split log-additively into TBV/share growth and P/TBV change; cash yield = TSR − price return. "
           "TBV = common equity − preferred − goodwill − intangibles; shares from 10-K cover pages, split-adjusted.")

    l, r = st.columns(2)
    dd = ctx["decomp"].set_index("ticker")
    with l:
        st.plotly_chart(I._scatter_fit(dd, "multiple_change_pp", "tsr_pct", "P/TBV re-rating (pp/yr)", "TSR (%/yr)",
                                       "What explains TSR? Re-rating"), width="stretch", config=CFG)
    with r:
        st.plotly_chart(I._scatter_fit(dd, "p_tbv_start", "multiple_change_pp", f"P/TBV at end-{y0} (x)",
                                       "Re-rating (pp/yr)", "Starting valuation vs re-rating"), width="stretch", config=CFG)
    from scipy import stats as _st
    corr = []
    for c, lab in [("multiple_change_pp", "P/TBV re-rating"), ("tbvps_growth_pp", "TBV/share growth"),
                   ("cash_yield_pp", "Cash yield"), ("avg_rote_pct", "Average RoTE"), ("avg_loss_ratio_pct", "Average loss ratio")]:
        z = dd.dropna(subset=[c, "tsr_pct"])
        if len(z) >= 8:
            rho, p = _st.spearmanr(z[c], z["tsr_pct"])
            corr.append({"Driver": lab, "Spearman ρ with TSR": round(rho, 2), "p-value": round(p, 3), "n": len(z),
                         "Verdict": "significant" if p < 0.05 else ("suggestive" if p < 0.10 else "not significant")})
    st.markdown("#### Which driver explains the ranking?")
    st.dataframe(pd.DataFrame(corr), hide_index=True, width="stretch")

    st.markdown("#### Outliers and influence")
    oc = I.outlier_cards(ctx["decomp_all"], y0, y1)
    inf = A.influence(ctx["decomp_all"].set_index("ticker")["tsr_pct"])
    inf.index = [NAME[t] for t in inf.index]
    l, r = st.columns([1.3, 1])
    with l:
        st.plotly_chart(I._bars(inf["influence_pp"], "Leave-one-out influence on universe average TSR", " pp"),
                        width="stretch", config=CFG)
    with r:
        for o in oc:
            st.markdown(f'<div class="icard" style="--accent:{T.SLATE}"><div class="row"><span class="chip">Outlier</span>'
                        f'<span class="badge strong">robust {o.value}</span></div><div class="h">{o.headline}</div>'
                        f'<div class="ev">{o.evidence}</div></div>', unsafe_allow_html=True)
        st.write("")
        st.markdown(f"Outlier exclusion is **{'on' if excl else 'off'}** (sidebar). With it on, baskets, medians, "
                    "tests and insight cards are recomputed without flagged carriers.")
        seg_tbl = ctx["decomp"].groupby("segment")[["tsr_pct", "tbvps_growth_pp", "multiple_change_pp", "cash_yield_pp"]].median()
        seg_tbl.columns = ["TSR %", "TBV/sh pp", "Re-rating pp", "Cash pp"]
        st.dataframe(seg_tbl.round(1), width="stretch")
        st.caption("Segment medians (robust to single outliers).")

# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("## Fundamentals")
    st.markdown('<p class="lede">Each year\'s distribution across carriers. Stars mark years whose shift is statistically '
                "significant (Wilcoxon signed-rank: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01).</p>", unsafe_allow_html=True)
    fund = ctx["fund"]
    fy = fund[fund["fiscal_year"].between(y0 + 1, y1)]
    ysig = ctx["years"].set_index("year")

    def stars(p):
        return "" if pd.isna(p) else ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "")

    def dist(col, title, unit, pcol, color):
        fig = go.Figure()
        for y in range(y0 + 1, y1 + 1):
            v = fy[fy["fiscal_year"] == y][[col, "ticker"]].dropna()
            fig.add_box(y=v[col], x=[f"FY{y}"] * len(v), name=f"FY{y}", boxpoints="all", jitter=0.4, pointpos=0,
                        marker=dict(color=color, size=6, opacity=0.75), line=dict(color=T.NAVY, width=1.2),
                        fillcolor="rgba(232,238,248,0.6)", text=[NAME[t] for t in v["ticker"]],
                        hovertemplate="%{text}: %{y:.1f}" + unit + "<extra></extra>", showlegend=False)
            st_ = stars(ysig.loc[y, pcol]) if (pcol and y in ysig.index) else ""
            if st_:
                fig.add_annotation(x=f"FY{y}", y=1.0, yref="paper", text=st_, showarrow=False, yanchor="bottom",
                                   font=dict(size=14, color=T.MAGENTA))
        fig.update_layout(title=title, height=340, margin=dict(t=64, b=30),
                          xaxis=dict(type="category", categoryorder="category ascending"))
        return fig

    l, r = st.columns(2)
    l.plotly_chart(dist("tbvps_growth_pct", "TBV/share growth (%)", "%", "tbvps_growth_p", T.NAVY), width="stretch", config=CFG)
    r.plotly_chart(dist("aoci_to_tbv_pct", "Unrealized gains/losses (AOCI) as % of TBV", "%", "aoci_p", T.MAGENTA),
                   width="stretch", config=CFG)
    l, r = st.columns(2)
    l.plotly_chart(dist("rote_pct", "Return on tangible equity (%)", "%", "rote_p", T.BLUE), width="stretch", config=CFG)
    r.plotly_chart(dist("p_tbv", "Price / tangible book (x)", "x", None, T.SLATE), width="stretch", config=CFG)

    st.markdown("#### Loss ratio by carrier (losses and LAE ÷ net earned premium)")
    lr = fund.pivot(index="ticker", columns="fiscal_year", values="loss_ratio_pct").loc[:, y0:y1]
    lr = lr.dropna(how="all")
    lr = lr.loc[lr.mean(axis=1).sort_values().index]
    fig = go.Figure(go.Heatmap(z=lr.values, x=[str(c) for c in lr.columns], y=[f"{NAME[t]} ({SEG[t]})" for t in lr.index],
                               colorscale=[[0, T.BLUE], [0.5, "#F4F6FB"], [1, T.RED]], zmid=float(np.nanmedian(lr.values)),
                               text=np.where(np.isnan(lr.values), "", np.round(lr.values, 1).astype(str)),
                               texttemplate="%{text}", textfont=dict(size=11, color=T.INK),
                               colorbar=dict(title=dict(text="%", side="top"), thickness=10)))
    fig.update_layout(height=40 + 26 * len(lr), margin=dict(l=8, r=8, t=16, b=8), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config=CFG)
    T.note("Blue = below the universe median (better); red = above. Ratios outside 35–120% (partial XBRL tags) are blanked. "
           "Diversified groups' figures include non-P&C benefits where reported together.")

# ---------------------------------------------------------------------------
# Year by year
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("## Year by year: inflection points")
    st.markdown('<p class="lede">Each year is named only from shifts that pass a significance test across carriers. '
                "Hover the matrix for medians and p-values.</p>", unsafe_allow_html=True)
    ys = ctx["years"]
    yc = st.columns(len(ys))
    for col, (_, row) in zip(yc, ys.iterrows()):
        title, notes = A.year_theme(row)
        accent = T.RED if any(w in title.lower() for w in ["shock", "squeeze", "inflation"]) else (
            T.BLUE if "no statistically" not in title.lower() else T.SLATE)
        items = "".join(f"<li>{n}</li>" for n in notes) or "<li>No metric moved significantly.</li>"
        col.markdown(f'<div class="ycard" style="--accent:{accent}"><div class="y">FY{int(row["year"])}</div>'
                     f'<div class="h">{title}</div><ul>{items}</ul></div>', unsafe_allow_html=True)
    st.write("")
    metrics = [("tbvps_growth_median", "tbvps_growth_p", "TBV/share growth", "%"),
               ("aoci_to_tbv_change_pp", "aoci_p", "AOCI / TBV change", " pp"),
               ("rote_change_pp", "rote_p", "RoTE change", " pp"),
               ("loss_ratio_change_pp", "loss_ratio_p", "Loss ratio change", " pp"),
               ("premium_growth_median", "premium_growth_p", "Premium growth", "%"),
               ("excess_return_pct", "excess_return_p", "Excess return vs S&P", " pp")]
    good_if_up = {"Loss ratio change": False}
    z, txt, hov = [], [], []
    for mcol, pcol, lab, unit in metrics:
        zr, tr_, hr = [], [], []
        for _, row in ys.iterrows():
            v, p = row[mcol], row[pcol]
            sgn = np.sign(v) * (1 if good_if_up.get(lab, True) else -1) if pd.notna(v) else 0
            strength = 0 if pd.isna(p) else (3 if p < 0.01 else 2 if p < 0.05 else 1 if p < 0.10 else 0)
            zr.append(sgn * strength)
            tr_.append("" if pd.isna(v) else f"{v:+.1f}{unit}{stars(p)}")
            hr.append(f"{lab} {int(row['year'])}: {v:+.2f}{unit}<br>p = {p:.3g}" if pd.notna(v) and pd.notna(p) else "")
        z.append(zr); txt.append(tr_); hov.append(hr)
    xs = [f"FY{int(y)}" for y in ys["year"]]
    ylab = [m[2] for m in metrics]
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ylab, hovertext=hov, hoverinfo="text", zmin=-3, zmax=3, showscale=False,
                               xgap=2, ygap=2,
                               colorscale=[[0, T.RED], [0.33, "#F8B3C0"], [0.5, "#F4F6FB"], [0.67, "#B3C6FF"], [1, T.BLUE]]))
    for yi, (zr, tr_) in enumerate(zip(z, txt)):
        for xi, (zv, tv) in enumerate(zip(zr, tr_)):
            fig.add_annotation(x=xs[xi], y=ylab[yi], text=tv, showarrow=False,
                               font=dict(size=12, color=T.WHITE if abs(zv) >= 2 else T.INK))
    fig.update_layout(title="Signal matrix: color intensity = statistical strength, blue = favorable, red = adverse",
                      height=330, yaxis=dict(autorange="reversed"), xaxis=dict(type="category"), margin=dict(t=50, b=30))
    st.plotly_chart(fig, width="stretch", config=CFG)
    T.note("Medians across carriers; paired Wilcoxon tests for year-over-year changes, one-sample Wilcoxon for growth; "
           "weekly t-test for excess return. Stars: * p<0.10, ** p<0.05, *** p<0.01.")

# ---------------------------------------------------------------------------
# Data & methods
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("## Data & methods")
    datasets = {
        "Fundamentals by carrier-year (derived)": ctx["fund_all"][["ticker", "name", "segment", "fiscal_year", "premiums_earned",
                                                                  "net_income", "tbv", "tbvps", "p_tbv", "rote_pct", "loss_ratio_pct",
                                                                  "aoci_to_tbv_pct", "tbvps_growth_pct", "premium_growth_pct"]],
        "SEC XBRL extract (raw)": load_static()[0],
        f"TSR decomposition {y0}–{y1}": ctx["decomp_all"],
        "Year signals and tests": ctx["years"],
        "Shock days": ctx["shocks"].reset_index().rename(columns={"index": "date", "Date": "date"}),
        "Coverage universe": pd.DataFrame(UNIVERSE),
    }
    choice = st.selectbox("Dataset", list(datasets))
    df = datasets[choice].copy()
    if "segment" in df:
        segs = st.multiselect("Segment filter", sorted(df["segment"].dropna().unique()), default=sorted(df["segment"].dropna().unique()))
        df = df[df["segment"].isin(segs)]
    st.dataframe(df.round(3), hide_index=True, width="stretch", height=380)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=choice.split(" (")[0].lower().replace(" ", "_") + ".csv",
                       mime="text/csv")
    st.markdown("#### Methods")
    st.markdown(
        f"- **Universe:** {len(ALL_TICKERS)} US-listed P&C carriers with continuous trading since 2020, grouped into four "
        "segments by dominant business mix (project classification).\n"
        "- **Fundamentals:** SEC EDGAR XBRL company facts from annual 10-K filings (`scripts/build_fundamentals.py`). "
        "Tangible book = common equity − preferred − goodwill − intangibles; share counts from 10-K cover pages, put on "
        "today's split basis.\n"
        "- **Prices:** Yahoo Finance daily closes; total return uses dividend-adjusted closes. A committed snapshot is used "
        "if the live feed is unavailable (`scripts/build_prices.py`).\n"
        "- **Outliers:** robust z-score (median / 1.4826·MAD) of 5-year TSR; |z| > 2.5 flagged. Leave-one-out influence "
        "shows how much each carrier moves the average.\n"
        "- **Shocks:** basket return minus rolling 1-year alpha and beta to the S&P 500; days beyond the chosen σ threshold of "
        "the prior 63-day residual volatility. Catalysts matched after detection from a fixed dated event list.\n"
        "- **Regimes:** binary segmentation on weekly excess returns with a standardized CUSUM statistic; a break is kept only "
        "if it beats 999 random permutations at p < 0.05.\n"
        "- **Insight cards:** generated from tests on this data; only findings with p < 0.10 are eligible. "
        "Shuffle draws cards weighted by statistical strength, one per theme.\n"
        "- **Limits:** equal-weight baskets; daily-return tests assume independence; XBRL tag coverage varies by filer "
        "(gaps are left blank, not estimated). Not investment advice.")
