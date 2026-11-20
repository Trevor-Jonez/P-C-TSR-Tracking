from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import theme as T

st.set_page_config(page_title="US P&C Value Creators Tracker", layout="wide", page_icon="📈")
T.register_plotly_template()
T.inject_css()

DATA = Path(__file__).parent / "data"
REPORT_URL = "https://www.bcg.com/publications/2026/soft-rates-and-big-risks-reshape-the-us-p-and-c-market"
REPORT_CITE = (
    f'Source: BCG, <a href="{REPORT_URL}" target="_blank">"Softer Rates and Bigger Risks Are Reshaping the US P&amp;C '
    "Market\"</a> (2026 Insurance Value Creators Report, June 25, 2026); S&amp;P Capital IQ; BCG ValueScience Center."
)
CHART_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATASETS = {
    "Exhibit 1 · TSR decomposition, US P&C vs global": "exhibit1_tsr_decomposition.csv",
    "Exhibit 2 · RoTE contribution, top vs bottom quartile": "exhibit2_rote_contribution.csv",
    "Exhibit 3 · Company TSR results (1st & 2nd quartile)": "exhibit3_company_tsr.csv",
    "Exhibit 4 · Subsegment TSR decomposition": "exhibit4_subsegment_tsr.csv",
    "Exhibit 5 · TSR spread by subsector": "exhibit5_tsr_spread.csv",
    "Market indicators cited in the report": "market_indicators.csv",
}


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


ex1 = load(DATASETS["Exhibit 1 · TSR decomposition, US P&C vs global"])
ex2 = load(DATASETS["Exhibit 2 · RoTE contribution, top vs bottom quartile"])
ex3 = load(DATASETS["Exhibit 3 · Company TSR results (1st & 2nd quartile)"])
ex4 = load(DATASETS["Exhibit 4 · Subsegment TSR decomposition"])
ex5 = load(DATASETS["Exhibit 5 · TSR spread by subsector"])
ind = load(DATASETS["Market indicators cited in the report"])

BENCHMARKS = {
    "S&P 500": "^GSPC",
    "S&P 500 Insurance (industry group)": "^SP500-4030",
    "SPDR S&P Insurance ETF (KIE)": "KIE",
}
COMPANIES = dict(zip(ex3["company"], ex3["ticker"]))
TICKERS = {**COMPANIES, **BENCHMARKS}
TICKER_TO_NAME = {v: k for k, v in TICKERS.items()}
START = pd.Timestamp("2020-12-31")
BCG_END = pd.Timestamp("2025-12-31")
YTD_START = pd.Timestamp("2025-12-31")


@st.cache_data(ttl=3600, show_spinner="Fetching live prices from Yahoo Finance…")
def fetch_prices() -> pd.DataFrame:
    """Daily dividend-adjusted closes, columns named by entity."""
    raw = yf.download(list(TICKERS.values()), start=f"{START:%Y-%m-%d}", auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
    # yfinance orders columns alphabetically by ticker: always map by ticker, never by position
    return closes.rename(columns=TICKER_TO_NAME).sort_index()


def _px_on(s: pd.Series, when: pd.Timestamp):
    s = s.dropna()
    s = s[s.index <= when]
    return (s.index[-1], s.iloc[-1]) if len(s) else (None, None)


def ann_return(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp | None = None) -> float:
    s = s.dropna()
    if end is not None:
        s = s[s.index <= end]
    d0, p0 = _px_on(s, start)
    if d0 is None or s.empty:
        return float("nan")
    d1, p1 = s.index[-1], s.iloc[-1]
    yrs = (d1 - d0).days / 365.25
    return ((p1 / p0) ** (1 / yrs) - 1) * 100 if yrs > 0.05 else float("nan")


def cum_return(s: pd.Series, start: pd.Timestamp) -> float:
    s = s.dropna()
    d0, p0 = _px_on(s, start)
    if d0 is None or s.empty or s.index[-1] <= d0:
        return float("nan")
    return (s.iloc[-1] / p0 - 1) * 100


prices, live_error = None, None
try:
    prices = fetch_prices()
    empty = [c for c in prices.columns if prices[c].dropna().empty]
    prices = prices.drop(columns=empty)
except Exception as exc:  # rate limits / outages must not take the storyline down
    live_error = type(exc).__name__

live = pd.DataFrame()
if prices is not None and not prices.empty:
    last_date = prices.dropna(how="all").index[-1]
    rows = []
    for name in prices.columns:
        s = prices[name]
        rows.append({
            "entity": name,
            "ticker": TICKERS.get(name, ""),
            "replicated_tsr_2021_2025_pct": ann_return(s, START, BCG_END),
            "trailing_5yr_tsr_pct": ann_return(s, last_date - pd.DateOffset(years=5)),
            "since_ye2020_tsr_pct": ann_return(s, START),
            "ytd_2026_cumulative_pct": cum_return(s, YTD_START),
        })
    live = pd.DataFrame(rows)
    live_note = f"Live prices through {last_date:%b %d, %Y} (Yahoo Finance, dividend-adjusted, refreshed hourly)."
else:
    last_date = None
    live_note = "Live prices unavailable right now; BCG report data is shown in full."


def bcg_vs_live() -> pd.DataFrame:
    df = ex3.merge(live, left_on="company", right_on="entity", how="left") if not live.empty else ex3.copy()
    return df


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="eyebrow">US P&amp;C Insurance · 2026 Value Creators · Live tracker</div>
  <h1>Softer rates, bigger risks:<br/>tracking the US P&amp;C value story live</h1>
  <p>BCG's 2026 Insurance Value Creators findings, rebuilt as interactive data and benchmarked
  against live market returns for the same carriers.</p>
</div>
""",
    unsafe_allow_html=True,
)

c = st.columns(4)
c[0].markdown(T.kpi("18%", "US P&C five-year TSR", "2021–2025 avg annual · vs 15% global", "navy"), unsafe_allow_html=True)
c[1].markdown(T.kpi("+11 pp", "Top-quartile underwriting RoTE", "vs −2 pp bottom quartile"), unsafe_allow_html=True)
c[2].markdown(T.kpi("$50B+", "Insured SCS losses, each year", "2023, 2024, 2025 · vs ~$30B prior decade", "mag"), unsafe_allow_html=True)
c[3].markdown(T.kpi("$62B", "Adverse development, commercial liability", "2015–2024 · Swiss Re", "alert"), unsafe_allow_html=True)

st.write("")
themes = [
    ("Pricing momentum has moderated", "Rate alone no longer fixes results; data, segmentation and expense discipline must."),
    ("Underwriting excellence is non-negotiable", "Top-quartile carriers earn higher RoTE through superior loss ratios."),
    ("Structural risks are deepening", "Long-tail casualty, secondary perils and litigation need proactive portfolio moves."),
    ("AI is moving from experiment to advantage", "Leaders deploy AI at scale across underwriting, claims and pricing."),
]
tc = st.columns(4)
for col, (h, b) in zip(tc, themes):
    col.markdown(f'<div class="theme-card"><b>{h}</b><span>{b}</span></div>', unsafe_allow_html=True)
st.write("")
st.caption(live_note if live_error is None else f"⚠️ {live_note} ({live_error})")

tabs = st.tabs([
    "01 · TSR performance", "02 · Underwriting divide", "03 · Subsegments",
    "04 · Personal lines", "05 · Structural risks", "Live tracker", "Data explorer",
])

# ---------------------------------------------------------------------------
# 01 TSR
# ---------------------------------------------------------------------------
with tabs[0]:
    T.chapter("01", "TSR performance: strong but moderating",
              "US P&C insurers delivered roughly 18% average annual TSR in 2021–2025, ahead of the 15% global "
              "insurance average. Growth in tangible book value, powered by underwriting income, was the primary engine; "
              "one-year 2025 data already shows US insurers decelerating versus European and Asia-Pacific peers.")
    l, r = st.columns([1.1, 1])
    with l:
        comp = ex1[ex1["component"] != "2021–2025 five-year TSR"]
        tot = ex1[ex1["component"] == "2021–2025 five-year TSR"].set_index("sample")["pp"]
        fig = go.Figure()
        for comp_name, color in T.COMPONENT_COLORS.items():
            d = comp[comp["component"] == comp_name]
            fig.add_bar(y=d["sample"] + " (n=" + d["n"].astype(str) + ")", x=d["pp"], name=comp_name, orientation="h",
                        marker_color=color, text=d["pp"], textposition="inside", insidetextfont=dict(color="white"))
        for i, (s, v) in enumerate(tot.items()):
            n = ex1.loc[ex1["sample"] == s, "n"].iloc[0]
            fig.add_annotation(x=v, y=f"{s} (n={n})", text=f"<b>{v}%</b>", showarrow=False, xanchor="left", xshift=8,
                               font=dict(size=16, color=T.NAVY))
        fig.update_layout(barmode="stack", title="Exhibit 1 · Contribution to average annual TSR, 2021–2025 (pp)",
                          height=320, xaxis=dict(range=[0, 21], title=None), yaxis=dict(autorange="reversed"),
                          legend=dict(traceorder="normal"))
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
        T.source("Exhibit 1. US P&amp;C leads global insurance by +3 pp on TBV growth and +1 pp on multiple; "
                 "cash flow trails by 1 pp. " + REPORT_CITE)
    with r:
        if not live.empty:
            sel = live.dropna(subset=["trailing_5yr_tsr_pct"]).sort_values("trailing_5yr_tsr_pct")
            colors = [T.SLATE if e in BENCHMARKS else (T.BLUE if v >= 18 else T.NAVY)
                      for e, v in zip(sel["entity"], sel["trailing_5yr_tsr_pct"])]
            fig = go.Figure(go.Bar(x=sel["trailing_5yr_tsr_pct"], y=sel["entity"], orientation="h", marker_color=colors,
                                   text=sel["trailing_5yr_tsr_pct"].map(lambda v: f"{v:.1f}%"), textposition="outside"))
            fig.add_vline(x=18, line_dash="dash", line_color=T.MAGENTA)
            fig.add_annotation(x=18, y=1.0, yref="paper", text="BCG 2021–25: 18%", showarrow=False, yanchor="bottom",
                               font=dict(color=T.MAGENTA, size=12), xanchor="left")
            fig.update_traces(cliponaxis=False)
            fig.update_layout(title="Live · trailing 5-year annualized TSR (%)", height=540, showlegend=False,
                              xaxis=dict(title=None, range=[min(0, sel["trailing_5yr_tsr_pct"].min() * 1.2),
                                                            max(20, sel["trailing_5yr_tsr_pct"].max()) * 1.15]),
                              margin=dict(r=40, t=80))
            st.plotly_chart(fig, width="stretch", config=CHART_CFG)
            T.source(f"Blue = above BCG's 18% benchmark; grey = market benchmarks. Window: "
                     f"{last_date - pd.DateOffset(years=5):%b %d, %Y} – {last_date:%b %d, %Y}.")
        else:
            st.info("Live TSR comparison will appear when market data is available.")

    if prices is not None and not prices.empty:
        st.markdown("#### Live total return index against the storyline")
        default = [n for n in ["Progressive", "Hartford", "Chubb", "Allstate", "S&P 500"] if n in prices.columns]
        pick = st.multiselect("Companies and benchmarks", list(prices.columns), default=default, key="rebased")
        if pick:
            reb = prices[pick].ffill()
            reb = reb.div(reb.loc[reb.index >= START].bfill().iloc[0]).mul(100)
            fig = go.Figure()
            for i, n in enumerate(pick):
                fig.add_scatter(x=reb.index, y=reb[n], name=n, mode="lines",
                                line=dict(width=2.4 if i == 0 else 1.6, color=T.SERIES[i % len(T.SERIES)]))
            events = [
                ("2022-01-01", "2023-12-31", "Personal lines profitability squeeze", T.RED),
                ("2026-01-01", "2026-05-31", "BCG YTD window", T.BLUE),
            ]
            for x0, x1, label, colr in events:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=colr, opacity=0.07, line_width=0)
                fig.add_annotation(x=x0, y=1.0, yref="paper", text=label, showarrow=False, xanchor="left", yanchor="top",
                                   font=dict(size=11, color=colr))
            for x, label in [("2025-12-31", "BCG 5-yr window closes"), ("2026-06-25", "Report published")]:
                fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color=T.NAVY, dash="dot", width=1))
                fig.add_annotation(x=x, y=0.02, yref="paper", text=label, showarrow=False, textangle=-90, xanchor="right",
                                   yanchor="bottom", font=dict(size=11, color=T.NAVY))
            fig.update_layout(height=460, yaxis_title="Total return index (Dec 31, 2020 = 100)", hovermode="x unified",
                              title="Dividend-adjusted total return, rebased")
            st.plotly_chart(fig, width="stretch", config=CHART_CFG)

# ---------------------------------------------------------------------------
# 02 Underwriting
# ---------------------------------------------------------------------------
with tabs[1]:
    T.chapter("02", "Underwriting determines the profitability divide",
              "Top-quartile carriers generated about 11 pp of RoTE from underwriting in 2021–2025; the bottom quartile "
              "lost money underwriting. Elevated investment income in 2024–2025 masked the gap, and it will narrow as "
              "rates normalize, leaving underwriting quality as the differentiator.")
    cols = st.columns(2)
    for col, q in zip(cols, ["Top quartile", "Bottom quartile"]):
        d = ex2[ex2["quartile"] == q]
        measure = ["relative", "relative", "relative", "total"]
        fig = go.Figure(go.Waterfall(
            x=d["component"], y=d["pp"], measure=measure, text=d["pp"].map(lambda v: f"{v:+d}" if v else "0"),
            textposition="outside", connector=dict(line=dict(color=T.MID)),
            increasing=dict(marker=dict(color=T.BLUE)), decreasing=dict(marker=dict(color=T.RED)),
            totals=dict(marker=dict(color=T.NAVY)),
        ))
        fig.update_layout(title=f"{q} (n={d['n'].iloc[0]}) · RoTE {d['pp'].iloc[-1]}%", height=360,
                          yaxis=dict(range=[-4, 26], title="pp"), showlegend=False)
        col.plotly_chart(fig, width="stretch", config=CHART_CFG)
    T.source("Exhibit 2 · Contribution to return on tangible equity, 2021–2025 (pp). RoTE = pretax operating income gross "
             "of interest expense ÷ year-end tangible book value net of preferred stock. Quartiles based on TSR. " + REPORT_CITE)

    st.markdown("#### Exhibit 3 · Who created the value")
    l, r = st.columns([1, 1.15])
    with l:
        fig = px.scatter(ex3, x="rote_5yr_avg_pct", y="tsr_2021_2025_pct", color="quartile", text="company",
                         color_discrete_map={"First": T.BLUE, "Second": T.NAVY},
                         labels={"rote_5yr_avg_pct": "Five-year average RoTE (%)", "tsr_2021_2025_pct": "TSR 2021–2025 (%)",
                                 "quartile": "TSR quartile", "subsector": "Subsector"},
                         hover_data={"subsector": True, "ytd_tsr_2026_annualized_pct": True, "rank_change": True})
        pos = {"American Financial": "bottom center", "Progressive": "top center", "AIG": "middle right",
               "AXIS Capital": "middle right", "Mercury": "bottom center", "Cincinnati": "bottom center",
               "Allstate": "bottom center", "Chubb": "middle right", "Travelers": "middle right"}
        for tr in fig.data:
            tr.textposition = [pos.get(t, "top center") for t in tr.text]
        fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")), textfont=dict(size=11, color=T.SLATE))
        fig.update_layout(title="Return on tangible equity vs TSR", height=470, legend_title_text="TSR quartile",
                          xaxis=dict(range=[6, 32]), yaxis=dict(range=[14.5, 27.5]))
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
    with r:
        parts = {"tbv_growth_pp": ("Growth of TBV", T.NAVY), "pptbv_multiple_change_pp": ("Change in P/TBV multiple", T.BLUE),
                 "dividend_yield_pp": ("Dividend yield", T.MAGENTA), "share_count_change_pp": ("Change in share count", T.SLATE)}
        d = ex3.sort_values("tsr_2021_2025_pct")
        fig = go.Figure()
        for k, (label, colr) in parts.items():
            fig.add_bar(y=d["company"], x=d[k], name=label, orientation="h", marker_color=colr)
        fig.add_scatter(y=d["company"], x=d["tsr_2021_2025_pct"], mode="markers+text", name="Five-year TSR",
                        marker=dict(symbol="line-ns-open", size=16, color=T.INK, line=dict(width=2)),
                        text=d["tsr_2021_2025_pct"].astype(str) + "%", textposition="middle right",
                        textfont=dict(color=T.NAVY, size=11))
        fig.update_layout(barmode="relative", title="TSR building blocks by company, 2021–2025 (pp)", height=500,
                          xaxis=dict(range=[-6, 34]), legend=dict(y=-0.08, yanchor="top"), margin=dict(b=60))
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
    T.source("Exhibit 3 · TSR results for the first and second quartiles of US P&amp;C insurers. Components may not sum to "
             "totals because of rounding. Sources: S&amp;P Capital IQ; Refinitiv; BCG ValueScience Center.")

    st.markdown("#### BCG findings vs live market check")
    df = bcg_vs_live()
    if "replicated_tsr_2021_2025_pct" in df:
        l, r = st.columns(2)
        with l:
            d = df.sort_values("tsr_2021_2025_pct")
            fig = go.Figure()
            for _, row in d.iterrows():
                fig.add_shape(type="line", x0=row["tsr_2021_2025_pct"], x1=row["replicated_tsr_2021_2025_pct"],
                              y0=row["company"], y1=row["company"], line=dict(color=T.MID, width=3))
            fig.add_scatter(x=d["tsr_2021_2025_pct"], y=d["company"], mode="markers", name="BCG reported",
                            marker=dict(size=13, color=T.NAVY))
            fig.add_scatter(x=d["replicated_tsr_2021_2025_pct"], y=d["company"], mode="markers", name="Live replication",
                            marker=dict(size=13, color=T.BLUE, symbol="diamond"))
            fig.update_layout(title="2021–2025 TSR: BCG vs replicated from market prices (%)", height=460)
            st.plotly_chart(fig, width="stretch", config=CHART_CFG)
            T.source("Replication = dividend-adjusted price, Dec 31, 2020 → Dec 31, 2025, annualized. Gaps reflect "
                     "data-vendor and dividend-reinvestment methodology differences.")
        with r:
            d = df.sort_values("ytd_2026_cumulative_pct")
            fig = go.Figure()
            fig.add_bar(y=d["company"], x=d["ytd_2026_cumulative_pct"], orientation="h", name="Live YTD 2026 (cumulative)",
                        marker_color=[T.POS if v >= 0 else T.NEG for v in d["ytd_2026_cumulative_pct"]])
            fig.add_scatter(y=d["company"], x=d["ytd_tsr_2026_annualized_pct"], mode="markers", name="BCG Jan–May 2026 (annualized)",
                            marker=dict(size=12, color=T.MAGENTA, symbol="diamond", line=dict(width=1, color="white")))
            fig.add_vline(x=0, line_color=T.SLATE, line_width=1)
            fig.update_layout(title="2026 so far: has the moderation continued? (%)", height=460)
            st.plotly_chart(fig, width="stretch", config=CHART_CFG)
            T.source(f"Live: Dec 31, 2025 → {last_date:%b %d, %Y}, cumulative. BCG: Jan 1 – May 31, 2026, annualized "
                     "(different basis; compare direction, not magnitude).")
    else:
        st.info("Live comparison will appear when market data is available.")

# ---------------------------------------------------------------------------
# 03 Subsegments
# ---------------------------------------------------------------------------
with tabs[2]:
    T.chapter("03", "Diverging subsegment fortunes",
              "Personal and commercial lines posted similar 2021–2025 TSRs (17% and 18%), but the paths diverged: "
              "commercial rode a prolonged hard market while personal lines struggled in 2022–2023 before a sharp "
              "2024 recovery. Progressive alone lifted personal lines from 13% to 17%.")
    l, r = st.columns([1.1, 1])
    with l:
        segs = ["Personal lines", "Personal and commercial lines", "Commercial lines"]
        ns = ex4.drop_duplicates("subsegment").set_index("subsegment")["n"]
        xl = [f"{s}<br>(n={ns[s]})" for s in segs]
        val = lambda comp: ex4[ex4["component"] == comp].set_index("subsegment").loc[segs, "pp"]
        fig = go.Figure()
        for comp_name, colr in T.COMPONENT_COLORS.items():
            fig.add_bar(x=xl, y=val(comp_name), name=comp_name, marker_color=colr, text=val(comp_name),
                        textposition="inside", insidetextfont=dict(color="white"), offsetgroup="now", legendgroup="now")
        prior, now = val("2020–2024 five-year TSR"), val("2021–2025 five-year TSR")
        fig.add_bar(x=xl, y=prior, name="2020–2024 five-year TSR", marker_color="#B9B6D6", offsetgroup="prior",
                    text=prior.astype(str) + "%", textposition="outside", textfont=dict(color=T.SLATE))
        fig.add_scatter(x=xl, y=now + 0.2, mode="text", text=["<b>" + str(v) + "%</b>" for v in now],
                        textposition="top center", textfont=dict(size=14, color=T.NAVY), showlegend=False, hoverinfo="skip")
        for x, d in zip(xl, now - prior):
            fig.add_annotation(x=x, y=24.5, text=f"<b>{d:+d} pp</b> vs 2020–24", showarrow=False,
                               font=dict(color=T.BLUE if d > 0 else T.RED, size=12))
        fig.update_layout(barmode="stack", title="Exhibit 4 · TSR building blocks by subsegment, 2021–2025 (pp)",
                          height=500, yaxis=dict(range=[0, 26]), bargroupgap=0.08,
                          legend=dict(y=-0.14, yanchor="top"), margin=dict(b=70))
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
        T.source("Shift vs 2020–2024: personal −4 pp, personal &amp; commercial +4 pp, commercial +5 pp. "
                 "Component order follows Exhibit 1 (TBV, multiple, cash flow). " + REPORT_CITE)
    with r:
        d = ex5.copy()
        fig = go.Figure()
        for _, row in d.iterrows():
            if pd.notna(row["top_quartile_pct"]):
                fig.add_shape(type="line", x0=row["group"], x1=row["group"], y0=row["bottom_quartile_pct"],
                              y1=row["top_quartile_pct"], line=dict(color=T.MID, width=6))
        fig.add_scatter(x=d["group"], y=d["top_quartile_pct"], mode="markers+text", name="Top quartile",
                        text=d["top_quartile_pct"], textposition="middle right", marker=dict(symbol="square", size=12, color=T.BLUE))
        fig.add_scatter(x=d["group"], y=d["sector_average_pct"], mode="markers+text", name="Sector average",
                        text=d["sector_average_pct"], textposition="middle left", marker=dict(symbol="diamond", size=13, color=T.NAVY))
        fig.add_scatter(x=d["group"], y=d["bottom_quartile_pct"], mode="markers+text", name="Bottom quartile",
                        text=d["bottom_quartile_pct"], textposition="middle right", marker=dict(symbol="square", size=12, color=T.RED))
        fig.update_xaxes(ticktext=["Global<br>insurance", "US P&C", "Personal<br>lines", "Personal &<br>commercial",
                                   "Mostly<br>specialty", "Mostly<br>P&C"], tickvals=list(d["group"]))
        fig.update_layout(title="Exhibit 5 · Average annual TSR spread, 2021–2025 (%)", height=480, yaxis=dict(range=[-14, 30]))
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
        T.source("Personal lines show the widest spread (21% to −10%), driven by one outlier; specialty narrowed its spread "
                 "vs last year. " + REPORT_CITE)

    if "replicated_tsr_2021_2025_pct" in bcg_vs_live():
        st.markdown("#### Live check · subsector averages from tracked carriers")
        df = bcg_vs_live().dropna(subset=["trailing_5yr_tsr_pct"])
        g = df.groupby("subsector")[["tsr_2021_2025_pct", "trailing_5yr_tsr_pct", "ytd_2026_cumulative_pct"]].mean().round(1)
        g.columns = ["BCG 2021–25 TSR (simple avg, %)", "Live trailing 5-yr TSR (%)", "Live YTD 2026 (%)"]
        st.dataframe(g, width="stretch")
        st.caption("Simple averages of the 12 Exhibit 3 carriers only (not BCG's market-cap-weighted 24-company sample).")

# ---------------------------------------------------------------------------
# 04 Personal lines
# ---------------------------------------------------------------------------
with tabs[3]:
    T.chapter("04", "Near-term trends in personal lines",
              "Auto has moved from defense to offense, but not for everyone. Homeowners posted its best result since "
              "2006 in 2025, helped by a hurricane-free year that BCG warns is not a structural reset.")
    k = st.columns(4)
    k[0].markdown(T.kpi("112% → 92%", "Personal auto combined ratio", "2022 → 2025 · best since 2020"), unsafe_allow_html=True)
    k[1].markdown(T.kpi("~88%", "Progressive personal auto CR", "2025 · now #1 US personal auto writer", "navy"), unsafe_allow_html=True)
    k[2].markdown(T.kpi("88%", "Homeowners combined ratio", "2025 · best since 2006", "mag"), unsafe_allow_html=True)
    k[3].markdown(T.kpi("6–8%", "Annual auto loss-cost trend", "outpacing rate in some states", "alert"), unsafe_allow_html=True)
    st.write("")
    l, r = st.columns(2)
    with l:
        d = pd.DataFrame({"label": ["Personal auto 2022", "Personal auto 2025", "Progressive auto 2025", "Homeowners 2025"],
                          "cr": [112, 92, 88, 88], "c": [T.RED, T.BLUE, T.NAVY, T.MAGENTA]})
        fig = go.Figure(go.Bar(x=d["label"], y=d["cr"], marker_color=d["c"], text=d["cr"].astype(str) + "%", textposition="outside"))
        fig.add_hline(y=100, line_color=T.INK, line_width=1)
        fig.add_annotation(x=0.99, xref="paper", y=100, text="Underwriting break-even", showarrow=False,
                           yanchor="bottom", xanchor="right", font=dict(size=11, color=T.INK))
        fig.add_hline(y=95, line_dash="dot", line_color=T.SLATE)
        fig.add_annotation(x=0.99, xref="paper", y=95, text="95%: pre-2020 auto never beat this after 2004", showarrow=False,
                           yanchor="top", xanchor="right", font=dict(size=11, color=T.SLATE))
        fig.update_layout(title="Combined ratios cited in the report (%)", height=420, yaxis=dict(range=[70, 120]), showlegend=False)
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
        T.source("Sources: S&amp;P Global Market Intelligence; NAIC; BCG. Lower is better.")
    with r:
        d = pd.DataFrame({"period": ["Prior-decade avg", "2023", "2024", "2025"], "v": [30, 50, 50, 50]})
        fig = go.Figure(go.Bar(x=d["period"], y=d["v"], marker_color=[T.SLATE, T.MAGENTA, T.MAGENTA, T.MAGENTA],
                               text=["~$30B", ">$50B", ">$50B", ">$50B"], textposition="outside"))
        fig.update_layout(title="Insured severe convective storm losses ($B)", height=420, yaxis=dict(range=[0, 62]), showlegend=False)
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
        T.source("Sources: Verisk; Munich Re (as cited by BCG). 2023–2025 bars show the $50B floor the report states; "
                 "actual totals were higher. SCS now exceeds hurricanes as the primary US cat driver.")
    l, r = st.columns(2)
    with l:
        st.markdown("##### Auto: the risks in going on offense")
        st.markdown("- Claims **frequency** returning from pandemic lows; **severity** still rising\n"
                    "- Loss costs of **6–8% a year** outpacing rate increases in some states\n"
                    "- **Large mutuals** may absorb margin pressure to buy share")
    with r:
        st.markdown("##### Homeowners: strategic considerations")
        st.markdown("- **Policy form management:** terms, exclusions and sublimits matter more\n"
                    "- **Scale as a moat:** top five personal lines writers keep extending share\n"
                    "- **Reinsurance optimization:** 2025 renewals down ~15–20%, but terms still tighter than pre-2022")

# ---------------------------------------------------------------------------
# 05 Structural
# ---------------------------------------------------------------------------
with tabs[4]:
    T.chapter("05", "Longer-term structural trends",
              "Long-tail casualty is the industry's most significant unresolved risk. Attorney involvement, third-party "
              "litigation funding and nuclear verdicts are recasting loss patterns beyond what historical models imply.")
    k = st.columns(4)
    k[0].markdown(T.kpi("135", "Nuclear verdicts ($10M+) in 2024", "most since tracking began in 2009", "alert"), unsafe_allow_html=True)
    k[1].markdown(T.kpi("$15–18B", "Third-party litigation finance market", "Swiss Re Institute estimate"), unsafe_allow_html=True)
    k[2].markdown(T.kpi("$62B", "Adverse development, commercial liability", "2015–2024 · excl. med pro", "navy"), unsafe_allow_html=True)
    k[3].markdown(T.kpi("19 yrs", "Favorable aggregate reserve development", "broader US P&C · consecutive", "mag"), unsafe_allow_html=True)
    st.write("")
    l, r = st.columns([1, 1])
    with l:
        fig = go.Figure()
        fig.add_bar(x=["Commercial liability<br>2015–2024"], y=[62], name="Adverse development ($B)", marker_color=T.RED,
                    text=["$62B adverse"], textposition="outside")
        fig.add_bar(x=["Third-party litigation<br>finance market"], y=[18], base=[0], marker_color=T.MID, name="Range high",
                    text=["$15–18B"], textposition="outside", showlegend=False)
        fig.add_bar(x=["Third-party litigation<br>finance market"], y=[15], marker_color=T.NAVY, name="Range low", showlegend=False)
        fig.update_layout(title="Social inflation in dollars ($B)", height=400, barmode="overlay", showlegend=False,
                          yaxis=dict(range=[0, 72]))
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
        T.source("Sources: Swiss Re; Swiss Re Institute; Marathon Strategies (as cited by BCG).")
    with r:
        st.markdown("##### Implications for carriers")
        st.markdown("- **Reserve uncertainty:** carriers with prior-year commercial liability and auto exposure face potential "
                    "adverse development; analysts and rating agencies are scrutinizing adequacy harder.\n"
                    "- **Underwriting posture:** leaders are shifting capacity toward shorter-tail exposures and excess & "
                    "surplus lines, which offer more flexible terms and pricing.")
        st.info("The downloadable report this dashboard was built from ends here. Later sections of the web article "
                "(including the AI theme) are not yet captured in the dataset.")

# ---------------------------------------------------------------------------
# Live tracker
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown('<div class="chapter">Live</div>', unsafe_allow_html=True)
    st.markdown("## Live TSR tracker")
    if live.empty:
        st.error("Could not load live market data from Yahoo Finance right now. This is usually a temporary rate limit; "
                 "please refresh in a few minutes.")
    else:
        metric = st.radio("Window", {"trailing_5yr_tsr_pct": "Trailing 5 years (annualized)",
                                     "since_ye2020_tsr_pct": "Since year-end 2020 (annualized)",
                                     "replicated_tsr_2021_2025_pct": "BCG window 2021–2025 (annualized)",
                                     "ytd_2026_cumulative_pct": "2026 year to date (cumulative)"}.items(),
                          format_func=lambda kv: kv[1], horizontal=True)
        col = metric[0]
        d = live.dropna(subset=[col]).sort_values(col)
        fig = go.Figure(go.Bar(x=d[col], y=d["entity"], orientation="h",
                               marker_color=[T.SLATE if e in BENCHMARKS else (T.POS if v >= 0 else T.NEG) for e, v in zip(d["entity"], d[col])],
                               text=d[col].map(lambda v: f"{v:.1f}%"), textposition="outside"))
        if col != "ytd_2026_cumulative_pct":
            fig.add_vline(x=18, line_dash="dash", line_color=T.MAGENTA)
            fig.add_annotation(x=18, y=1.0, yref="paper", text="BCG US P&C 2021–25: 18%", showarrow=False,
                               yanchor="bottom", font=dict(color=T.MAGENTA), xanchor="left")
        lo, hi = d[col].min(), d[col].max()
        fig.update_traces(cliponaxis=False)
        fig.update_layout(title=metric[1] + " (%)", height=580, showlegend=False, margin=dict(r=50, t=80),
                          xaxis=dict(range=[min(0, lo) * 1.25 - (2 if lo < 0 else 0), max(20, hi) * 1.15]))
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)
        st.caption(live_note + " Benchmarks in grey. ^SP500-4030 is a price index (excludes dividends) and spans all "
                   "insurance, not only P&C.")

    st.markdown("#### Upcoming data releases to watch")
    drops = pd.DataFrame([
        {"Dataset": "CIAB Q3/Q4 Commercial Rate Survey", "Expected window": "Dec 2026 / Feb 2027", "What it tests": "Pace of commercial rate softening"},
        {"Dataset": "FY 2026 rolling TSR window close", "Expected window": "Dec 31, 2026", "What it tests": "Shifts 5-yr window to 2022–2026 (drops 2021)"},
        {"Dataset": "NAIC / III FY26 statutory combined ratios", "Expected window": "March 2027", "What it tests": "Whether 92% auto / 88% homeowners CRs hold"},
        {"Dataset": "BCG 2027 Insurance Value Creators", "Expected window": "Spring–summer 2027", "What it tests": "Updated TBV, cash flow and multiple splits"},
    ])
    st.dataframe(drops, hide_index=True, width="stretch")
    st.caption("Manually maintained estimates, not live data.")

# ---------------------------------------------------------------------------
# Data explorer
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown('<div class="chapter">Data</div>', unsafe_allow_html=True)
    st.markdown("## Explore the full dataset")
    st.markdown('<p class="lede">Every figure in this dashboard, transcribed from the BCG report exhibits and text, '
                "plus live market metrics. Filter, sort, chart and download.</p>", unsafe_allow_html=True)
    options = list(DATASETS) + (["Live market metrics (Yahoo Finance)"] if not live.empty else [])
    choice = st.selectbox("Dataset", options)
    df = live.round(2) if choice.startswith("Live") else load(DATASETS[choice])

    cat_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique() <= 30]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        fcols = st.columns(min(len(cat_cols), 4))
        for fc, cname in zip(fcols, cat_cols[:4]):
            vals = sorted(df[cname].dropna().unique())
            chosen = fc.multiselect(cname, vals, default=vals, key=f"f_{choice}_{cname}")
            df = df[df[cname].isin(chosen)]
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=f"{choice.split(' ·')[0].replace(' ', '_').lower()}.csv",
                       mime="text/csv")

    if num_cols and len(df):
        st.markdown("#### Quick chart")
        a, b, cc = st.columns(3)
        label_candidates = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
        x = a.selectbox("Category", label_candidates)
        y = b.selectbox("Value", num_cols)
        color = cc.selectbox("Color by", ["(none)"] + [c for c in label_candidates if c != x])
        fig = px.bar(df, x=x, y=y, color=None if color == "(none)" else color, barmode="group",
                     text=df[y].map(lambda v: f"{v:g}" if pd.notna(v) else ""))
        fig.update_layout(height=420, title=f"{y} by {x}")
        st.plotly_chart(fig, width="stretch", config=CHART_CFG)

    st.markdown("#### Methodology")
    st.markdown(
        "- **BCG data** is transcribed by hand from the report's exhibits and text; values are as printed (rounded to whole "
        "points) and components may not sum to totals.\n"
        "- **Exhibit 4** prints a color legend that conflicts with Exhibit 1's ordering; this dashboard follows Exhibit 1's "
        "order (TBV, multiple, cash flow), which matches the commercial-lines and total values.\n"
        "- **Live TSR** uses Yahoo Finance dividend-adjusted closes. BCG uses S&P Capital IQ fundamentals and market-cap "
        "weighting, so replicated figures will differ.\n"
        f"- Report: [{REPORT_URL}]({REPORT_URL})"
    )

st.markdown('<div class="source" style="margin-top:2rem">Independent analysis dashboard. BCG content © 2026 Boston Consulting '
            "Group, cited for analysis and commentary. Not investment advice.</div>", unsafe_allow_html=True)
