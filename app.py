from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy import stats as _st

import analysis as A
import cyber as CY
import db
import industry as IND
import insights as I
import story as SY
import subind as SB
import theme as T
from universe import BENCHMARKS, CARRIERS, INTERNATIONAL, MUTUALS, UNIVERSE

st.set_page_config(page_title="US P&C Value Signals", layout="wide", page_icon="📈")
T.register_plotly_template()
T.inject_css()

DATA = Path(__file__).parent / "data"
CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
ALL_TICKERS = [c["ticker"] for c in CARRIERS]
NAME = {c["ticker"]: c["name"] for c in UNIVERSE}
SEG = {c["ticker"]: c["segment"] for c in CARRIERS}
PX_TICKERS = [c["ticker"] for c in UNIVERSE] + list(INTERNATIONAL.values()) + list(BENCHMARKS.values())
SUB_COLORS = {"Carrier": T.NAVY, "Reinsurer": T.MAGENTA, "Retail broker": T.BLUE, "Wholesale & MGA": T.SLATE, "Insurtech": T.RED}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading live prices…")
def load_prices() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    try:
        raw = yf.download(PX_TICKERS, start="2019-12-31", auto_adjust=False, progress=False, threads=False)
        tr, cl = raw["Adj Close"], raw["Close"]
        if tr.dropna(how="all").shape[0] < 500 or tr.notna().any().sum() < len(PX_TICKERS) - 4:
            raise RuntimeError("incomplete download")
        return tr.sort_index(), cl.sort_index(), "live"
    except Exception:
        tr, cl = db.prices()
        return tr, cl, "snapshot"


@st.cache_data
def load_static() -> tuple[pd.DataFrame, pd.DataFrame]:
    return db.fundamentals_raw(), db.splits()


@st.cache_resource(ttl=3600, show_spinner="Testing industry series…")
def industry_build():
    return IND.build({})


@st.cache_data(ttl=3600, show_spinner="Running statistical tests…")
def compute(active: tuple[str, ...], z_cut: float, px_tag: str):
    tr, cl, _ = load_prices()
    raw, splits = load_static()
    fund = A.build_fundamentals(raw[raw["ticker"].isin(ALL_TICKERS)], cl, splits)
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
    return dict(fund=fund_a, fund_all=fund, decomp=dec, decomp_all=dec_all, res=res, shocks=shocks, years=years,
                y0=y0, y1=y1, rel=rel, weekly=weekly, best_break=A.best_break(weekly))


@st.cache_data(ttl=3600)
def sub_build(px_tag: str):
    tr, cl, _ = load_prices()
    ct = SB.company_table(tr, cl, load_static()[0])
    return ct, SB.sub_index(tr), SB.tests(ct)


tr, cl, px_source = load_prices()
last_px = tr.dropna(how="all").index[-1]
build_log = db.table("build_log")
built_at = pd.Timestamp(build_log["built_at"].iloc[0]).tz_localize(None)
deck_v = pd.Timestamp(db.q("SELECT MAX(vintage_date) v FROM sources")["v"].iloc[0])


def stamp(kind: str) -> None:
    """Quiet data-freshness line under each visual."""
    px_age = (pd.Timestamp.today().normalize() - last_px).days
    txt = {
        "px": (f"prices {last_px:%b %d, %Y} · {'live' if px_source == 'live' else 'snapshot'}", px_age > 5),
        "sec": (f"SEC FY{y1} · built {built_at:%b %d}", False),
        "deck": (f"deck {deck_v:%b %d, %Y} · reconciled {built_at:%b %d}", (built_at - deck_v).days > 200),
        "fred": (f"FRED pulled {built_at:%b %d, %Y}", False),
        "ctx": (f"sourced facts · researched {built_at:%b %d, %Y}", False),
        "mix": (f"prices {last_px:%b %d} · SEC FY{y1} · built {built_at:%b %d, %Y}", px_age > 5),
    }[kind]
    T.fresh("updated: " + txt[0], stale=txt[1])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Analysis settings")
    excl = st.toggle("Exclude statistical outliers", value=False,
                     help="Drops carriers whose 5-year TSR has a robust (median/MAD) z-score beyond ±2.5.")
    z_cut = st.slider("Shock threshold (σ)", 2.5, 4.0, 3.0, 0.25,
                      help="A day counts as a sector-specific shock when the beta-adjusted residual exceeds this many rolling standard deviations.")
    st.markdown("**Carrier segments**")
    seg_pick = [s for s in T.SEGMENT_COLORS if st.checkbox(s, value=True, key=f"seg_{s}")]
    st.caption("Settings recalculate every test, chart, insight and narrative.")
    st.caption(f"Mutuals without listed equity ({', '.join(MUTUALS)}) appear only inside industry aggregates.")

base = compute(tuple(t for t in ALL_TICKERS if SEG[t] in seg_pick) or tuple(ALL_TICKERS), z_cut, px_source)
y1 = base["y1"]
outlier_list = A.outliers(base["decomp_all"].set_index("ticker")["tsr_pct"])
active = tuple(t for t in ALL_TICKERS if SEG[t] in seg_pick and not (excl and t in outlier_list)) or tuple(ALL_TICKERS)
ctx = compute(active, z_cut, px_source)
y0, y1 = ctx["y0"], ctx["y1"]
ind_cards, ind = industry_build()
sub_ct, sub_idx, sub_t = sub_build(px_source)
cyf = CY.facts()
F = SY.facts(ctx, ind, sub_ct, sub_t, cyf)
S = SY.build(F)

# ---------------------------------------------------------------------------
# Header + insight deck (filterable by year)
# ---------------------------------------------------------------------------
st.markdown(
    f'<div class="topbar"><div class="t">US P&amp;C <span>Value Signals</span></div>'
    f'<div class="m">{len(UNIVERSE)} listed companies · {len(active)} carriers in analytics · industry data reconciled to FRED &amp; SEC</div></div>',
    unsafe_allow_html=True,
)
stamp("mix")

pool = I.build(ctx) + ind_cards
years_avail = sorted({i.year for i in pool if i.year and i.year >= y0}, reverse=True)
if "deal" not in st.session_state:
    st.session_state.deal = 0
h1, h2, h3 = st.columns([1.8, 3.8, 0.9])
h1.markdown(f"**Market insights** &nbsp;<span style='color:{T.SLATE};font-size:.85rem'>{len(pool)} tested findings (p&lt;0.10)</span>",
            unsafe_allow_html=True)
with h2:
    opts = ["Latest"] + [str(y) for y in years_avail] + ["All"]
    try:
        yr_pick = st.segmented_control("Insight year", opts, default="Latest", label_visibility="collapsed", key="yr_pick")
    except Exception:  # older Streamlit
        yr_pick = st.radio("Insight year", opts, horizontal=True, label_visibility="collapsed", key="yr_pick")
    yr_pick = yr_pick or "Latest"
if h3.button("↻ Shuffle", use_container_width=True):
    st.session_state.deal += 1

if yr_pick == "Latest":
    cards = I.deal(pool, 3, st.session_state.deal, latest_year=y1)
else:
    sub_pool = pool if yr_pick == "All" else [i for i in pool if i.year == int(yr_pick)]
    if st.session_state.deal == 0:
        ranked = sorted(sub_pool, key=lambda i: i.p)
    else:
        rng = np.random.default_rng(st.session_state.deal)
        ranked = [sub_pool[k] for k in rng.permutation(len(sub_pool))]
    seen, cards = set(), []
    for ins in ranked:  # distinct themes first, then fill
        if ins.theme not in seen:
            cards.append(ins)
            seen.add(ins.theme)
    cards += [i for i in ranked if i not in cards]
    cards = cards[:3]

cols = st.columns(3)
for col, ins in zip(cols, cards):
    cls, label = T.strength(ins.p)
    with col:
        sw = f'<div class="ev" style="color:{T.NAVY}"><b>So what:</b> {ins.so_what}</div>' if ins.so_what else ""
        st.markdown(
            f'<div class="icard" style="--accent:{ins.accent}">'
            f'<div class="row"><span class="chip">{ins.theme}{" · " + str(ins.year) if ins.year else ""}</span>'
            f'<span class="badge {cls}">{label}</span></div>'
            f'<div class="h">{ins.headline}</div>'
            f'<div class="num">{ins.value}</div><div class="lbl">{ins.value_label}</div>'
            f'<div class="ev">{ins.evidence}</div>{sw}'
            f'<div class="src">n={ins.n} · {ins.source}</div></div>',
            unsafe_allow_html=True,
        )
        if ins.fig or ins.table is not None:
            with st.expander("Supporting evidence"):
                if ins.fig:
                    st.plotly_chart(ins.fig(), width="stretch", config=CFG, key=f"ev_{ins.key}_{st.session_state.deal}_{yr_pick}")
                if ins.table is not None:
                    tbl = ins.table.copy()
                    tbl.index = [A.NAME.get(i, i) for i in tbl.index]
                    st.dataframe(tbl, width="stretch", height=min(240, 38 + 35 * len(tbl)))
if not cards:
    st.info("No statistically supported findings for this year yet.")

st.write("")
tabs = st.tabs(["Executive story", "Industry", "Sub-industries", "Cyber & AI", "Market signals", "Value drivers",
                "Fundamentals", "Year by year", "Data & methods"])

# ---------------------------------------------------------------------------
# Executive story
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("## The story in five themes")
    st.markdown('<p class="lede">Every claim below points to a chart and a test elsewhere in this app. Themes are ranked by '
                "how much they should change what an executive team does in the next 12 months.</p>", unsafe_allow_html=True)
    cards_html = "".join(
        f'<div class="tcard" style="--accent:{T.THEME_ACCENT[t["t"]]}"><div class="n">THEME {t["n"]}</div>'
        f'<div class="t">{t["t"]}</div><div class="k">{t["k"]}</div><div class="e">{t["e"]}</div></div>'
        for t in SY.themes(F))
    st.markdown(f'<div class="theme-row">{cards_html}</div>', unsafe_allow_html=True)
    stamp("mix")
    st.write("")
    l, r = st.columns([1.6, 1])
    with l:
        html = "".join(f"<h3>{h}</h3><p>{b}</p>" for h, b in SY.executive(F))
        st.markdown(f'<div class="story">{html}</div>', unsafe_allow_html=True)
    with r:
        st.markdown("#### Where to look")
        st.markdown(
            "| Theme | Evidence tab |\n|---|---|\n"
            f"| {SY.TH1} | Industry · Market signals · Fundamentals |\n"
            f"| {SY.TH2} | Value drivers · Fundamentals |\n"
            f"| {SY.TH3} | Industry · Market signals |\n"
            f"| {SY.TH4} | Sub-industries |\n"
            f"| {SY.TH5} | Cyber & AI |")
        st.markdown("#### Investments that matter")
        st.markdown(
            "- **Pricing speed:** indication-to-filing in weeks (the decisive capability in personal auto this cycle)\n"
            "- **Property data:** geospatial, roof, wildfire and convective-storm scoring as rating variables\n"
            "- **AI in the workflow:** submission triage, claims severity prediction, reserve early-warning\n"
            "- **Delegated-authority oversight:** real-time bordereaux and MGA performance data\n"
            "- **Cyber prevention as product:** scanning, response and controls bundled into cover")
        st.markdown("#### Behaviours to reinforce")
        st.markdown(
            "- Reward underwriters for business **declined**, not just bound\n"
            "- Report **ex-AOCI** book value and **accident-year** loss ratios\n"
            "- Guide conservatively; surface reserve philosophy **before** earnings\n"
            "- Treat excess capital as **shareholder money** in a soft market")

# ---------------------------------------------------------------------------
# Industry
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("## Industry")
    st.markdown('<p class="lede">Industry results compiled in the USC Risk &amp; Uncertainty Management Center presentations '
                "(latest vintage of each figure), reconciled against FRED and this project's SEC-filing universe.</p>",
                unsafe_allow_html=True)
    cr_s, cr_f, cr_d = ind["cr"]
    ci_s, ci_b, ci_d = ind["ciab"]
    cat_s, cat_b, cat_d = ind["cat"]
    roe = IND.series("industry_roe", forecast=True)
    ytd = db.metric("industry_combined_ratio", period_type="half")
    k = st.columns(4)
    k[0].markdown(T.kpi(f"{(ytd['value'].iloc[-1] if len(ytd) else cr_s.iloc[-1]):.1f}",
                        f"Combined ratio, {ytd['period'].iloc[-1] if len(ytd) else cr_s.index[-1]}",
                        f"FY{cr_s.index[-1]}: {cr_s.iloc[-1]:.1f} · forecast {cr_f.index[-1]}: {cr_f.iloc[-1]:.1f}" if len(cr_f) else ""),
                  unsafe_allow_html=True)
    k[1].markdown(T.kpi(f"{ci_s.iloc[-1]:+.1f}%", f"Commercial rate change, {ci_s.index[-1]}",
                        f"regime avg since {ci_b[-1]['date']}: {ci_s.iloc[ci_b[-1]['index']:].mean():+.1f}%" if ci_b else ""),
                  unsafe_allow_html=True)
    k[2].markdown(T.kpi(f"${cat_s.iloc[-1]:.0f}B", f"Insured cat losses, {cat_s.index[-1]} (real)",
                        f"since {cat_b[-1]['date']}: avg ${cat_s[cat_s.index >= cat_b[-1]['date']].mean():.0f}B/yr" if cat_b else ""),
                  unsafe_allow_html=True)
    ra = roe[roe["is_forecast"] == 0].iloc[-1]
    rf = roe[roe["is_forecast"] == 1]
    k[3].markdown(T.kpi(f"{ra['value']:.1f}%", f"Industry ROE, {ra['period']}",
                        f"forecast {rf['period'].iloc[-1]}: {rf['value'].iloc[-1]:.1f}%" if len(rf) else ""), unsafe_allow_html=True)
    st.write("")
    l, r = st.columns(2)
    with l:
        st.plotly_chart(IND.regimes_fig(ci_s, ci_b, "Commercial pricing regimes: average rate change, all lines (%)", "%", T.NAVY),
                        width="stretch", config=CFG)
        stamp("deck")
        T.sowhat(S.get("ind_ciab"), "ind_ciab")
    with r:
        st.plotly_chart(IND.regimes_fig(cr_s, [], "Industry combined ratio (%), actual and deck forecast", "", T.NAVY,
                                        forecast=cr_f, zero=False), width="stretch", config=CFG)
        stamp("deck")
        T.sowhat(S.get("ind_cr"), "ind_cr")
    l, r = st.columns(2)
    with l:
        st.plotly_chart(IND.regimes_fig(cat_s, cat_b, "Insured catastrophe losses, inflation-adjusted ($B)", "B", T.MAGENTA,
                                        zero=False), width="stretch", config=CFG)
        stamp("deck")
        T.sowhat(S.get("ind_cat"), "ind_cat")
    with r:
        j = ind.get("dwp_gdp")
        if j is not None and len(j):
            fig = go.Figure()
            fig.add_scatter(x=j.index, y=j["premium"], name="Direct premium growth (deck)", line=dict(color=T.NAVY, width=2))
            fig.add_scatter(x=j.index, y=j["gdp"], name="Nominal GDP growth (FRED, live)", line=dict(color=T.BLUE, width=2, dash="dash"))
            fig.add_hline(y=0, line_color=T.MID)
            fig.update_layout(title="Premium growth vs nominal GDP growth (y/y %)", height=360, xaxis=dict(type="category"),
                              margin=dict(t=64, b=70))
            st.plotly_chart(fig, width="stretch", config=CFG)
            stamp("fred")
            T.sowhat(S.get("ind_gdp"), "ind_gdp")

    st.markdown("#### Deck figures against our own data")
    l, r = st.columns(2)
    fa = ctx["fund_all"]
    with l:
        pa = IND.series("personal_auto_combined_ratio")
        pers = fa[(fa["segment"] == "Personal") & fa["loss_ratio_pct"].notna()]
        pl = pers.groupby("fiscal_year").apply(lambda g: g["losses_incurred"].sum() / g["premiums_earned"].sum() * 100)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=pa["period"], y=pa["value"], name="Personal auto combined ratio (industry)", line=dict(color=T.MAGENTA, width=2.2))
        fig.add_scatter(x=[str(y) for y in pl.index], y=pl.values, name="Personal carriers' loss ratio (SEC, ours)",
                        line=dict(color=T.NAVY, width=2, dash="dash"), secondary_y=True)
        fig.update_layout(title="Personal auto: industry combined ratio vs our personal carriers", height=360,
                          xaxis=dict(type="category"), margin=dict(t=64, b=80))
        fig.update_yaxes(title_text="Combined ratio %", secondary_y=False)
        fig.update_yaxes(title_text="Loss ratio %", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, width="stretch", config=CFG)
        stamp("sec")
        T.sowhat(S.get("ind_pa"), "ind_pa")
    with r:
        y10 = db.fred("DGS10").resample("YE").last()
        ao = fa.groupby("fiscal_year")["aoci_to_tbv_pct"].median()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=[str(y) for y in ao.index if y >= 2019], y=[ao[y] for y in ao.index if y >= 2019],
                    name="Median AOCI as % of TBV (SEC, ours)", marker_color=T.MAGENTA)
        yy = y10[y10.index.year >= 2019]
        fig.add_scatter(x=[str(d.year) for d in yy.index], y=yy.values, name="10-yr Treasury, year-end (FRED)",
                        line=dict(color=T.NAVY, width=2.4), secondary_y=True)
        fig.update_layout(title="Rates explain the 2022 book-value shock", height=360, xaxis=dict(type="category"),
                          margin=dict(t=64, b=80))
        fig.update_yaxes(title_text="AOCI % of TBV", secondary_y=False)
        fig.update_yaxes(title_text="10-yr yield %", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, width="stretch", config=CFG)
        stamp("fred")
        T.sowhat(S.get("ind_aoci"), "ind_aoci")

# ---------------------------------------------------------------------------
# Sub-industries
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("## Sub-industries and market lenses")
    st.markdown('<p class="lede">The same cycle hits each link of the chain differently: carriers (personal, commercial, '
                "specialty / E&amp;S), reinsurers, retail brokers, wholesale and MGA/MGU platforms, and insurtech. Admitted vs "
                "non-admitted and segment lenses slice the carrier view.</p>", unsafe_allow_html=True)
    med = sub_ct.groupby("sub_industry")[["ret_1y", "pe"]].median()
    k = st.columns(5)
    for col, sub in zip(k, SB.ORDER):
        if sub in med.index:
            n = int((sub_ct["sub_industry"] == sub).sum())
            pe = med.loc[sub, "pe"]
            col.markdown(T.kpi(f"{med.loc[sub, 'ret_1y']:+.0f}%", f"{sub} · 1-yr median return",
                               f"n={n}" + (f" · P/E {pe:.0f}×" if pd.notna(pe) else "")), unsafe_allow_html=True)
    st.write("")
    l, r = st.columns([1.2, 1])
    with l:
        fig = go.Figure()
        for sub in SB.ORDER:
            if sub in sub_idx:
                fig.add_scatter(x=sub_idx.index, y=sub_idx[sub], name=sub, line=dict(color=SUB_COLORS[sub], width=2.2))
        sp = tr["^GSPC"][tr.index >= "2020-12-31"].dropna()
        fig.add_scatter(x=sp.index, y=sp / sp.iloc[0] * 100, name="S&P 500", line=dict(color=T.MID, width=2))
        fig.update_layout(title="Equal-weight total return by sub-industry (end-2020 = 100)", height=400, margin=dict(t=56, b=80))
        st.plotly_chart(fig, width="stretch", config=CFG)
        stamp("px")
        T.sowhat(S.get("sub_returns"), "sub_returns")
    with r:
        gcol = sub_t.get("growth_col", "rev_growth_2025")
        d = sub_ct.dropna(subset=["pe", gcol])
        d = d[d["pe"] < 60]
        fig = go.Figure()
        for sub in SB.ORDER:
            s_ = d[d["sub_industry"] == sub]
            if len(s_):
                far = (s_["pe"] > d["pe"].quantile(.8)) | (s_[gcol] > d[gcol].quantile(.8)) | (s_["sub_industry"] != "Carrier")
                fig.add_scatter(x=s_[gcol], y=s_["pe"], mode="markers+text", text=np.where(far, s_["ticker"], ""), textposition="top center",
                                name=sub, marker=dict(size=np.clip(np.sqrt(s_["market_cap_bn"].fillna(5)) * 3, 8, 34),
                                                      color=SUB_COLORS[sub], opacity=.85, line=dict(width=1, color="white")),
                                textfont=dict(size=10, color=T.INK),
                                customdata=s_["name"], hovertemplate="%{customdata}<br>growth %{x:.1f}% · P/E %{y:.1f}×<extra></extra>")
        fig.update_layout(title=f"P/E vs {gcol[-4:]} revenue growth (bubble = market cap)", height=400,
                          xaxis_title="Revenue growth %", yaxis_title="P/E (×)", margin=dict(t=56, b=80))
        st.plotly_chart(fig, width="stretch", config=CFG)
        stamp("mix")
        T.sowhat(S.get("sub_val"), "sub_val")

    l, r = st.columns([1, 1.2])
    with l:
        car = sub_ct[sub_ct["sub_industry"] == "Carrier"].copy()
        car["view"] = car["market"].replace({"Mixed": "Mixed admitted / E&S"})
        g = car.groupby("view")[[gcol, "ret_1y"]].median().reindex(["Admitted", "Mixed admitted / E&S", "Non-admitted (E&S)"])
        cnt = car["view"].value_counts()
        fig = go.Figure()
        fig.add_bar(x=g.index, y=g[gcol], name="Median revenue growth %", marker_color=T.NAVY,
                    text=[f"{v:.0f}%" for v in g[gcol]], textposition="outside", cliponaxis=False)
        fig.add_bar(x=g.index, y=g["ret_1y"], name="Median 1-yr return %", marker_color=T.BLUE_TINT,
                    text=[f"{v:.0f}%" for v in g["ret_1y"]], textposition="outside", cliponaxis=False)
        fig.update_xaxes(ticktext=[f"{i}<br>(n={cnt.get(i, 0)})" for i in g.index], tickvals=list(g.index))
        fig.update_layout(title="Carriers by market: admitted vs non-admitted (E&S)", barmode="group", height=380,
                          margin=dict(t=56, b=80))
        st.plotly_chart(fig, width="stretch", config=CFG)
        stamp("mix")
        T.sowhat(S.get("sub_lens"), "sub_lens")
    with r:
        lens = sub_ct.groupby("lens").agg(n=("ticker", "count"), ret_1y=("ret_1y", "median"), growth=(gcol, "median"),
                                          pe=("pe", "median"), vol=("vol_1y", "median")).sort_values("ret_1y")
        fig = go.Figure(go.Bar(x=lens["ret_1y"], y=[f"{i} (n={n})" for i, n in zip(lens.index, lens["n"])], orientation="h",
                               marker_color=[T.POS if v >= 0 else T.NEG for v in lens["ret_1y"]],
                               text=[f"{v:+.0f}%" for v in lens["ret_1y"]], textposition="outside", cliponaxis=False))
        lo = min(0, lens["ret_1y"].min())
        fig.update_layout(title="1-year median return by lens", height=380, showlegend=False, margin=dict(t=56, b=30, r=40),
                          xaxis=dict(range=[lo * 1.35, lens["ret_1y"].max() * 1.25]))
        st.plotly_chart(fig, width="stretch", config=CFG)
        stamp("px")
        T.sowhat(S.get("sub_context"), "sub_context")

    st.markdown("#### Market context by link in the chain")
    ctxf = db.q("SELECT domain, metric, period, value, unit, note, source_name, source_url FROM context_facts "
                "WHERE domain IN ('E&S','MGA','Reinsurance','Brokers','Pricing') ORDER BY domain, metric, period")
    st.dataframe(ctxf, hide_index=True, width="stretch", height=300,
                 column_config={"source_url": st.column_config.LinkColumn("Source")})
    stamp("ctx")
    st.markdown("#### Company table")
    show = sub_ct.copy()
    show["listed_since"] = show["listed_since"].astype(str).replace("None", "")
    st.dataframe(show.round(1), hide_index=True, width="stretch", height=360)
    st.caption(f"Mutual and private carriers ({', '.join(MUTUALS)}) have no listed equity and are captured only in industry aggregates. "
               "Revenue growth for brokers includes acquisitions (organic growth is shown in the context table).")

# ---------------------------------------------------------------------------
# Cyber & AI
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("## Cyber & AI insurance")
    st.markdown('<p class="lede">Cyber market size, pricing, losses and claims trends, and the emerging market for AI liability: '
                "who is covering it, who is excluding it, and what that implies.</p>", unsafe_allow_html=True)
    k = st.columns(4)
    k[0].markdown(T.kpi(f"${F.get('cy_prem_25', np.nan):.2f}B", "US cyber DWP, 2025", "NAIC · incl. alien surplus lines"), unsafe_allow_html=True)
    k[1].markdown(T.kpi(f"{F.get('cy_lr_25', np.nan):.0f}%", "US cyber loss ratio, 2025", f"up from {F.get('cy_lr_23', np.nan):.0f}% in 2023"), unsafe_allow_html=True)
    k[2].markdown(T.kpi(f"{F.get('cy_pol', np.nan):.2f}M", "Cyber policies in force", f"{F.get('cy_endo', np.nan):.0f}% are endorsements"), unsafe_allow_html=True)
    k[3].markdown(T.kpi(f"${F.get('cy_global', np.nan):.1f}B", "Global cyber premium, 2024", "Munich Re · >2× by 2030 expected"), unsafe_allow_html=True)
    st.write("")
    l, r = st.columns(2)
    with l:
        st.plotly_chart(CY.market_fig(cyf), width="stretch", config=CFG)
        stamp("ctx")
        T.sowhat(S.get("cy_market"), "cy_market")
    with r:
        st.plotly_chart(CY.rates_fig(cyf), width="stretch", config=CFG)
        stamp("ctx")
        T.sowhat(S.get("cy_rates"), "cy_rates")
    l, r = st.columns(2)
    with l:
        st.plotly_chart(CY.claims_fig(cyf), width="stretch", config=CFG)
        stamp("ctx")
        T.sowhat(S.get("cy_claims"), "cy_claims")
    with r:
        ev = CY.ai_events()
        st.plotly_chart(CY.ai_timeline_fig(ev), width="stretch", config=CFG)
        stamp("ctx")
        T.sowhat(S.get("cy_ai"), "cy_ai")
    sf = CY.specialist_fig(tr)
    if sf is not None:
        l, r = st.columns([1.3, 1])
        with l:
            st.plotly_chart(sf, width="stretch", config=CFG)
            stamp("px")
        with r:
            T.sowhat(S.get("cy_specialist"), "cy_specialist")
            st.markdown("#### AI insurance metrics to track")
            st.markdown(
                "| Metric | Why it matters |\n|---|---|\n"
                "| Affirmative AI products vs exclusion actions | Direction of the coverage gap |\n"
                "| ISO GenAI endorsement adoption | How fast silent AI becomes excluded AI |\n"
                "| AI liability limits available (Lloyd's, specialists) | Capacity = market maturity |\n"
                "| BEC / funds-transfer fraud frequency | AI-enabled social engineering losses |\n"
                "| First AI liability verdicts | When the line gets priced in earnest |")
    st.markdown("#### Cyber & AI timeline")
    tl = ev.copy()
    tl["date"] = tl["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(tl[["date", "category", "label", "detail", "confidence", "source_name", "source_url"]].sort_values("date", ascending=False),
                 hide_index=True, width="stretch", height=320, column_config={"source_url": st.column_config.LinkColumn("Source")})
    stamp("ctx")

# ---------------------------------------------------------------------------
# Market signals
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("## Market signals")
    st.markdown('<p class="lede">How the equal-weight carrier basket has traded against the S&amp;P 500, which days moved it '
                "for insurance-specific reasons and why, and whether its trend has statistically changed.</p>", unsafe_allow_html=True)
    res, shocks, weekly, rel = ctx["res"], ctx["shocks"], ctx["weekly"], ctx["rel"]
    cum_ex = ((1 + weekly).cumprod() - 1) * 100
    k = st.columns(4)
    bk = A.basket_returns(tr[tr.index >= f"{y0}-12-31"], list(active))
    yrs = (bk.index[-1] - bk.index[0]).days / 365.25
    b_ann = ((1 + bk).prod() ** (1 / yrs) - 1) * 100
    s = tr["^GSPC"].dropna()
    s = s[s.index >= f"{y0}-12-31"]
    m_ann = ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
    k[0].markdown(T.kpi(f"{b_ann:.1f}%", "Carrier basket TSR, annualized", f"since end-{y0} · equal weight"), unsafe_allow_html=True)
    k[1].markdown(T.kpi(f"{b_ann - m_ann:+.1f} pp", "vs S&P 500 per year", f"S&P 500 {m_ann:.1f}%/yr"), unsafe_allow_html=True)
    k[2].markdown(T.kpi(f"{res['beta'].iloc[-1]:.2f}", "Current market beta", "rolling 1-year, lagged"), unsafe_allow_html=True)
    k[3].markdown(T.kpi(f"{len(shocks)}", "Sector-specific shock days",
                        f"{(shocks['z'] < 0).sum()} sell-offs · {shocks['catalyst'].notna().sum()} attributed"), unsafe_allow_html=True)
    st.write("")
    ex_shock = st.toggle("Remove shock days from the trend", value=False)
    fig = go.Figure()
    fig.add_scatter(x=cum_ex.index, y=cum_ex, mode="lines", name="Cumulative excess return vs S&P 500", line=dict(color=T.NAVY, width=2.2))
    shock_contrib = None
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
                            customdata=np.stack([d["residual_pct"], d["z"], d["catalyst"].fillna("Unattributed"), d["why"].fillna("")], axis=1),
                            hovertemplate="%{x|%b %d, %Y}<br>Residual %{customdata[0]:.1f}% (z %{customdata[1]:.1f})<br><b>%{customdata[2]}</b><br>%{customdata[3]}<extra></extra>")
        for dt, r_ in shocks.head(4).iterrows():
            if isinstance(r_["catalyst"], str):
                fig.add_annotation(x=dt, y=cum_ex.reindex([dt], method="ffill").iloc[0], text=r_["catalyst"][:42], showarrow=True,
                                   arrowhead=0, ax=0, ay=-40 if r_["z"] < 0 else 40, font=dict(size=11, color=T.NAVY),
                                   bgcolor="rgba(255,255,255,.88)")
    fig.add_hline(y=0, line_color=T.MID)
    fig.update_layout(title="Carrier basket: cumulative return in excess of the S&P 500 (%)", height=440, margin=dict(b=70))
    st.plotly_chart(fig, width="stretch", config=CFG)
    stamp("px")
    if shock_contrib is not None:
        T.note(f"Shock days account for {shock_contrib:+.1f} pp of the cumulative excess return.")
    T.sowhat(S.get("ms_excess"), "ms_excess")
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
        stamp("px")
        T.sowhat(S.get("ms_beta"), "ms_beta")
    with r:
        fig = go.Figure()
        for seg_name, colr in T.SEGMENT_COLORS.items():
            ts = [t for t in active if SEG.get(t) == seg_name]
            if ts:
                br_ = A.basket_returns(tr[tr.index >= f"{y0}-12-31"], ts)
                fig.add_scatter(x=br_.index, y=(1 + br_).cumprod() * 100, mode="lines", name=f"{seg_name} ({len(ts)})",
                                line=dict(color=colr, width=2))
        fig.add_scatter(x=s.index, y=s / s.iloc[0] * 100, mode="lines", name="S&P 500", line=dict(color=T.MID, width=2))
        fig.update_layout(title=f"Total return by carrier segment (end-{y0} = 100)", height=360, margin=dict(b=90))
        st.plotly_chart(fig, width="stretch", config=CFG)
        stamp("px")
        T.sowhat(S.get("ms_segments"), "ms_segments")
    st.markdown("#### Shock-day log: what happened and why")
    if len(shocks):
        show = shocks.copy()
        show.index = show.index.strftime("%Y-%m-%d")
        show["catalyst"] = show["catalyst"].fillna("Unattributed: no verified event within ±1 day")
        st.dataframe(show[["direction", "residual_pct", "z", "catalyst", "why", "event_category", "confidence", "source_url",
                           "basket_pct", "sp500_pct", "carriers_down_pct"]]
                     .rename(columns={"residual_pct": "Residual %", "basket_pct": "Basket %", "sp500_pct": "S&P 500 %",
                                      "carriers_down_pct": "% carriers down", "z": "z", "why": "Why it moved", "event_category": "Type"}).round(2),
                     width="stretch", height=340, column_config={"source_url": st.column_config.LinkColumn("Source")})
        stamp("ctx")
        T.note("Residual = basket return minus rolling-beta × S&P 500 (and rolling alpha). Shocks are detected statistically first; "
               "causes are attached afterwards from a researched, sourced event log. 'Probable' = cause consistent with the move "
               "but not stated explicitly by the source.")

# ---------------------------------------------------------------------------
# Value drivers
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("## Value drivers")
    st.markdown(f'<p class="lede">Where each carrier\'s {y0}–{y1} shareholder return came from: growth in tangible book per '
                "share, change in the price-to-tangible-book multiple, and cash returned.</p>", unsafe_allow_html=True)
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
                    marker=dict(symbol="line-ns-open", size=20, color=T.INK, line=dict(width=3)), hovertemplate="TSR %{x:.1f}%<extra></extra>")
    med_ = dec["tsr_pct"].median()
    fig.add_vline(x=med_, line_dash="dash", line_color=T.MAGENTA)
    fig.add_annotation(x=med_, y=1, yref="paper", yanchor="bottom", text=f"median {med_:.1f}%", showarrow=False, font=dict(color=T.NAVY, size=11))
    fig.update_layout(barmode="relative", title=f"Annualized TSR decomposition, end-{y0} to end-{y1} (pp per year)",
                      height=max(420, 30 * len(dec) + 150), margin=dict(t=64, b=70, r=20), bargap=0.22)
    st.plotly_chart(fig, width="stretch", config=CFG)
    stamp("mix")
    T.sowhat(S.get("vd_decomp"), "vd_decomp")
    l, r = st.columns(2)
    dd = ctx["decomp"].set_index("ticker")
    with l:
        st.plotly_chart(I._scatter_fit(dd, "multiple_change_pp", "tsr_pct", "P/TBV re-rating (pp/yr)", "TSR (%/yr)",
                                       "What explains TSR? Re-rating"), width="stretch", config=CFG)
        stamp("mix")
        T.sowhat(S.get("vd_rerating"), "vd_rerating")
    with r:
        st.plotly_chart(I._scatter_fit(dd, "p_tbv_start", "multiple_change_pp", f"P/TBV at end-{y0} (x)", "Re-rating (pp/yr)",
                                       "Starting valuation vs re-rating"), width="stretch", config=CFG)
        stamp("mix")
        T.sowhat(S.get("vd_meanrev"), "vd_meanrev")
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
    inf = A.influence(ctx["decomp_all"].set_index("ticker")["tsr_pct"])
    inf.index = [NAME[t] for t in inf.index]
    l, r = st.columns([1.3, 1])
    with l:
        st.plotly_chart(I._bars(inf["influence_pp"], "Leave-one-out influence on universe average TSR", " pp"), width="stretch", config=CFG)
        stamp("mix")
    with r:
        T.sowhat(S.get("vd_influence"), "vd_influence")
        seg_tbl = ctx["decomp"].groupby("segment")[["tsr_pct", "tbvps_growth_pp", "multiple_change_pp", "cash_yield_pp"]].median()
        seg_tbl.columns = ["TSR %", "TBV/sh pp", "Re-rating pp", "Cash pp"]
        st.dataframe(seg_tbl.round(1), width="stretch")
        st.caption(f"Segment medians. Outlier exclusion is {'on' if excl else 'off'} (sidebar). Carriers listed after {y0} are "
                   "excluded from the 5-year decomposition.")

# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
with tabs[6]:
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
        fig.update_layout(title=title, height=340, margin=dict(t=64, b=30), xaxis=dict(type="category", categoryorder="category ascending"))
        return fig

    l, r = st.columns(2)
    with l:
        st.plotly_chart(dist("tbvps_growth_pct", "TBV/share growth (%)", "%", "tbvps_growth_p", T.NAVY), width="stretch", config=CFG)
        stamp("sec")
        T.sowhat(S.get("fu_tbv"), "fu_tbv")
    with r:
        st.plotly_chart(dist("aoci_to_tbv_pct", "Unrealized gains/losses (AOCI) as % of TBV", "%", "aoci_p", T.MAGENTA), width="stretch", config=CFG)
        stamp("sec")
        T.sowhat(S.get("fu_aoci"), "fu_aoci")
    l, r = st.columns(2)
    with l:
        st.plotly_chart(dist("rote_pct", "Return on tangible equity (%)", "%", "rote_p", T.BLUE), width="stretch", config=CFG)
        stamp("sec")
        T.sowhat(S.get("fu_rote"), "fu_rote")
    with r:
        st.plotly_chart(dist("p_tbv", "Price / tangible book (x)", "x", None, T.SLATE), width="stretch", config=CFG)
        stamp("mix")
        T.sowhat(S.get("fu_ptbv"), "fu_ptbv")
    st.markdown("#### Loss ratio by carrier (losses and LAE ÷ net earned premium)")
    lr = fund.pivot(index="ticker", columns="fiscal_year", values="loss_ratio_pct").loc[:, y0:y1].dropna(how="all")
    lr = lr.loc[lr.mean(axis=1).sort_values().index]
    fig = go.Figure(go.Heatmap(z=lr.values, x=[f"FY{c}" for c in lr.columns], y=[f"{NAME[t]} ({SEG[t]})" for t in lr.index],
                               colorscale=[[0, T.BLUE], [0.5, "#F4F6FB"], [1, T.RED]], zmid=float(np.nanmedian(lr.values)),
                               text=np.where(np.isnan(lr.values), "", np.round(lr.values, 1).astype(str)),
                               texttemplate="%{text}", textfont=dict(size=11, color=T.INK),
                               colorbar=dict(title=dict(text="%", side="top"), thickness=10)))
    fig.update_layout(height=40 + 26 * len(lr), margin=dict(l=8, r=8, t=16, b=8), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config=CFG)
    stamp("sec")
    T.sowhat(S.get("fu_lr"), "fu_lr")

# ---------------------------------------------------------------------------
# Year by year
# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("## Year by year: inflection points")
    st.markdown('<p class="lede">Each year is named only from shifts that pass a significance test across carriers, with the '
                "year's biggest attributed market shock alongside.</p>", unsafe_allow_html=True)
    ys = ctx["years"]
    sh_y = ctx["shocks"][ctx["shocks"]["catalyst"].notna()] if len(ctx["shocks"]) else ctx["shocks"]
    yc = st.columns(len(ys))
    for col, (_, row) in zip(yc, ys.iterrows()):
        title, notes = A.year_theme(row)
        accent = T.RED if any(w in title.lower() for w in ["shock", "squeeze", "inflation"]) else (
            T.BLUE if "no statistically" not in title.lower() else T.SLATE)
        g = sh_y[sh_y.index.year == int(row["year"])]
        ev_line = f"<li><i>Market moment:</i> {g.iloc[0]['catalyst']} ({g.iloc[0]['residual_pct']:+.1f}%)</li>" if len(g) else ""
        items = "".join(f"<li>{n}</li>" for n in notes) or "<li>No metric moved significantly.</li>"
        col.markdown(f'<div class="ycard" style="--accent:{accent}"><div class="y">FY{int(row["year"])}</div>'
                     f'<div class="h">{title}</div><ul>{items}{ev_line}</ul></div>', unsafe_allow_html=True)
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
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ylab, hovertext=hov, hoverinfo="text", zmin=-3, zmax=3, showscale=False, xgap=2, ygap=2,
                               colorscale=[[0, T.RED], [0.33, "#F8B3C0"], [0.5, "#F4F6FB"], [0.67, "#B3C6FF"], [1, T.BLUE]]))
    for yi, (zr, tr_) in enumerate(zip(z, txt)):
        for xi, (zv, tv) in enumerate(zip(zr, tr_)):
            fig.add_annotation(x=xs[xi], y=ylab[yi], text=tv, showarrow=False, font=dict(size=12, color=T.WHITE if abs(zv) >= 2 else T.INK))
    fig.update_layout(title="Signal matrix: color intensity = statistical strength, blue = favorable, red = adverse",
                      height=330, yaxis=dict(autorange="reversed"), xaxis=dict(type="category"), margin=dict(t=50, b=30))
    st.plotly_chart(fig, width="stretch", config=CFG)
    stamp("mix")
    T.sowhat(S.get("yy_matrix"), "yy_matrix")

# ---------------------------------------------------------------------------
# Data & methods
# ---------------------------------------------------------------------------
with tabs[8]:
    st.markdown("## Data & methods")
    rec = db.table("reconciliation")
    srcs = db.table("sources")
    st.markdown("#### Reconciliation")
    st.markdown('<p class="lede">Every deck figure with an independent public reference is checked against it at build time; '
                "&ldquo;close&rdquo; or &ldquo;variance&rdquo; results are explained, not dropped.</p>", unsafe_allow_html=True)
    stc = rec["status"].value_counts()
    kk = st.columns(5)
    for col, (lab, keys) in zip(kk, [("Match", ["match"]), ("Close", ["close"]), ("Variance", ["variance"]),
                                     ("Consistent / divergent", ["consistent", "divergent"]), ("Reference unavailable", ["reference unavailable"])]):
        col.markdown(T.kpi(f"{int(sum(stc.get(x, 0) for x in keys))}", lab, ""), unsafe_allow_html=True)
    summ = rec.groupby(["metric", "reference_source"])["status"].value_counts().unstack(fill_value=0).reset_index()
    st.dataframe(summ, hide_index=True, width="stretch", height=280)
    flt = st.multiselect("Show checks with status", sorted(rec["status"].unique()),
                         default=[s_ for s_ in ["variance", "close", "divergent"] if s_ in set(rec["status"])])
    st.dataframe(rec[rec["status"].isin(flt)].round(2), hide_index=True, width="stretch", height=260)
    st.caption(f"Database built {built_at:%Y-%m-%d %H:%M} UTC · " + " · ".join(build_log["detail"]))
    st.markdown("#### Sources")
    st.dataframe(srcs[["vintage_date", "title", "publisher", "url", "pdf_url", "sha256"]].sort_values("vintage_date", ascending=False),
                 hide_index=True, width="stretch", height=240,
                 column_config={"url": st.column_config.LinkColumn("Deck (pptx)"), "pdf_url": st.column_config.LinkColumn("PDF")})
    st.dataframe(db.table("fred_catalog"), hide_index=True, width="stretch", column_config={"url": st.column_config.LinkColumn("FRED page")})
    with open(DATA / "pc_market.db", "rb") as fh:
        st.download_button("Download the full database (SQLite)", fh.read(), file_name="pc_market.db", mime="application/x-sqlite3")
    st.markdown("#### Explore")
    datasets = {
        "Industry metrics (latest vintage)": db.q("SELECT l.metric, c.label, l.dimension, l.period, l.value, l.unit, l.flag, "
                                                  "l.vintage_date, l.source_id, l.slide FROM industry_latest l "
                                                  "LEFT JOIN metric_catalog c USING(metric) ORDER BY metric, dimension, period"),
        "Industry metrics (all vintages)": db.table("observations"),
        "Vintage revisions": db.table("revisions"),
        "Reconciliation checks": rec,
        "Event log (shocks, cat, macro, cyber, AI)": db.table("events"),
        "Market context facts": db.table("context_facts"),
        "Sub-industry company table": sub_ct,
        "Fundamentals by carrier-year (derived)": ctx["fund_all"][["ticker", "name", "segment", "fiscal_year", "premiums_earned",
                                                                  "net_income", "tbv", "tbvps", "p_tbv", "rote_pct", "loss_ratio_pct",
                                                                  "aoci_to_tbv_pct", "tbvps_growth_pct", "premium_growth_pct"]],
        "SEC XBRL extract (raw, all companies)": load_static()[0],
        f"TSR decomposition {y0}–{y1}": ctx["decomp_all"],
        "Year signals and tests": ctx["years"],
        "Shock days": ctx["shocks"].reset_index().rename(columns={"index": "date", "Date": "date"}),
        "Coverage universe": pd.DataFrame(UNIVERSE),
    }
    choice = st.selectbox("Dataset", list(datasets))
    df = datasets[choice].copy()
    if "segment" in df and df["segment"].notna().any():
        segs = st.multiselect("Segment filter", sorted(df["segment"].dropna().unique()), default=sorted(df["segment"].dropna().unique()))
        df = df[df["segment"].isin(segs) | df["segment"].isna()]
    st.dataframe(df.round(3) if hasattr(df, "round") else df, hide_index=True, width="stretch", height=380)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=choice.split(" (")[0].lower().replace(" ", "_") + ".csv",
                       mime="text/csv")
    st.markdown("#### Methods")
    st.markdown(
        "- **Database:** `data/pc_market.db` (SQLite), rebuilt by `scripts/build_database.py`: chart series embedded in the USC Risk "
        "& Uncertainty Management Center presentations (each deck's URL, vintage date and SHA-256 stored), all vintages kept; FRED "
        "series; SEC fundamentals for every listed company; prices; reconciliation and revision tables; a researched event log and "
        "market-context facts (each with source link and confidence).\n"
        f"- **Universe:** {len(UNIVERSE)} listed companies across carriers ({len(ALL_TICKERS)}), reinsurers, retail brokers, wholesale "
        "& MGA platforms and insurtech, each tagged by lens and admitted/non-admitted market (project classification). Mutuals have no "
        "listed equity and appear only in industry aggregates.\n"
        "- **Narrative:** every 'so what' is generated from the numbers on the page; role guidance (CFO / CUO / CEO) is analysis, "
        "not advice.\n"
        "- **Shocks:** beta-adjusted residual moves beyond the chosen σ; causes attached afterwards from sourced events (±1 day).\n"
        "- **Regimes:** CUSUM binary segmentation with permutation tests (moving-block for autocorrelated series).\n"
        "- **Insight cards:** findings with p < 0.10 only; filter by year or shuffle (weighted by statistical strength).\n"
        "- **Limits:** equal-weight baskets; XBRL coverage varies by filer (gaps left blank); context facts compiled from public "
        "reports and trade press where primary data is paywalled. Not investment advice.")
