"""Current-year (YTD) analytics: 10-Q fundamentals, YTD returns, event study, leader/laggard behaviours, implied returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

import db
import theme as T
from insights import Insight
from universe import CARRIERS, UNIVERSE

NAME = {c["ticker"]: c["name"] for c in UNIVERSE}
SUB = {c["ticker"]: c["sub_industry"] for c in UNIVERSE}
COE, G = 0.09, 0.03  # cost of equity and long-run growth used for market-implied returns (stated assumption)


def implied_rote(p_tbv: float) -> float:
    """Justified P/B = (ROE − g)/(COE − g)  ⇒  market-implied sustainable ROE = P/B·(COE − g) + g."""
    return (p_tbv * (COE - G) + G) * 100 if pd.notna(p_tbv) else np.nan


def ytd_table(tr: pd.DataFrame, cl: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    y = db.table("fundamentals_ytd").set_index("ticker")
    year = int(str(y["period_end"].iloc[0])[:4])
    y["tbv_now"] = y["equity_now"] - y["goodwill_now"].fillna(0) - y["intangibles_now"].fillna(0)
    y["tbv_ye"] = y["equity_ye"] - y["goodwill_ye"].fillna(0) - y["intangibles_ye"].fillna(0)
    lr = y["losses_incurred_ytd"] / y["premiums_earned_ytd"] * 100
    lrp = y["losses_incurred_ytd_prior"] / y["premiums_earned_ytd_prior"] * 100
    ok = lr.between(25, 120) & lrp.between(25, 120)
    out = pd.DataFrame(index=y.index)
    out["period_end"] = y["period_end"]
    out["loss_ratio_ytd"] = lr.where(ok)
    out["loss_ratio_prior_ytd"] = lrp.where(ok)
    out["loss_ratio_chg_pp"] = (lr - lrp).where(ok)
    out["net_income_growth_pct"] = ((y["net_income_ytd"] / y["net_income_ytd_prior"] - 1) * 100).where(y["net_income_ytd_prior"] > 0)
    out["revenue_growth_pct"] = (y["revenue_ytd"] / y["revenue_ytd_prior"] - 1) * 100
    out["premium_growth_pct"] = (y["premiums_earned_ytd"] / y["premiums_earned_ytd_prior"] - 1) * 100
    out["tbv_ytd_pct"] = ((y["tbv_now"] / y["tbv_ye"] - 1) * 100).where((y["tbv_ye"] > 0) & (y["tbv_now"] > 0))
    out["aoci_chg_pct_tbv"] = ((y["aoci_now"] - y["aoci_ye"]) / y["tbv_ye"] * 100).where(y["tbv_ye"] > 0)
    avg_tbv = (y["tbv_now"] + y["tbv_ye"]) / 2
    out["rote_ytd_ann"] = (y["net_income_ytd"] * (12 / y["months"]) / avg_tbv * 100).where((y["tbv_ye"] > 0) & (y["tbv_now"] > 0))
    start = pd.Timestamp(f"{year - 1}-12-31")
    px = tr.loc[tr.index >= start].ffill()
    out["return_ytd_pct"] = [(px[t].dropna().iloc[-1] / px[t].dropna().iloc[0] - 1) * 100 if t in px and px[t].notna().sum() > 20 else np.nan
                             for t in out.index]
    # current P/TBV on latest 10-Q book and FY-end share count (split-adjusted via fundamentals)
    sh = fund[fund["fiscal_year"] == year - 1].set_index("ticker")["shares_adj"] if "shares_adj" in fund else pd.Series(dtype=float)
    last = cl.ffill().iloc[-1]
    out["p_tbv_now"] = [(last.get(t) / (y.loc[t, "tbv_now"] / sh.get(t))) if t in sh.index and pd.notna(sh.get(t)) and y.loc[t, "tbv_now"] > 0 else np.nan
                        for t in out.index]
    out["implied_rote_pct"] = out["p_tbv_now"].map(implied_rote)
    out["name"] = [NAME.get(t, t) for t in out.index]
    out["sub_industry"] = [SUB.get(t) for t in out.index]
    out.attrs["year"] = year
    return out


def tests(yt: pd.DataFrame, tr: pd.DataFrame, fund: pd.DataFrame) -> dict:
    car = yt[yt["sub_industry"] == "Carrier"]
    T_: dict = {"year": yt.attrs.get("year")}

    def w(s):
        s = s.dropna()
        return (float(s.median()), float(stats.wilcoxon(s).pvalue), int(len(s)), int((s < 0).sum())) if len(s) >= 6 else (np.nan, np.nan, len(s), 0)

    T_["lr"] = w(car["loss_ratio_chg_pp"])
    T_["ni"] = w(car["net_income_growth_pct"])
    T_["tbv"] = w(car["tbv_ytd_pct"])
    T_["aoci"] = w(car["aoci_chg_pct_tbv"])
    ptbv_ye = fund[fund["fiscal_year"] == T_["year"] - 1].set_index("ticker")["p_tbv"]
    T_["ptbv"] = w((car["p_tbv_now"] - ptbv_ye.reindex(car.index)))
    ret = yt["return_ytd_pct"]
    groups = [g.dropna().values for _, g in yt.groupby("sub_industry")["return_ytd_pct"] if g.notna().sum() >= 2]
    T_["sub_kw_p"] = float(stats.kruskal(*groups).pvalue) if len(groups) >= 3 else np.nan
    T_["ret_med"] = yt.groupby("sub_industry")["return_ytd_pct"].median().to_dict()
    spx = tr["^GSPC"].dropna()
    spx = spx[spx.index >= f"{T_['year'] - 1}-12-31"]
    T_["spx_ytd"] = float((spx.iloc[-1] / spx.iloc[0] - 1) * 100)
    T_["carrier_ret_med"] = float(car["return_ytd_pct"].median())
    y10 = db.fred("DGS10").dropna()
    if len(y10):
        ye = y10[y10.index <= f"{T_['year'] - 1}-12-31"]
        T_["y10_now"], T_["y10_ye"] = float(y10.iloc[-1]), float(ye.iloc[-1]) if len(ye) else np.nan
        T_["y10_chg_bp"] = (T_["y10_now"] - T_["y10_ye"]) * 100
    T_["leaders"] = car["return_ytd_pct"].sort_values(ascending=False).head(4)
    T_["laggards"] = car["return_ytd_pct"].sort_values().head(4)
    T_["lr_improvers"] = car["loss_ratio_chg_pp"].sort_values().head(4)
    T_["lr_deteriorators"] = car["loss_ratio_chg_pp"].sort_values(ascending=False).head(3)
    return T_


def event_study(tr: pd.DataFrame) -> pd.DataFrame:
    """Abnormal return (stock − equal-weight carrier basket) on the event day and the following 20 trading days."""
    tick = [c["ticker"] for c in CARRIERS]
    r = tr.ffill().pct_change(fill_method=None)  # forward-fill across market closures (e.g. 9 Jan 2025)
    r = r[r.index.dayofweek < 5]
    bask = r[[t for t in tick if t in r]].mean(axis=1)
    ev = db.table("events")
    ev["date"] = pd.to_datetime(ev["date"])
    rows = []
    for _, e in ev.iterrows():
        for t in [x for x in str(e["tickers"]).split(";") if x in r]:
            i = r.index.searchsorted(e["date"])
            if i + 21 >= len(r):
                continue
            ab = (r[t] - bask).iloc[i:i + 21]
            if pd.isna(ab.iloc[0]):
                continue
            grp = "Reserve / earnings charge" if "reserves" in e["category"] else e["category"]
            rows.append({"date": e["date"].date(), "group": grp, "ticker": t, "event": e["label"],
                         "day0_pct": ab.iloc[0] * 100, "next20_pct": ab.iloc[1:21].sum() * 100})
    return pd.DataFrame(rows)


def event_test(es: pd.DataFrame) -> dict:
    a = es[es["group"] == "Reserve / earnings charge"]
    c = es[es["group"] == "Catastrophe"]
    out = {"res_day0": a["day0_pct"].mean(), "res_next20": a["next20_pct"].mean(), "n_res": len(a),
           "cat_day0": c["day0_pct"].mean(), "cat_next20": c["next20_pct"].mean(), "n_cat": len(c)}
    out["p"] = float(stats.mannwhitneyu(a["next20_pct"], c["next20_pct"]).pvalue) if len(a) >= 3 and len(c) >= 3 else np.nan
    return out


def behaviours(fund: pd.DataFrame, decomp: pd.DataFrame, y0: int, y1: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top- vs bottom-quartile (by 5-yr TSR) behaviours, with Mann-Whitney tests."""
    q = decomp.set_index("ticker")["tsr_pct"].dropna()
    fy = fund[fund["fiscal_year"].between(y0 + 1, y1)]
    g = fy.groupby("ticker")
    beh = pd.DataFrame({
        "Average loss ratio %": g["loss_ratio_pct"].mean(),
        "Loss-ratio volatility (pp)": g["loss_ratio_pct"].std(),
        "Average RoTE %": g["rote_pct"].mean(),
        "Median TBV/share growth %": g["tbvps_growth_pct"].median(),
        "Premium growth % / yr": g["premium_growth_pct"].mean(),
        "Capital returned % of earnings": g.apply(lambda d: ((d["buybacks"].fillna(0) + d["dividends_paid"].fillna(0)).sum()
                                                             / d["net_income"].sum() * 100) if d["net_income"].sum() > 0 else np.nan),
    })
    ev = db.table("events")
    res = ev[ev["category"].str.contains("reserves")]
    beh["Public reserve / earnings shocks"] = [int(res["tickers"].str.contains(t).sum()) for t in beh.index]
    top = q[q >= q.quantile(.75)].index
    bot = q[q <= q.quantile(.25)].index
    rows = []
    for c in beh.columns:
        a, b = beh.loc[beh.index.intersection(top), c].dropna(), beh.loc[beh.index.intersection(bot), c].dropna()
        p = float(stats.mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else np.nan
        rows.append({"Behaviour": c, "Top quartile (median)": a.median(), "Bottom quartile (median)": b.median(), "p-value": p,
                     "Read": "significant" if p < .05 else "directional" if (pd.notna(p) and abs(a.median() - b.median()) > 0) else "n/a"})
    beh["quartile"] = ["Top" if t in top else "Bottom" if t in bot else "Middle" for t in beh.index]
    beh["tsr_5y"] = q.reindex(beh.index)
    return pd.DataFrame(rows), beh


def insights_current(yt: pd.DataFrame, T_: dict, et: dict) -> list[Insight]:
    yr = T_["year"]
    out: list[Insight] = []
    car = yt[yt["sub_industry"] == "Carrier"]

    def bars(col, title, unit):
        def f(col=col, title=title, unit=unit):
            s = car[col].dropna().sort_values()
            s.index = [NAME.get(t, t) for t in s.index]
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                                   marker_color=[T.NEG if (v > 0 if "loss" in col else v < 0) else T.POS for v in s.values],
                                   text=[f"{v:+.1f}{unit}" for v in s.values], textposition="outside", cliponaxis=False))
            fig.add_vline(x=s.median(), line_dash="dash", line_color=T.MAGENTA)
            fig.update_layout(title=title, height=max(300, 18 * len(s) + 90), showlegend=False, margin=dict(t=56, b=30))
            return fig
        return f

    med, p, n, neg = T_["lr"]
    if pd.notna(p) and p < .10:
        out.append(Insight(f"lr-{yr}", "Underwriting", f"{yr} year-to-date: loss ratios improved again across carriers",
                           f"{med:+.1f} pp", f"median 9M change · {neg} of {n} improved",
                           f"Wilcoxon on 10-Q year-to-date loss ratios vs the same period last year. Biggest improvers: "
                           + ", ".join(f"{NAME[t]} ({v:+.1f})" for t, v in T_["lr_improvers"].items()) + ". Deteriorating: "
                           + ", ".join(f"{NAME[t]} ({v:+.1f})" for t, v in T_["lr_deteriorators"].items()) + ".",
                           p, n, "SEC 10-Q XBRL", T.BLUE, bars("loss_ratio_chg_pp", f"{yr} YTD loss-ratio change vs prior year (pp)", " pp"),
                           car[["name", "loss_ratio_ytd", "loss_ratio_prior_ytd", "loss_ratio_chg_pp"]].round(1), year=yr,
                           so_what="Peak margins are being earned now, before softer rates earn through: bank them, don't spend them on price."))
    med, p, n, neg = T_["ni"]
    if pd.notna(p) and p < .10:
        out.append(Insight(f"ni-{yr}", "Profitability", f"{yr}: earnings still compounding: median net income {med:+.0f}%",
                           f"{med:+.0f}%", f"median YTD net income growth · {n - neg} of {n} up",
                           "10-Q year-to-date vs prior-year period (Wilcoxon).", p, n, "SEC 10-Q XBRL", T.BLUE,
                           bars("net_income_growth_pct", f"{yr} YTD net income growth (%)", "%"),
                           car[["name", "net_income_growth_pct", "rote_ytd_ann"]].round(1), year=yr,
                           so_what="This is the top of the earnings cycle: the decisions that matter are about 2027–28, not this quarter."))
    med, p, n, neg = T_["ptbv"]
    if pd.notna(p) and p < .10:
        out.append(Insight(f"ptbv-{yr}", "Valuation", f"{yr}: the market re-rated carriers {'up' if med > 0 else 'down'} despite flat book value",
                           f"{med:+.2f}x", "median P/TBV change since year-end",
                           f"Tangible book is roughly flat YTD ({T_['tbv'][0]:+.1f}% median) as higher yields push AOCI back down "
                           f"({T_['aoci'][0]:+.1f} pp of TBV); the multiple did the work.",
                           p, n, "SEC 10-Q XBRL · Yahoo Finance", T.NAVY, None,
                           car[["name", "p_tbv_now", "implied_rote_pct", "tbv_ytd_pct"]].round(2), year=yr,
                           so_what="Investors are paying for durability while book value stalls: they are underwriting your 2027 margins today."))
    med, p, n, neg = T_["aoci"]
    if pd.notna(p) and p < .10:
        out.append(Insight(f"aoci-{yr}", "Rates & book value", f"{yr}: rising yields are eroding book value again, a smaller echo of 2022",
                           f"{med:+.1f} pp", f"median AOCI change as % of TBV · {neg} of {n} down",
                           f"10-year Treasury {T_.get('y10_chg_bp', np.nan):+.0f} bp since year-end (FRED); unrealized losses widened (Wilcoxon).",
                           p, n, "SEC 10-Q XBRL · FRED", T.MAGENTA, bars("aoci_chg_pct_tbv", f"{yr} YTD AOCI change (% of TBV)", " pp"),
                           car[["name", "aoci_chg_pct_tbv", "tbv_ytd_pct"]].round(1), year=yr,
                           so_what="Earnings are strong but book value isn't growing: report ex-AOCI and keep duration matched."))
    if pd.notna(et.get("p")) and et["p"] < .10:
        out.append(Insight("event-study", "Market signal", "Reserve charges are priced on day one; catastrophe exposure bleeds for weeks",
                           f"{et['res_day0']:+.1f}% vs {et['cat_next20']:+.1f}%",
                           "reserve-shock day-0 vs cat next-20-day abnormal return",
                           f"Event study on {et['n_res']} reserve/earnings and {et['n_cat']} catastrophe stock-events: reserve shocks average "
                           f"{et['res_day0']:+.1f}% on the day then {et['res_next20']:+.1f}% over 20 days; cat exposures {et['cat_day0']:+.1f}% "
                           f"then {et['cat_next20']:+.1f}% (Mann-Whitney on the 20-day drift).",
                           et["p"], et["n_res"] + et["n_cat"], "Event log · Yahoo Finance", T.RED, None, None, year=None,
                           so_what="Credibility shocks are instant; cat shocks compound as estimates develop: manage disclosure accordingly."))
    return out
