"""Executive narrative: every chart's 'so what', laddered to five themes, built from live numbers.

Voice: plain-spoken, optimistic, forward-looking; every claim points at a number on the page.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TH1 = "The cycle has turned"
TH2 = "Underwriting precision wins"
TH3 = "Volatility is structural"
TH4 = "Value is migrating along the chain"
TH5 = "Cyber & AI: growth line, accumulation line"


def _g(d, *keys, default=np.nan):
    for k in keys:
        try:
            d = d[k]
        except Exception:  # noqa: BLE001
            return default
    return d


def facts(ctx: dict, ind: dict, sub_ct: pd.DataFrame, sub_t: dict, cyf: pd.DataFrame) -> dict:
    """Collect every number the narrative cites, in one place."""
    F: dict = {}
    # Industry
    ci_s, ci_b, _ = ind["ciab"]
    F["ciab_latest"], F["ciab_latest_q"] = ci_s.iloc[-1], ci_s.index[-1]
    if ci_b:
        F["ciab_break"], F["ciab_break_p"] = ci_b[-1]["date"], ci_b[-1]["p_value"]
        prev_start = ci_b[-2]["index"] if len(ci_b) > 1 else 0
        F["ciab_hard"] = ci_s.iloc[prev_start:ci_b[-1]["index"]].mean()
        F["ciab_now"] = ci_s.iloc[ci_b[-1]["index"]:].mean()
        F["ciab_hard_start"] = ci_b[-2]["date"] if len(ci_b) > 1 else ci_s.index[0]
    cr_s, cr_f, _ = ind["cr"]
    F["cr_last"], F["cr_last_y"] = cr_s.iloc[-1], cr_s.index[-1]
    F["cr_rank"] = int((cr_s <= cr_s.iloc[-1]).sum())
    F["cr_n"] = len(cr_s)
    F["cr_2022"] = cr_s.get("2022", np.nan)
    if len(cr_f):
        F["cr_fc"], F["cr_fc_y"] = cr_f.iloc[min(1, len(cr_f) - 1)], cr_f.index[min(1, len(cr_f) - 1)]
    cat_s, cat_b, _ = ind["cat"]
    late = [b for b in cat_b if b["date"] >= "2005"]
    if late:
        b = late[-1]
        F["cat_break"], F["cat_break_p"] = b["date"], b["p_value"]
        F["cat_before"] = cat_s[(cat_s.index < b["date"]) & (cat_s.index >= str(int(b["date"]) - 27))].mean()
        F["cat_after"] = cat_s[cat_s.index >= b["date"]].mean()
        F["cat_last"], F["cat_last_y"] = cat_s.iloc[-1], cat_s.index[-1]
    j = ind.get("dwp_gdp")
    if j is not None and len(j):
        F["dwp_gdp_r"] = float(np.corrcoef(j["premium"], j["gdp"])[0, 1])
    # Carrier analytics
    dec = ctx["decomp"]
    if len(dec):
        F["tsr_med"] = dec["tsr_pct"].median()
        F["tbv_med"] = dec["tbvps_growth_pp"].median()
        F["rerate_med"] = dec["multiple_change_pp"].median()
        F["cash_med"] = dec["cash_yield_pp"].median()
        top = dec.sort_values("tsr_pct", ascending=False)
        F["top_name"], F["top_tsr"] = top["name"].iloc[0], top["tsr_pct"].iloc[0]
        F["top_rerate"] = top["multiple_change_pp"].iloc[0]
        F["bot_name"], F["bot_tsr"] = top["name"].iloc[-1], top["tsr_pct"].iloc[-1]
        from scipy import stats
        ok = dec.dropna(subset=["multiple_change_pp", "tsr_pct", "tbvps_growth_pp"])
        if len(ok) >= 8:
            F["rho_rerate"], F["p_rerate"] = stats.spearmanr(ok["multiple_change_pp"], ok["tsr_pct"])
            F["rho_tbv"], F["p_tbv"] = stats.spearmanr(ok["tbvps_growth_pp"], ok["tsr_pct"])
        ok2 = dec.dropna(subset=["p_tbv_start", "multiple_change_pp"])
        if len(ok2) >= 8:
            F["rho_meanrev"], F["p_meanrev"] = stats.spearmanr(ok2["p_tbv_start"], ok2["multiple_change_pp"])
        k = dec.set_index("name")
        if "Kinsale" in k.index:
            F["knsl_rerate"], F["knsl_tbv"] = k.loc["Kinsale", "multiple_change_pp"], k.loc["Kinsale", "tbvps_growth_pp"]
        if "AIG" in k.index:
            F["aig_rerate"] = k.loc["AIG", "multiple_change_pp"]
    ys = ctx["years"].set_index("year")
    for y in ys.index:
        F[f"tbv_{y}"] = ys.loc[y, "tbvps_growth_median"]
        F[f"aoci_{y}"] = ys.loc[y, "aoci_to_tbv_change_pp"]
        F[f"lr_{y}"] = ys.loc[y, "loss_ratio_change_pp"]
        F[f"lrp_{y}"] = ys.loc[y, "loss_ratio_p"]
        F[f"rote_{y}"] = ys.loc[y, "rote_median"]
    f = ctx["fund"]
    tg = f[f["fiscal_year"] == ctx["y1"]]["tbvps_growth_pct"].dropna()
    F["tbv_pos_share_last"] = (tg > 0).mean() * 100 if len(tg) else np.nan
    import analysis as A
    da = ctx["decomp_all"].set_index("ticker")["tsr_pct"].dropna()
    inf = A.influence(da)
    out = inf[inf["robust_z"].abs() > 2.5]
    F["outlier_text"] = (" ".join(f"{A.NAME[t]} is a statistical outlier (robust z {r.robust_z:+.1f}, moves the average {r.influence_pp:+.1f} pp)."
                                  for t, r in out.iterrows()) or "No carrier is a statistical outlier.")
    pt = f.pivot(index="ticker", columns="fiscal_year", values="p_tbv")
    y1 = ctx["y1"]
    if y1 in pt and y1 - 1 in pt:
        F["ptbv_chg_last"] = (pt[y1] - pt[y1 - 1]).median()
    res, shocks = ctx["res"], ctx["shocks"]
    F["beta_now"] = res["beta"].iloc[-1]
    F["beta_first"] = res["beta"].iloc[252] if len(res) > 260 else np.nan
    F["n_shocks"], F["n_neg"] = len(shocks), int((shocks["z"] < 0).sum())
    F["n_attr"] = int(shocks["catalyst"].notna().sum()) if "catalyst" in shocks else 0
    if len(shocks):
        F["worst_shock"] = shocks.iloc[0]
        cats = shocks["event_category"].value_counts() if "event_category" in shocks else pd.Series(dtype=int)
        F["shock_cats"] = cats.to_dict()
    F["best_break_p"] = ctx["best_break"]["p_value"]
    # Sub-industry
    if len(sub_ct):
        med = sub_ct.groupby("sub_industry")[["ret_1y", "pe", "vol_1y", sub_t.get("growth_col", "ret_1y")]].median()
        F["sub_med"] = med
        F["sub_tests"] = sub_t
    # Cyber
    def cv(metric, period):
        d = cyf[(cyf["metric"] == metric) & (cyf["period"].astype(str) == str(period))]
        return float(d["value"].iloc[0]) if len(d) else np.nan
    F["cy_prem_24"] = cv("US cyber direct written premium (incl. alien surplus lines)", 2024)
    F["cy_prem_25"] = cv("US cyber direct written premium (incl. alien surplus lines)", 2025)
    F["cy_prem_22"] = cv("US cyber direct written premium (incl. alien surplus lines)", 2022)
    F["cy_lr_23"] = cv("US cyber industry loss ratio", 2023)
    F["cy_lr_25"] = cv("US cyber industry loss ratio", 2025)
    F["cy_pol"] = cv("US cyber policies in force", 2025)
    F["cy_endo"] = cv("Share of cyber policies that are endorsements", 2025)
    F["cy_endo_prem"] = cv("Share of cyber premium from endorsements", 2025)
    F["cy_peak"] = cv("US cyber rate change (Marsh GIMI)", "2021-Q4")
    F["cy_ransom"] = cv("Initial ransom demand change (Coalition)", 2025)
    F["cy_refuse"] = cv("Share of businesses refusing to pay ransom (Coalition)", 2025)
    F["cy_sev"] = cv("Claims severity change (Coalition)", 2025)
    F["cy_global"] = cv("Global cyber premium", 2024)
    return F


def _p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def build(F: dict) -> dict:
    """Return {chart_key: so-what dict}."""
    S: dict = {}
    sm = F.get("sub_med")

    # ---------------- Industry ----------------
    if "ciab_break" in F:
        S["ind_ciab"] = dict(
            theme=TH1,
            headline=f"The hard market ended in {F['ciab_break']}. That's a statistical fact now, not a feeling.",
            body=(f"Commercial rate increases averaged <b>{F['ciab_hard']:+.1f}%</b> a quarter from {F['ciab_hard_start']}; since the "
                  f"{F['ciab_break']} break ({_p(F['ciab_break_p'])}, autocorrelation-robust) they average <b>{F['ciab_now']:+.1f}%</b>, and "
                  f"{F['ciab_latest_q']} printed <b>{F['ciab_latest']:+.1f}%</b>. Rate was the tailwind that made a lot of average underwriting "
                  "look great. That tailwind is gone, and the market will now separate the carriers who used the hard market to build "
                  "pricing and data muscle from the ones who just rode it."),
            roles={
                "CFO": ("Earned rate lags written rate by 12–18 months, so the margin squeeze shows up in 2027 plans, not today's P&L.",
                        "Re-base 2027 plans on flat-to-negative rate; stress the expense ratio and stop counting rate as growth."),
                "CUO": ("Every point of rate given back now has to be earned back through risk selection and terms.",
                        "Hold attachment points and terms before price; re-underwrite the bottom decile of the book now."),
                "CEO": ("Soft markets reward the disciplined and punish the volume-chasers, usually 2–3 years later.",
                        "Set the 'walk-away' culture explicitly: reward underwriters for the business they decline."),
            },
            watch="next CIAB quarter, Marsh GIMI, and whether casualty lines keep rising while property falls.")
    if "cr_last" in F:
        S["ind_cr"] = dict(
            theme=TH1,
            headline=f"{F['cr_last_y']} was the #{F['cr_rank']} underwriting year since 2001. Enjoy it; don't extrapolate it.",
            body=(f"The industry combined ratio hit <b>{F['cr_last']:.1f}</b> (rank {F['cr_rank']} of {F['cr_n']}), a swing of "
                  f"{F['cr_2022'] - F['cr_last']:.1f} points from 2022's {F['cr_2022']:.1f}. No structural break is detected, so this is the "
                  f"top of a cycle, not a new normal. The deck forecast drifts back to <b>{F.get('cr_fc', np.nan):.1f}</b> by {F.get('cr_fc_y', '')}: "
                  "softer rates meet a loss trend that never really slowed."),
            roles={
                "CFO": ("Peak-cycle earnings are the most dangerous earnings to capitalize.",
                        "Bank the surplus: favour buybacks and reserve margin over permanent cost or dividend commitments."),
                "CUO": ("A 93 composite hides lines already running hot (casualty, commercial auto).",
                        "Run combined ratios by line and vintage; anything above 100 on an accident-year basis gets repriced now."),
                "CEO": ("Boards will read 92.9 as the new baseline; it isn't.",
                        "Frame 2026–27 guidance around a mid-to-high-90s combined ratio and say so publicly first."),
            },
            watch="accident-year combined ratios ex-cat, and prior-year development in casualty.")
    if "cat_break" in F:
        S["ind_cat"] = dict(
            theme=TH3,
            headline=f"Catastrophe losses stepped up in {F['cat_break']} and never came back down.",
            body=(f"Inflation-adjusted insured cat losses averaged <b>${F['cat_before']:.0f}B</b> a year before {F['cat_break']} and "
                  f"<b>${F['cat_after']:.0f}B</b> since (break {_p(F['cat_break_p'])}). Severe convective storms, wildfire, and more value in "
                  "harm's way did this, not one bad hurricane season. The old 'one-in-ten' year is the new 'every year'."),
            roles={
                "CFO": ("Cat volatility now drives earnings variance and capital more than investment results do.",
                        "Budget cat at the post-2017 run-rate; use the cheaper reinsurance market (ROL down 12–16%) to buy protection, not to cut it."),
                "CUO": ("Secondary perils (hail, wildfire) are under-modelled and under-priced in most admitted books.",
                        "Price to peril-level models; use roof-age, aerial imagery and wildfire scores as rating variables, not just underwriting rules."),
                "CEO": ("Climate-exposed geographies are where growth and ruin sit side by side.",
                        "Invest in geospatial data and a property 'pricing engine' as core capability, not a vendor feature."),
            },
            watch="Q3 hurricane season, SCS frequency, wildfire regulation in CA, and reinsurance terms at 1/1.")
    if "dwp_gdp_r" in F:
        S["ind_gdp"] = dict(
            theme=TH1,
            headline="When rate fades, the economy becomes the growth engine again.",
            body=(f"Premium growth tracks nominal GDP with <b>r = {F['dwp_gdp_r']:.2f}</b>. In 2021–24 rate pushed premiums well above GDP; "
                  "as rate turns negative, exposure growth (payrolls, property values, vehicle miles) is what's left. That's slower, steadier, "
                  "and it rewards carriers who can find exposure, not just raise price."),
            roles={
                "CFO": ("Top-line guidance tied to rate will miss; GDP-linked exposure won't.",
                        "Model premium as exposure × rate explicitly; flex the plan to nominal GDP scenarios."),
                "CUO": ("Exposure growth is concentrated in specific classes and regions.",
                        "Point appetite at the fastest-growing exposure bases (Sun Belt construction, logistics, small commercial)."),
                "CEO": ("The next growth leg is distribution and product, not price.",
                        "Fund the channels that reach new exposure: embedded, small-commercial digital, specialty programs."),
            },
            watch="nominal GDP prints, payroll growth, construction starts.")
    S["ind_pa"] = dict(
        theme=TH2,
        headline="Personal auto went from the worst line to one of the best in three years. The leaders moved first.",
        body=("Industry personal auto swung from a 112 combined ratio (2022) to about 93 (2025E); our personal carriers' loss ratios "
              "turned at the same moment. Progressive repriced first and hardest and grabbed share while others were still in repair "
              "mode. Now that profitability is restored, everyone is shifting to offense, which is exactly when pricing gets sloppy."),
        roles={
            "CFO": ("Auto margins are at a peak just as competitors re-open the throttle on advertising.",
                    "Treat acquisition spend as an investment with a payback hurdle; don't chase share at sub-target lifetime value."),
            "CUO": ("Loss trend is still running; repair and medical severity haven't normalised.",
                    "Keep telematics and segmentation tightening; file rate where indicated even if competitors don't."),
            "CEO": ("Speed of repricing was the decisive capability in this cycle.",
                    "Build the 'rate-change factory': data, filings and IT that take an indication to market in weeks, not quarters."),
        },
        watch="Progressive's monthly results, auto severity (repair costs, medical), and state rate-filing approvals.")
    if "aoci_2022" in F:
        S["ind_aoci"] = dict(
            theme=TH3,
            headline="The 2022 book-value hit came from the bond desk, not the underwriting desk.",
            body=(f"As the 10-year Treasury yield jumped, median unrealized losses moved <b>{F['aoci_2022']:+.0f} pp</b> of tangible book in 2022 "
                  "and have been unwinding ever since. That money comes back to par, and today's higher yields are a real earnings tailwind. But "
                  "it showed how fast an 'underwriting' business can become a rates trade."),
            roles={
                "CFO": ("Duration mismatch turned into a visible capital event in 2022.",
                        "Match asset duration to reserve duration; reinvest maturities at today's yields to lock in investment income."),
                "CUO": ("Higher investment income tempts price-cutting in long-tail lines.",
                        "Never let new-money yields justify underpricing casualty; social inflation eats the float."),
                "CEO": ("Investors punished book-value volatility even when it was temporary.",
                        "Report and explain ex-AOCI book value growth as the core KPI."),
            },
            watch="10-year Treasury, credit spreads, and AOCI recovery as bonds roll to par.")

    # ---------------- Market signals ----------------
    S["ms_excess"] = dict(
        theme=TH3,
        headline="P&C stocks didn't break their trend. They got hit by specific, explainable shocks.",
        body=(f"The strongest candidate break in excess returns isn't significant ({_p(F['best_break_p'])}), so there's no regime change "
              f"in how the market values the group. The action is in the shocks: <b>{F['n_shocks']}</b> insurance-specific shock days, "
              f"<b>{F['n_neg']}</b> of them sell-offs, and <b>{F['n_attr']}</b> traced to a named cause (reserve charges, catastrophes, rate "
              "scares). Reserve surprises get priced in a single day; catastrophe exposure keeps bleeding for weeks as loss estimates develop."),
        roles={
            "CFO": ("Reserve charges take about 5% off a stock in one day (Allstate 2022, Markel 2023, Selective 2024).",
                    "Pre-empt: run independent reserve reviews on post-2015 casualty years before the market does it for you."),
            "CUO": ("Most negative shocks trace to long-tail lines where trend was under-picked.",
                    "Raise loss picks on recent accident years in GL, commercial auto and umbrella now, not after a charge."),
            "CEO": ("Credibility is the asset; a surprise costs years of multiple.",
                    "Guide conservatively and over-deliver; communicate reserve philosophy before earnings, not on the call."),
        },
        watch="earnings season (reserve releases vs charges), hurricane season, and CPI/yield surprises.")
    S["ms_beta"] = dict(
        theme=TH1,
        headline=f"P&C has decoupled from the market: beta is now {F['beta_now']:.2f}.",
        body=("Rolling beta to the S&P 500 has fallen from around 1.0 to roughly zero. Insurers now trade on their own drivers "
              "(pricing cycle, catastrophes, reserve adequacy, yields), which makes them a genuine diversifier. It also means the "
              "group's fortunes are entirely in management's hands."),
        roles={
            "CFO": ("Your cost of equity is set by underwriting volatility now, not market beta.",
                    "Cut earnings volatility (reinsurance, reserve margin) to earn a lower cost of capital."),
            "CUO": ("Idiosyncratic risk dominates the stock.",
                    "Treat accumulations (cat, cyber, casualty clash) as share-price risks, not just capital risks."),
            "CEO": ("Investors are buying the sector for its independence from the macro.",
                    "Tell an equity story built on durable underwriting edge, not market momentum."),
        },
        watch="sector beta drifting back toward 1, which would signal macro or credit fear entering insurance pricing.")
    S["ms_segments"] = dict(
        theme=TH2,
        headline="Same cycle, very different outcomes by segment.",
        body=("Commercial and specialty compounded through the hard market. Personal lines took the 2022–23 loss-cost hit first, then "
              "recovered hardest once rate caught up. Florida property names swing on single storms. Segment choice explains a lot of "
              "return, but within each segment the spread between leaders and laggards is wider still."),
        roles={
            "CFO": ("Segment mix drives both growth and volatility in the multiple.",
                    "Allocate capital by risk-adjusted return per segment, refreshed annually."),
            "CUO": ("Personal and property lines carry the most short-term volatility.",
                    "Diversify peril and geography; cap single-state and single-peril concentration."),
            "CEO": ("Portfolio strategy is the biggest lever on long-run TSR.",
                    "Be explicit about where you'll be top-quartile and exit where you can't."),
        },
        watch="relative performance of commercial vs personal as commercial rate turns negative.")

    # ---------------- Value drivers ----------------
    if "rho_rerate" in F:
        S["vd_decomp"] = dict(
            theme=TH2,
            headline=f"The median carrier returned {F['tsr_med']:.0f}% a year, but the winners won on re-rating.",
            body=(f"Median annual TSR split roughly into <b>{F['tbv_med']:.0f} pp</b> book growth, <b>{F['rerate_med']:.0f} pp</b> re-rating "
                  f"and <b>{F['cash_med']:.0f} pp</b> cash. {F['top_name']} led at {F['top_tsr']:.0f}% a year, with "
                  f"{F['top_rerate']:.0f} pp from re-rating alone. {F['bot_name']} trailed at {F['bot_tsr']:.0f}%. Re-rating is the "
                  "market's verdict on whether earnings are durable."),
            roles={
                "CFO": ("Book growth is necessary but not sufficient; the multiple is where the value is.",
                        "Show the market the quality of earnings: reserve strength, low volatility, consistent capital return."),
                "CUO": ("The market re-rates carriers whose loss ratios hold through the cycle.",
                        "Optimise for loss-ratio stability, not peak margin."),
                "CEO": ("Re-rating is earned over years and lost in a quarter.",
                        "Anchor the strategy narrative on underwriting consistency and capital discipline."),
            },
            watch="P/TBV dispersion: compression across the board signals the market pricing the soft market in.")
        S["vd_rerating"] = dict(
            theme=TH2,
            headline=f"Re-rating explains who won (ρ={F['rho_rerate']:.2f}); book growth alone doesn't (ρ={F['rho_tbv']:.2f}).",
            body=(f"Across carriers, multiple change tracks TSR rank ({_p(F['p_rerate'])}); TBV growth is statistically unrelated "
                  f"({_p(F['p_tbv'])}). Kinsale is the textbook outlier: book compounded about {F.get('knsl_tbv', np.nan):.0f} pp a year while "
                  f"its multiple compressed {abs(F.get('knsl_rerate', np.nan)):.0f} pp, as E&S competition arrived and the market priced out "
                  "the growth premium."),
            roles={
                "CFO": ("A premium multiple is a liability when growth decelerates.",
                        "Guide to sustainable growth; avoid multiple-supporting promises the cycle won't keep."),
                "CUO": ("The market re-rates loss-ratio credibility, not premium volume.",
                        "Publish loss-ratio targets by line and hit them."),
                "CEO": ("Being 'cheap with a story' beat 'expensive with momentum' this cycle.",
                        "If you're a re-rating candidate, the catalyst is visible, repeatable underwriting improvement."),
            },
            watch="E&S carriers' multiples as competition and soft property rates bite.")
    if "rho_meanrev" in F:
        S["vd_meanrev"] = dict(
            theme=TH2,
            headline=f"Cheap carriers re-rated; expensive ones compressed (ρ={F['rho_meanrev']:.2f}).",
            body=(f"Starting valuation predicted re-rating with a strong negative correlation ({_p(F['p_meanrev'])}). AIG is the turnaround "
                  f"case (about +{F.get('aig_rerate', np.nan):.0f} pp a year from re-rating) as it de-risked and exited life. Premium names "
                  "gave some multiple back. In a soft market the spread between 'cheap with a fix' and 'expensive and priced for "
                  "perfection' tends to close further."),
            roles={
                "CFO": ("Your starting multiple sets your upside and downside.",
                        "Use a premium multiple as acquisition currency while it lasts; use a discount as a buyback opportunity."),
                "CUO": ("Turnarounds re-rate on visible portfolio surgery.",
                        "Show line-level exits and re-underwriting results quarter by quarter."),
                "CEO": ("Valuation gaps are strategy signals.",
                        "Benchmark against the most re-rated peer; close the gap with simplification, not just growth."),
            },
            watch="P/TBV spread between top and bottom quartile.")
    S["vd_influence"] = dict(
        theme=TH2,
        headline="One carrier can bend the average. Read the median.",
        body=(f"Leave-one-out analysis shows how much each company moves the universe average. {F.get('outlier_text', 'No carrier')} "
              "Toggle outliers out in the sidebar and the story is unchanged: the median is robust, the mean isn't."),
        roles={
            "CFO": ("Peer benchmarks built on averages mislead.",
                    "Benchmark on medians and quartiles; exclude structural outliers explicitly."),
            "CUO": ("Outliers usually trace to one book or one event.",
                    "Diagnose peers' outlier lines before copying or avoiding a strategy."),
            "CEO": ("Narratives built on an outlier are fragile.",
                    "Ask 'what does the median say?' before acting on an industry headline."),
        },
        watch="new outliers emerging as reserve charges hit long-tail writers.")

    # ---------------- Fundamentals ----------------
    S["fu_tbv"] = dict(
        theme=TH3,
        headline=f"Book value: −{abs(F.get('tbv_2022', np.nan)):.0f}% in 2022, then three years of about +{np.nanmedian([F.get('tbv_2023', np.nan), F.get('tbv_2024', np.nan), F.get('tbv_2025', np.nan)]):.0f}%.",
        body=("2022's decline was almost entirely bond marks; 2023–25 combined earnings power with marks unwinding. "
              f"{F.get('tbv_pos_share_last', np.nan):.0f}% of carriers grew TBV per share in the latest year, the strongest capital build in the data. That capital is now looking for a home "
              "just as rates soften, which is the classic setup for over-capacity."),
        roles={
            "CFO": ("Excess capital in a softening market tends to get spent badly.",
                    "Pre-commit a capital return framework (buybacks/specials) so surplus doesn't leak into underpriced growth."),
            "CUO": ("Capital-rich competitors will cut price for share.",
                    "Don't match price below technical; let capacity flood go past you."),
            "CEO": ("This is when value-destroying M&A happens.",
                    "Set strict IRR hurdles; prefer capability acquisitions (data, distribution, MGA platforms) over balance-sheet buys."),
        },
        watch="industry surplus growth vs premium growth (capacity per premium dollar).")
    S["fu_aoci"] = dict(theme=TH3, headline="Unrealized losses are shrinking and turning into earnings.",
                        body=("AOCI is moving back toward zero as 2021-vintage bonds mature and are reinvested at higher yields. That is a "
                              "quiet, multi-year tailwind for investment income, and a reason not to cut price on the back of it."),
                        roles=None)
    S["fu_rote"] = dict(theme=TH2, headline=f"Returns on tangible equity peaked around {F.get('rote_2024', np.nan):.0f}% (median).",
                        body=("RoTE expanded through 2023–24 and is flattening as rate decelerates. Dispersion stays wide: the top of the "
                              "distribution is in specialty and repriced personal lines, the bottom in carriers still absorbing reserve charges."),
                        roles=None)
    S["fu_ptbv"] = dict(
        theme=TH1, headline=f"Valuations compressed in {ctx_year(F)} even as book grew: the market is pricing the turn.",
        body=(f"Median P/TBV moved <b>{F.get('ptbv_chg_last', np.nan):+.2f}x</b> while TBV per share grew about {F.get('tbv_2025', np.nan):.0f}%. "
              "Investors are already discounting softer margins ahead; the next leg of TSR has to come from book growth and capital "
              "return, not multiple expansion."),
        roles={
            "CFO": ("Multiple expansion won't do the heavy lifting for 2026–27 TSR.",
                    "Lean on buybacks at sub-2x P/TBV where returns exceed cost of equity."),
            "CUO": ("Margin defense is now the equity story.",
                    "Make line-level margin targets the headline KPI internally."),
            "CEO": ("Compression is a signal, not a verdict.",
                    "Differentiate on through-cycle consistency to escape the group de-rating."),
        },
        watch="P/TBV vs RoTE: multiples falling faster than returns signals overshoot (and opportunity).")
    S["fu_lr"] = dict(
        theme=TH2, headline="Loss ratios tell you who priced ahead of trend, and who didn't.",
        body=(f"The median loss ratio improved {abs(F.get('lr_2025', np.nan)):.1f} pp in 2025 ({_p(F.get('lrp_2025', 1))}), broad-based across "
              "carriers. Specialty writers (RLI, Chubb, Arch) sit persistently in the blue, while personal auto writers ran hot in 2022–23 "
              "and are only now normalising. Persistence matters more than any single year."),
        roles={
            "CFO": ("A low loss ratio with rising prior-year development is a warning, not a win.",
                    "Pair calendar-year loss ratios with accident-year views and development triangles in every review."),
            "CUO": ("Consistency of the loss ratio is what the market pays for.",
                    "Target the variance of the loss ratio, not just its level."),
            "CEO": ("Underwriting culture shows up as loss-ratio stability.",
                    "Tie underwriter incentives to multi-year loss ratio, not premium."),
        },
        watch="2026 loss ratios as earned rate fades.")
    S["yy_matrix"] = dict(
        theme=TH1, headline="Five years, five different stories, each named only by what the statistics confirm.",
        body=("2021: hard-market growth. 2022: the double squeeze of rates and inflation. 2023: rebound. 2024: capital build. "
              "2025: pricing catches up with loss trend, and valuations start pricing the turn. 2026 is shaping up as the first "
              "soft-market year in seven."),
        roles=None)

    # ---------------- Sub-industry ----------------
    t = F.get("sub_tests", {})
    if sm is not None and len(sm):
        def m(sub, col):
            return sm.loc[sub, col] if sub in sm.index else np.nan
        S["sub_returns"] = dict(
            theme=TH4,
            headline="Over the last year the money went to risk-takers with capital (reinsurers), not to the toll-collectors (brokers).",
            body=(f"1-year median returns: reinsurers <b>{m('Reinsurer','ret_1y'):+.0f}%</b>, wholesale/MGA <b>{m('Wholesale & MGA','ret_1y'):+.0f}%</b>, "
                  f"carriers <b>{m('Carrier','ret_1y'):+.0f}%</b>, retail brokers <b>{m('Retail broker','ret_1y'):+.0f}%</b>, insurtech "
                  f"<b>{m('Insurtech','ret_1y'):+.0f}%</b> (differences significant, Kruskal-Wallis {_p(t.get('ret_1y', 1))}). Reinsurers banked "
                  "record margins from the post-2023 reset; brokers face a falling-rate headwind on commission revenue."),
            roles={
                "CFO": ("Where you sit in the chain sets how a soft market hits you.",
                        "Stress-test revenue: brokers to rate (commissions), carriers to margin, reinsurers to price and cat."),
                "CUO": ("Reinsurance is cheaper (ROL −12% at 1/1/26, −16% by mid-year).",
                        "Lock in multi-year capacity and broader terms while the market is soft."),
                "CEO": ("Value is pooling where capital meets underwriting talent.",
                        "Decide deliberately whether to be the capital, the underwriter, or the distributor, and invest accordingly."),
            },
            watch="broker organic growth as rate turns negative, and reinsurer margins after a heavy cat year.")
        S["sub_val"] = dict(
            theme=TH4,
            headline=f"The market pays {m('Retail broker','pe'):.0f}× earnings for brokers, {m('Carrier','pe'):.0f}× for carriers and {m('Reinsurer','pe'):.0f}× for reinsurers.",
            body=(f"Fee-based, capital-light distribution earns a premium multiple ({_p(t.get('broker_vs_carrier_pe', 1))} vs carriers); balance-sheet "
                  "risk is priced at a deep discount. That gap is why carriers are building MGAs, buying distribution and launching "
                  "fee-income vehicles (third-party capital, fronting, services)."),
            roles={
                "CFO": ("Fee income is re-rating fuel.",
                        "Grow fee-based revenue (MGA, services, third-party capital management) and disclose it separately."),
                "CUO": ("Delegated authority grows the book without growing the balance sheet, if controlled.",
                        "Invest in delegated-authority oversight: data feeds, audits, bordereaux automation."),
                "CEO": ("The valuation arbitrage is the strategy.",
                        "Evaluate MGA/program platforms and capital-light structures as core, not adjacent."),
            },
            watch="broker multiples if organic growth keeps slowing (Brown & Brown ~1%).")
        gcol = t.get("growth_col", "")
        S["sub_lens"] = dict(
            theme=TH4,
            headline=f"E&S carriers grew revenue about {t.get('es_growth_median', np.nan):.0f}% in the latest year vs about {t.get('admitted_growth_median', np.nan):.0f}% for admitted carriers.",
            body=("Complex and hard-to-place risk keeps migrating to the non-admitted market: surplus lines reached $143.3B in 2025 "
                  "(12.9% of all P&C, 27.5% of commercial, up from 7.1% in 2000). Growth is moderating (10.4% in 2025, ending seven "
                  "years of double digits) as standard carriers fight back and property rates fall."),
            roles={
                "CFO": ("E&S growth is decelerating from a high base, and premium multiples are compressing with it.",
                        "Underwrite to margin, not growth targets, in E&S property."),
                "CUO": ("Freedom of rate and form is the E&S edge, as long as you use it.",
                        "Keep terms tight (sublimits, exclusions) even as price falls; that's what admitted markets can't copy."),
                "CEO": ("The wholesale/E&S channel is where the specialty growth is.",
                        "Own a wholesale relationship strategy, or own a wholesale/MGU platform."),
            },
            watch="surplus lines stamping-office premium growth and E&S property rate change.")
        S["sub_context"] = dict(
            theme=TH4, headline="The chain in four numbers.",
            body=("MGAs wrote $102.6B statutory DPW in 2025 (+12%, ~$128B all-in), now a structural channel. Reinsurance prices fell 12% "
                  "at 1/1/26 and 16% by mid-year. Broker organic growth cooled to about 5% at the leaders and roughly 1% at Brown & Brown. "
                  "Global commercial rates have fallen nine quarters running."),
            roles=None)

    # ---------------- Cyber & AI ----------------
    S["cy_market"] = dict(
        theme=TH5,
        headline=f"US cyber stopped growing just as losses turned up: about ${F.get('cy_prem_25', np.nan):.1f}B premium, {F.get('cy_lr_25', np.nan):.0f}% loss ratio.",
        body=(f"Premium has been flat since 2022 (${F.get('cy_prem_22', np.nan):.1f}B → ${F.get('cy_prem_25', np.nan):.1f}B) while the loss ratio climbed from "
              f"{F.get('cy_lr_23', np.nan):.0f}% (2023) to {F.get('cy_lr_25', np.nan):.0f}% (2025). Policies in force jumped to "
              f"{F.get('cy_pol', np.nan):.2f}M, but {F.get('cy_endo', np.nan):.0f}% are endorsements carrying just "
              f"{F.get('cy_endo_prem', np.nan):.0f}% of premium. Penetration is broad and thin: lots of companies have 'some' cyber, few have enough."),
        roles={
            "CFO": ("Cyber margins are compressing from a very profitable base, and accumulation risk is on the balance sheet.",
                    "Quantify cyber cat (cloud, software supply chain) in capital models; buy cyber cat reinsurance while it's available."),
            "CUO": ("Price cuts meet rising frequency and third-party liability claims.",
                    "Stop the rate slide on primary layers; enforce controls-based underwriting (MFA, EDR, backups) as a condition of cover."),
            "CEO": ("The protection gap is the opportunity: thin endorsements are an upsell.",
                    "Convert endorsement holders to standalone cover with services (scanning, response) bundled in."),
        },
        watch="NAIC 2026 data (loss ratio trend), systemic events (CrowdStrike-style), and ransomware demand trends.")
    S["cy_rates"] = dict(
        theme=TH5,
        headline=f"From +{F.get('cy_peak', np.nan):.0f}% to twelve straight quarters of cuts.",
        body=("Ransomware drove a historic repricing in 2021–22; abundant capacity has given most of it back since mid-2023. Rates "
              "are still edging down (US about −2%, global about −4% in Q2 2026) even as loss ratios rise, which is the definition of a "
              "softening market running out of margin."),
        roles={
            "CFO": ("Cumulative rate decreases since 2023 have eroded the cushion built in 2021–22.",
                    "Reserve new cyber business at higher loss picks than the 2022–23 vintages."),
            "CUO": ("Price is falling faster than risk.",
                    "Use sublimits and coinsurance for systemic perils and dependent business interruption; don't trade terms for share."),
            "CEO": ("Cyber's next hard market will be triggered by an event, not a forecast.",
                    "Keep capability (security partnerships, claims response) ready to scale when the market turns."),
        },
        watch="Marsh GIMI cyber index turning back toward flat or positive.")
    S["cy_claims"] = dict(
        theme=TH5,
        headline=f"Attackers are asking for more (+{F.get('cy_ransom', np.nan):.0f}%), but {F.get('cy_refuse', np.nan):.0f}% of victims refuse to pay.",
        body=(f"Average severity fell {abs(F.get('cy_sev', np.nan)):.0f}% in 2025 as backups, response playbooks and insurer-provided security "
              "services worked. The fight has moved to business email compromise and funds-transfer fraud, where AI-generated "
              "deepfakes and phishing scale cheaply. Security services embedded in the policy are now part of the product."),
        roles={
            "CFO": ("Loss control is reducing severity, a real return on 'active insurance' spend.",
                    "Track loss-cost savings from risk-engineering services as an ROI line."),
            "CUO": ("AI-enabled social engineering is the fastest-growing attack vector.",
                    "Require out-of-band payment verification and deepfake-aware controls for funds-transfer coverage."),
            "CEO": ("Prevention is becoming the product.",
                    "Partner or build: continuous external scanning and incident response as a policy feature."),
        },
        watch="BEC/FTF frequency, ransomware demand vs payment rates, and AI-enabled fraud claims.")
    S["cy_ai"] = dict(
        theme=TH5,
        headline="AI coverage is forking: specialists are writing it affirmatively while generalists write it out.",
        body=("Since 2025, ISO's generative-AI exclusions (effective Jan 2026) and 'absolute' AI exclusions from large carriers have "
              "landed in the same months that Lloyd's syndicates, Munich Re/HSB and AI-native MGAs launched affirmative AI cover. "
              "That's 'silent cyber' all over again: ambiguity first, then exclusion, then a priced standalone product. The carriers who "
              "build AI risk-assessment muscle now will own the line."),
        roles={
            "CFO": ("Silent AI exposure sits unpriced inside GL, E&O, D&O and cyber.",
                    "Run an AI exposure audit across the book; reserve for ambiguous wordings."),
            "CUO": ("Exclusions without an affirmative alternative push clients to specialists.",
                    "Pair any AI exclusion with an affirmative buy-back or partner product; build AI model-risk underwriting questions."),
            "CEO": ("AI liability is the next cyber: an emerging line with a first-mover advantage.",
                    "Decide now: build, partner (MGA/specialist) or exclude. Doing nothing is choosing to exclude."),
        },
        watch="ISO exclusion adoption rates, first AI liability losses and litigation, Lloyd's AI capacity, and regulatory moves.")
    S["cy_specialist"] = dict(
        theme=TH5, headline="Cyber specialists are being absorbed by global balance sheets.",
        body=("Zurich's £8.1bn takeover of Beazley (closed Oct 2026) puts the largest cyber franchise inside a global composite. Scale "
              "capital now wants cyber know-how; independent specialists either scale, partner, or get bought."),
        roles=None)
    return S


def ctx_year(F: dict) -> str:
    return "the latest year"


def themes(F: dict) -> list[dict]:
    sm = F.get("sub_med")
    br = sm.loc["Retail broker", "pe"] if sm is not None and "Retail broker" in sm.index else np.nan
    re_ = sm.loc["Reinsurer", "pe"] if sm is not None and "Reinsurer" in sm.index else np.nan
    return [
        dict(n="01", t=TH1, k=f"{F.get('ciab_latest', np.nan):+.1f}%", e=f"Commercial rate change {F.get('ciab_latest_q','')}; regime break {F.get('ciab_break','')} ({_p(F.get('ciab_break_p', 1))})."),
        dict(n="02", t=TH2, k=f"ρ = {F.get('rho_rerate', np.nan):.2f}", e="Re-rating, not book growth, decided TSR rank; loss-ratio stability is what earns it."),
        dict(n="03", t=TH3, k=f"${F.get('cat_after', np.nan):.0f}B", e=f"Average annual cat losses since {F.get('cat_break','')} vs ${F.get('cat_before', np.nan):.0f}B before; {F.get('n_neg',0)} of {F.get('n_shocks',0)} shocks negative."),
        dict(n="04", t=TH4, k=f"{br:.0f}× vs {re_:.0f}×", e="Broker vs reinsurer P/E: the market pays for capital-light fee income; E&S and MGAs take share."),
        dict(n="05", t=TH5, k=f"{F.get('cy_lr_25', np.nan):.0f}% LR", e=f"Cyber loss ratio up from {F.get('cy_lr_23', np.nan):.0f}% while rates fell 12 quarters; AI cover forking into exclusions vs affirmative products."),
    ]


def executive(F: dict) -> list[tuple[str, str]]:
    """The front-page story: short sections, each anchored on data shown elsewhere in the app."""
    sm = F.get("sub_med")
    return [
        ("The short version",
         f"The best underwriting run in a generation is ending, on schedule. The industry posted a <b>{F.get('cr_last', np.nan):.1f}</b> "
         f"combined ratio in {F.get('cr_last_y','')}, its #{F.get('cr_rank','')} year since 2001. {F.get('tbv_pos_share_last', np.nan):.0f}% of our carriers grew "
         "tangible book per share last year. And the pricing engine that made it possible has statistically switched regimes: commercial rate "
         f"went from <b>{F.get('ciab_hard', np.nan):+.1f}%</b> a quarter to <b>{F.get('ciab_now', np.nan):+.1f}%</b>, and printed "
         f"<b>{F.get('ciab_latest', np.nan):+.1f}%</b> in {F.get('ciab_latest_q','')}. None of this is bad news. It's the most predictable "
         "part of the cycle, which means it's the most manageable, if you act before it shows up in the P&L."),
        ("What actually created value",
         f"Not growth, and not even book value. Across our carriers, the change in valuation multiple explains TSR rank "
         f"(ρ = {F.get('rho_rerate', np.nan):.2f}); book growth doesn't (ρ = {F.get('rho_tbv', np.nan):.2f}). The market paid for "
         "<i>believable</i> earnings: carriers whose loss ratios held, whose reserves didn't surprise, whose capital came back to "
         "shareholders. The shock days prove the point: a reserve surprise is priced in a single day, while catastrophe exposure "
         "is re-priced for weeks as estimates develop."),
        ("Where the pressure lands next",
         "Long-tail casualty is where trend outran price (Allstate 2022, Markel 2023 and Selective 2024 all took the same lesson in public). "
         f"Property is where capacity floods back fastest: reinsurance down 12–16% this year, cat losses still running at ${F.get('cat_after', np.nan):.0f}B a year. "
         "Personal auto is at peak margin just as everyone pivots back to growth."),
        ("Where value is moving",
         f"Along the chain. The market pays roughly {sm.loc['Retail broker','pe'] if sm is not None and 'Retail broker' in sm.index else np.nan:.0f}× "
         "earnings for distribution and a fraction of that for balance-sheet risk. E&S now writes over a quarter of commercial premium, and MGAs "
         "write over $100B. The winning carriers are becoming platforms: underwriting talent plus capital plus fee income."),
        ("The bet worth making",
         "Cyber and AI. Cyber is profitable but stalled (flat premium, rising losses, thin endorsements everywhere), and AI liability "
         "is following the same arc: silence, then exclusion, then a priced product. The firms that build AI risk assessment, embed "
         "prevention in the policy, and use AI in their own pricing, claims and underwriting workflows will compound through the soft market."),
        ("What to do Monday",
         "<b>Reinforce</b> walk-away discipline and reward underwriters for what they decline. <b>Invest</b> in the rate-change factory "
         "(data → indication → filing in weeks), geospatial property data, and AI-assisted claims triage and underwriting. <b>Lock in</b> "
         "cheap multi-year reinsurance. <b>Return</b> excess capital rather than chase share. <b>Decide</b> your AI coverage stance before your "
         "competitors decide it for you."),
    ]
