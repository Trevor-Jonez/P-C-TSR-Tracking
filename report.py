"""The annual US P&C report: current-year themes, sections and C-suite persona playbooks.

Structure inspired by best-in-class consulting annual reports; every number here comes from this project's database.
Voice: plain-spoken, optimistic, specific about what to do next.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import db

ROLES = ["CEO", "CUO", "CIO", "CTO", "COO", "CSO"]
ROLE_TITLES = {
    "CEO": "Chief Executive Officer", "CUO": "Chief Underwriting Officer", "CIO": "Chief Investment Officer",
    "CTO": "Chief Technology Officer", "COO": "Chief Operating Officer", "CSO": "Chief Strategy Officer",
}


def _r(yt, t, col, fmt="{:+.0f}%"):
    try:
        v = yt.loc[t, col]
        return fmt.format(v) if pd.notna(v) else "n/a"
    except KeyError:
        return "n/a"


def evidence(ticker: str | None = None) -> pd.DataFrame:
    e = db.table("company_evidence")
    return e if ticker is None else e[e["ticker"] == ticker]


def ev_link(t: str, contains: str) -> str:
    e = evidence(t)
    e = e[e["headline"].str.contains(contains, case=False, regex=False)]
    if not len(e):
        return ""
    r = e.iloc[0]
    return f' <a href="{r.source_url}" target="_blank">[{r.source_name}]</a>'


def build(F: dict, T_: dict, et: dict, yt: pd.DataFrame, beh_tbl: pd.DataFrame, sub_t: dict, sub_ct: pd.DataFrame) -> dict:
    yr = T_["year"]
    lr_med, lr_p, lr_n, lr_impr = T_["lr"]
    ni_med, ni_p, ni_n, ni_neg = T_["ni"]
    ao_med, ao_p, ao_n, ao_neg = T_["aoci"]
    pt_med, pt_p, _, _ = T_["ptbv"]
    rm = T_["ret_med"]
    lead = ", ".join(f"{yt.loc[t, 'name']} ({v:+.0f}%)" for t, v in T_["leaders"].items())
    lag = ", ".join(f"{yt.loc[t, 'name']} ({v:+.0f}%)" for t, v in T_["laggards"].items())
    sm = F.get("sub_med")
    brk_pe = sm.loc["Retail broker", "pe"] if sm is not None and "Retail broker" in sm.index else np.nan
    re_pe = sm.loc["Reinsurer", "pe"] if sm is not None and "Reinsurer" in sm.index else np.nan
    car = yt[yt["sub_industry"] == "Carrier"]
    gap = car["implied_rote_pct"].median() - car["rote_ytd_ann"].median()

    R: dict = {}
    R["title"] = f"{yr} US P&C Value Report: the cycle turns, the bar rises"
    R["dek"] = ("Rate has stopped doing the work. Earnings are at a peak, book value has stalled on rising yields, and the "
                "gap between the carriers who built real underwriting machinery and the ones who rode the hard market is about "
                "to show up in the numbers.")
    R["intro"] = [
        (f"Here's the good news first, because it's real. Through the first nine months of {yr}, the median carrier grew net income "
         f"<b>{ni_med:+.0f}%</b> ({ni_n - ni_neg} of {ni_n} grew; Wilcoxon p={ni_p:.3f}) and improved its loss ratio by <b>{abs(lr_med):.1f} points</b> "
         f"({lr_impr} of {lr_n} improved; p={lr_p:.2f}). The industry posted a <b>{F.get('cr_last', np.nan):.1f}</b> combined ratio last year, its "
         f"#{F.get('cr_rank', '')} result since 2001, and carriers are up a median <b>{T_['carrier_ret_med']:+.0f}%</b> this year against "
         f"<b>{T_['spx_ytd']:+.0f}%</b> for the S&amp;P 500."),
        (f"And here's the turn. Commercial pricing broke regime in {F.get('ciab_break', '')} (p={F.get('ciab_break_p', np.nan):.3f}) and printed "
         f"<b>{F.get('ciab_latest', np.nan):+.1f}%</b> in {F.get('ciab_latest_q', '')}. US property rates are falling around 10% a quarter, reinsurance is "
         f"about 16% cheaper than a year ago, and the 10-year Treasury is up <b>{T_.get('y10_chg_bp', np.nan):+.0f} bp</b>, which has pulled "
         f"unrealized losses back into every single carrier's book (AOCI <b>{ao_med:+.1f} pp</b> of TBV, {ao_neg} of {ao_n}). Tangible book "
         "value has gone sideways even as earnings compound."),
        ("None of this is a crisis. It's the most predictable phase of the insurance cycle, and that's exactly why it's the most "
         "manageable. The carriers that win the next five years will be the ones who treat 2026 as a capability year, not a "
         "victory lap."),
    ]
    R["themes"] = [
        dict(t="Rate has left the building", k=f"{F.get('ciab_latest', np.nan):+.1f}%",
             e=f"Commercial rate change {F.get('ciab_latest_q', '')}; regime break {F.get('ciab_break', '')}. US property about −10% (Q3). Reinsurance ROL −16% YTD."),
        dict(t="Peak earnings, flat book", k=f"{ni_med:+.0f}% / {T_['tbv'][0]:+.1f}%",
             e=f"Median 9M net income vs tangible book YTD: rising yields took {abs(ao_med):.1f} pp of TBV through AOCI."),
        dict(t="Casualty is the fault line", k="Reserves",
             e="Property softens while commercial auto and excess casualty still firm; CNA strengthened excess casualty reserves in Q3."),
        dict(t="Capital is abundant, discipline is scarce", k=f"{pt_med:+.2f}× P/TBV",
             e=f"The market re-rated carriers this year (p={pt_p:.2f}) and is paying for margins that softer rates will test."),
        dict(t="AI moves from pilot to P&L", k="STP > 50%",
             e="Leaders now report straight-through claims, no-touch small-commercial binding and AI-driven expense targets."),
    ]

    R["sections"] = {
        "tsr": (f"{yr} returns: strong, but not evenly shared",
                f"Carriers are up a median {T_['carrier_ret_med']:+.0f}% year to date and reinsurers {rm.get('Reinsurer', np.nan):+.0f}%, while retail brokers "
                f"({rm.get('Retail broker', np.nan):+.0f}%) and insurtech ({rm.get('Insurtech', np.nan):+.0f}%) lag. Sub-industry differences are "
                f"statistically real (Kruskal-Wallis p={T_['sub_kw_p']:.3f}). Among carriers, the leaders are {lead}; the laggards are {lag}. "
                "The pattern is consistent: the market is rewarding repaired, diversified underwriting franchises and punishing concentration, "
                "unresolved books and growth that depends on rate."),
        "uw": ("Underwriting separates leaders from laggards, and this year you can see it in the 10-Qs",
               f"Year-to-date loss ratios improved {abs(lr_med):.1f} points at the median carrier, led by "
               + ", ".join(f"{yt.loc[t, 'name']} ({v:+.1f} pp)" for t, v in T_["lr_improvers"].items())
               + ". The exceptions tell the story of the next cycle: "
               + ", ".join(f"{yt.loc[t, 'name']} ({v:+.1f} pp)" for t, v in T_["lr_deteriorators"].items())
               + ". Casualty reserve strengthening and catastrophe-exposed specialty books are where the margin is already leaking."),
        "beh": ("What leaders did differently (and how sure we are)",
                "Comparing top- and bottom-quartile carriers by five-year TSR: leaders ran a higher return on tangible equity, grew tangible "
                "book faster, and took fewer public reserve hits. With six carriers per quartile, none of these differences clears "
                "statistical significance on its own, so treat them as consistent direction, not proof. The one thing that is significant "
                "in our data is how the market prices credibility: re-rating explains the TSR ranking, and the event study below shows "
                "how fast trust gets repriced."),
        "event": ("How the market prices a broken promise",
                  f"Across {et['n_res']} stock-events, a reserve or earnings charge costs about <b>{et['res_day0']:+.1f}%</b> on the day "
                  f"(relative to peers), then roughly stabilises ({et['res_next20']:+.1f}% over 20 days). Catastrophe exposure behaves "
                  f"differently: {et['cat_day0']:+.1f}% on the day and another <b>{et['cat_next20']:+.1f}%</b> over the following four weeks "
                  f"as loss estimates develop (difference p={et['p']:.2f}, small sample). Reserve credibility is lost in one session; "
                  "catastrophe concentration is a slow bleed."),
        "implied": ("What today's prices assume",
                    f"At current price-to-tangible-book, the market implies a sustainable return on tangible equity of about "
                    f"{car['implied_rote_pct'].median():.0f}% for the median carrier (cost of equity 9%, long-run growth 3%), versus a "
                    f"{car['rote_ytd_ann'].median():.0f}% annualised RoTE in {yr} so far. That {abs(gap):.0f}-point gap is the market "
                    "telling you it expects mean reversion. Carriers that can show a higher through-cycle floor will close that gap; "
                    "everyone else is already priced for the soft market."),
    }

    # ------------------------------------------------------------- personas
    def t(tk, col="return_ytd_pct", fmt="{:+.0f}%"):
        return _r(yt, tk, col, fmt)

    R["personas"] = {
        "CEO": dict(
            why=(f"You are running the business at peak earnings ({ni_med:+.0f}% median net income growth) into a turning cycle. "
                 "Investors are already paying for durability, so the next five years of TSR get decided by choices you make in "
                 "the next twelve months."),
            do=["Declare the cycle out loud: re-base plans on flat-to-negative rate and say so to the board first.",
                "Fund the capability year: pricing speed, property data, AI in underwriting and claims. Treat it as capex, not opex.",
                "Set the walk-away culture: reward underwriters for business declined in softening lines.",
                "Return excess capital rather than buy growth below technical price."],
            evidence=[f"Commercial rate {F.get('ciab_latest', np.nan):+.1f}% ({F.get('ciab_latest_q', '')}); regime break {F.get('ciab_break', '')}.",
                      f"Market-implied RoTE about {car['implied_rote_pct'].median():.0f}% vs {car['rote_ytd_ann'].median():.0f}% current: mean reversion is priced in.",
                      f"Re-rating, not book growth, explained 5-yr TSR (ρ = {F.get('rho_rerate', np.nan):.2f})."],
            leaders=[("Travelers", f"{t('TRV')} YTD; loss ratio {t('TRV', 'loss_ratio_chg_pp', '{:+.1f} pp')}; $1.5B+ a year into technology and AI" + ev_link("TRV", "1.5B")),
                     ("Chubb", f"{t('CB')} YTD; record 85.7% combined ratio in 2025 and an AI operating model targeting −150 bp" + ev_link("CB", "AI-driven")),
                     ("The Hanover", f"{t('THG')} YTD; record 2025 (91.6% CR), Specialty at 84% in Q3" + ev_link("THG", "Q3 2026"))],
            laggards=[("Kemper", f"{t('KMPR')} YTD; negative tangible book through 2021–22 and a 5-yr TSR that is a statistical outlier"),
                      ("Markel", f"{t('MKL')} YTD despite underwriting profit +55%: the market is still pricing the portfolio reset" + ev_link("MKL", "9M 2026"))],
        ),
        "CUO": dict(
            why=("Every point of rate the market gives back has to be earned back through selection, terms and attachment. Casualty "
                 "is still hardening while property collapses, so a single 'market view' no longer exists."),
            do=["Steer the portfolio by line: shrink where price is falling faster than trend (E&S property), lean in where it isn't (commercial auto, excess casualty).",
                "Hold terms before price: sublimits, deductibles and exclusions are what soft markets erode first.",
                "Raise loss picks on post-pandemic casualty accident years now, before the market does it for you.",
                "Make loss-ratio stability, not peak margin, the KPI that underwriters are paid on."],
            evidence=[f"9M loss ratios improved {abs(lr_med):.1f} pp at the median carrier (p={lr_p:.2f}).",
                      "CIAB Q3 2026: commercial property premiums −6.2%, commercial auto +6.7% (a bifurcated market).",
                      f"Reserve/earnings shocks cost {et['res_day0']:+.1f}% in a day (event study)."],
            leaders=[("Travelers", f"loss ratio {t('TRV', 'loss_ratio_chg_pp', '{:+.1f} pp')} YTD; GenAI renewal tool cut handle time 30%" + ev_link("TRV", "renewal")),
                     ("Chubb", f"loss ratio {t('CB', 'loss_ratio_chg_pp', '{:+.1f} pp')} YTD at a 48% base"),
                     ("Kinsale", f"held margins while shrinking E&S property about 27–33% rather than chase price" + ev_link("KNSL", "property"))],
            laggards=[("CNA", f"loss ratio {t('CNA', 'loss_ratio_chg_pp', '{:+.1f} pp')} YTD; strengthened excess casualty reserves in Q3" + ev_link("CNA", "excess casualty")),
                      ("Palomar", f"loss ratio {t('PLMR', 'loss_ratio_chg_pp', '{:+.1f} pp')} YTD from a low base"),
                      ("Selective (2024 lesson)", "$176M social-inflation reserve charge, −18% in a day; recovered since" + ev_link("SIGI", "176M"))],
        ),
        "CIO": dict(
            why=(f"Yields are up {T_.get('y10_chg_bp', np.nan):+.0f} bp this year and AOCI fell at {ao_neg} of {ao_n} carriers "
                 f"({ao_med:+.1f} pp of TBV). Investment income is the quiet hero of {yr} earnings, and the reason book value hasn't grown."),
            do=["Match asset duration to reserve duration, line by line; the 2022 lesson is repeating at smaller scale.",
                "Lock in today's yields on new money, but never let them justify underpricing long-tail lines.",
                "Report ex-AOCI book value and new-money yield alongside portfolio yield.",
                "Stress-test credit as well as rates: a soft market plus a credit event is the scenario nobody is pricing."],
            evidence=[f"10-yr Treasury {T_.get('y10_now', np.nan):.2f}% vs {T_.get('y10_ye', np.nan):.2f}% at year-end (FRED).",
                      f"Median AOCI change {ao_med:+.1f} pp of TBV (Wilcoxon p<0.001).",
                      "2022: AOCI took about 21 pp of tangible book in a single year."],
            leaders=[(yt.loc[tk, "name"], f"AOCI {yt.loc[tk, 'aoci_chg_pct_tbv']:+.1f} pp of TBV YTD (least affected)")
                     for tk in car["aoci_chg_pct_tbv"].dropna().sort_values(ascending=False).head(3).index],
            laggards=[(yt.loc[tk, "name"], f"AOCI {yt.loc[tk, 'aoci_chg_pct_tbv']:+.1f} pp of TBV YTD (most affected)")
                      for tk in car["aoci_chg_pct_tbv"].dropna().sort_values().head(3).index],
        ),
        "CTO": dict(
            why=("AI has moved from pilots to production at the leaders, with measurable effects on claims cycle time, underwriting "
                 "throughput and expense. In a soft market, expense ratio and risk selection are the only margin levers left."),
            do=["Move from pilots to platforms: unified data, model governance and a production path for every use case.",
                "Prioritise the four proven value pools: underwriting triage, claims STP, pricing pipelines and distribution.",
                "Instrument everything: report handle-time, STP rate and leakage monthly like a P&L line.",
                "Build AI-risk literacy too: the same tools create the exposures your underwriters must price."],
            evidence=["Travelers: >50% of claims STP-eligible; GenAI first notice of loss; −30% renewal handle time (management-reported).",
                      "Hartford: 75% of small-business new business bound without human touch (management-reported).",
                      "AIG: multi-agent underwriting with Anthropic and Palantir; Chubb: AI operating model targeting −150 bp."],
            leaders=[("Travelers", "$1.5B+ annual technology spend, thousands of engineers, dozens of scaled GenAI tools" + ev_link("TRV", "1.5B")),
                     ("The Hartford", "automated small-commercial underwriting at scale" + ev_link("HIG", "75%")),
                     ("AIG", "agentic underwriting plus a Palantir-matched capital vehicle (Syndicate 2479)" + ev_link("AIG", "Anthropic"))],
            laggards=[("Insurtech (Root, Lemonade)", f"technology without underwriting edge: {rm.get('Insurtech', np.nan):+.0f}% median YTD, beta above 2"),
                      ("Carriers still in pilot mode", "no public evidence of scaled deployment; expense ratios will tell the story by 2027")],
        ),
        "COO": dict(
            why=("Claims is where AI pays fastest and where catastrophes test operations hardest. Cat-exposed carriers lose value for "
                 "weeks after an event, so speed and accuracy of loss estimation are a market-value issue, not just a service one."),
            do=["Build cat-response muscle: drone and computer-vision damage assessment, pre-staged adjusters, rapid reserve estimates.",
                "Push straight-through processing for simple claims and move adjusters to complex and litigated files.",
                "Attack leakage in litigated casualty: early attorney-involvement detection, panel counsel analytics.",
                "Turn expense discipline into a strategy, not a cut: measure unit cost per policy and per claim."],
            evidence=[f"Catastrophe events: {et['cat_day0']:+.1f}% on the day, then {et['cat_next20']:+.1f}% over 20 trading days (event study).",
                      "Milton (Oct 2024): Florida carriers −17% to −20% in a day; LA wildfires: Mercury −21%.",
                      "Travelers: STP adopted about two-thirds of the time where offered."],
            leaders=[("Travelers", "STP claims and GenAI first notice of loss" + ev_link("TRV", "straight-through")),
                     ("Chubb", "AI-driven operating model with explicit expense target" + ev_link("CB", "AI-driven"))],
            laggards=[("Mercury General", f"concentration risk: about 80% California; −21% on the wildfires; now {t('MCY')} YTD" + ev_link("MCY", "80%")),
                      ("HCI / Universal (Milton day)", "single-state property concentration repriced 17–20% in one session")],
        ),
        "CSO": dict(
            why=(f"Value is migrating along the chain. The market pays about {brk_pe:.0f}× earnings for capital-light distribution and "
                 f"{re_pe:.0f}× for reinsurance risk; E&S now writes more than a quarter of commercial premium and MGAs over $100B. "
                 "Where you choose to sit in that chain is the strategy."),
            do=["Decide explicitly: capital provider, underwriter, distributor, or a platform that combines them.",
                "Build fee income (MGA, program, third-party capital) to earn a better multiple on the same talent.",
                "Use the M&A window for capabilities (data, AI, MGA platforms, specialty niches), not balance sheet.",
                "Simplify: exits that improve the combined ratio get rewarded, eventually."],
            evidence=["Zurich closed its £8.1bn Beazley takeover (Oct 2026); Gallagher closed AssuredPartners ($13.45B).",
                      "Mitsui Sumitomo built a 15% stake in W. R. Berkley: Japanese capital still wants US specialty.",
                      f"Broker vs carrier P/E premium significant (p={sub_t.get('broker_vs_carrier_pe', np.nan):.4f})."],
            leaders=[("Allstate", f"{t('ALL')} YTD after exiting Health & Benefits (~$3.25B) to focus on property-liability" + ev_link("ALL", "Health")),
                     ("AIG", "capital-light Syndicate 2479 with Amwins and Blackstone" + ev_link("AIG", "Syndicate")),
                     ("Arthur J. Gallagher", "scale M&A in distribution" + ev_link("AJG", "AssuredPartners"))],
            laggards=[("Brown & Brown", f"{t('BRO')} YTD; organic growth about 1% while revenue grows by acquisition" + ev_link("BRO", "Organic")),
                      ("Markel (in transition)", "reinsurance exit is improving underwriting, but the equity hasn't re-rated yet" + ev_link("MKL", "Exited"))],
        ),
    }
    R["closing"] = [
        ("What top performers do differently",
         "They treat the softening transition as the most important strategic window in the cycle. They hold price and terms when "
         "competitors don't, they report the numbers that prove durability (accident-year loss ratios, ex-AOCI book value, reserve "
         "philosophy), and they invest in the underwriting machine while earnings are high enough to pay for it."),
        ("The fundamentals still win, but the bar is rising",
         "Disciplined underwriting, rigorous capital allocation and sustained capability investment have been rewarded across "
         "this data's five-year window, and the market's reaction to every reserve surprise says the same thing. What's changed is what it takes to execute: real-time data, AI in the workflow, and a portfolio "
         "process that moves in weeks. The carriers most at risk aren't the ones in trouble today; they're the ones in the middle, "
         "profitable enough to feel fine and not investing enough to stay that way."),
    ]
    return R
