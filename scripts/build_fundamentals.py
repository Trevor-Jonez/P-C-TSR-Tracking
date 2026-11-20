"""Build data/fundamentals.csv from SEC EDGAR XBRL company facts (10-K annual data).

Usage:
    SEC_USER_AGENT="Your Name your.email@domain.com" python scripts/build_fundamentals.py

SEC requires a descriptive User-Agent with contact details for automated access.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from universe import UNIVERSE  # noqa: E402

UA = os.environ.get("SEC_USER_AGENT", "PC-TSR-Tracker research@example.com")
YEARS = range(2019, 2026)

# Ordered fallbacks: the first tag with a value for a given year wins.
TAGS = {
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "preferred": ["PreferredStockValue", "PreferredStockValueOutstanding"],
    "goodwill": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
    "aoci": ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
    "net_income": ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss", "ProfitLoss"],
    "premiums_earned": ["PremiumsEarnedNet", "PremiumsEarnedNetPropertyAndCasualty"],
    "losses_incurred": [
        "PolicyholderBenefitsAndClaimsIncurredNet",
        "IncurredClaimsPropertyCasualtyAndLiability",
        "LiabilityForUnpaidClaimsAndClaimsAdjustmentExpenseIncurredClaims1",
    ],
    "py_development": ["LiabilityForUnpaidClaimsAndClaimsAdjustmentExpenseIncurredClaimsPriorYears"],
    "net_investment_income": ["NetInvestmentIncome"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
}
INSTANT = {"equity", "preferred", "goodwill", "intangibles", "aoci"}


def fetch(cik: int) -> dict:
    req = urllib.request.Request(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers={"User-Agent": UA}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["facts"]


def series(facts: dict, tag: str, instant: bool) -> dict:
    for ns in ("us-gaap", "ifrs-full"):
        node = facts.get(ns, {}).get(tag)
        if not node:
            continue
        units = node["units"].get("USD") or next(iter(node["units"].values()))
        out = {}
        for u in units:
            fr = u.get("frame", "")
            if instant and fr.endswith("Q4I"):
                out[int(fr[2:6])] = u["val"]
            elif not instant and len(fr) == 6 and fr.startswith("CY"):
                out[int(fr[2:6])] = u["val"]
        # Fallback: 10-K facts without a calendar frame (common for rollforward tables).
        # Take the most recently filed value for a Dec-31 year end (restated figures win).
        fallback = {}
        for u in sorted(units, key=lambda x: x.get("filed", "")):
            if u.get("form") != "10-K" or not u.get("end", "").endswith("-12-31"):
                continue
            y = int(u["end"][:4])
            if not instant:
                if "start" not in u:
                    continue
                days = (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days
                if not 360 <= days <= 370:
                    continue
            fallback[y] = u["val"]
        for y, v in fallback.items():
            out.setdefault(y, v)
        return out
    return {}


def shares_by_year(facts: dict) -> dict:
    """Cover-page shares outstanding reported in each 10-K (dated shortly after fiscal year end)."""
    node = facts.get("dei", {}).get("EntityCommonStockSharesOutstanding")
    out = {}
    if node:
        for u in next(iter(node["units"].values())):
            # 10-K cover pages report shares as of a date shortly after the Dec-31 year end;
            # map by that date rather than the (sometimes inconsistent) fy field.
            if u.get("form") == "10-K" and int(u["end"][5:7]) <= 4:
                y = int(u["end"][:4]) - 1
                if u["val"] > out.get(y, (0, None))[0]:
                    out[y] = (u["val"], u["end"])  # (shares, as-of date on the 10-K cover page)
    return out


def main() -> None:
    rows = []
    for co in UNIVERSE:
        facts = fetch(co["cik"])
        time.sleep(0.2)
        got = {}
        for metric, tags in TAGS.items():
            merged = {}
            for tag in tags:
                for y, v in series(facts, tag, metric in INSTANT).items():
                    merged.setdefault(y, v)
            got[metric] = merged
        shares = shares_by_year(facts)
        for y in YEARS:
            sh, sh_date = shares.get(y, (None, None))
            rows.append({"ticker": co["ticker"], "fiscal_year": y, "shares_outstanding": sh, "shares_as_of": sh_date,
                         **{m: got[m].get(y) for m in TAGS}})
        print(f"{co['ticker']:5s} ok", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "fundamentals_raw.csv", index=False)
    print(f"wrote {len(df)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
