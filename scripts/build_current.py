"""Current-year (YTD) fundamentals from the latest 10-Q, and material 8-K events, from SEC EDGAR.

Writes data/fundamentals_ytd.csv and data/filings_8k.csv (loaded into the database by build_database.py).
Usage: SEC_USER_AGENT="Your Name you@example.com" python scripts/build_current.py
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
FLOW = {
    "premiums_earned": ["PremiumsEarnedNet", "PremiumsEarnedNetPropertyAndCasualty"],
    "losses_incurred": ["PolicyholderBenefitsAndClaimsIncurredNet", "IncurredClaimsPropertyCasualtyAndLiability"],
    "net_income": ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss", "ProfitLoss"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
}
STOCK = {
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "goodwill": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
    "aoci": ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
}
ITEMS = {
    "1.01": "Material agreement", "1.02": "Agreement terminated", "2.01": "Acquisition / disposition completed",
    "2.05": "Restructuring / exit costs", "2.06": "Material impairment", "5.02": "Executive / board change",
    "8.01": "Other material event", "7.01": "Reg FD disclosure",
}
KEEP = {"1.01", "1.02", "2.01", "2.05", "2.06", "5.02", "8.01"}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def units(facts: dict, tag: str) -> list:
    for ns in ("us-gaap",):
        n = facts.get(ns, {}).get(tag)
        if n:
            return n["units"].get("USD") or next(iter(n["units"].values()))
    return []


def ytd_value(facts: dict, tags: list, start: str, end: str):
    for t in tags:
        vals = [u for u in units(facts, t) if u.get("start") == start and u.get("end") == end]
        if vals:
            return sorted(vals, key=lambda u: u.get("filed", ""))[-1]["val"]
    return None


def instant(facts: dict, tags: list, end: str):
    for t in tags:
        vals = [u for u in units(facts, t) if u.get("end") == end and "start" not in u]
        if vals:
            return sorted(vals, key=lambda u: u.get("filed", ""))[-1]["val"]
    return None


def latest_q_end(facts: dict, year: int) -> str | None:
    ends = {u["end"] for t in FLOW["net_income"] for u in units(facts, t)
            if u.get("form") == "10-Q" and u.get("start") == f"{year}-01-01" and u.get("end", "").startswith(str(year))}
    return max(ends) if ends else None


def main() -> None:
    year = pd.Timestamp.today().year
    rows, ev = [], []
    for co in UNIVERSE:
        cik = co["cik"]
        try:
            facts = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")["facts"]
        except Exception as e:  # noqa: BLE001
            print(f"{co['ticker']}: facts failed {e}", file=sys.stderr)
            continue
        time.sleep(0.15)
        end = latest_q_end(facts, year)
        if end:
            p_end = f"{year - 1}{end[4:]}"
            r = {"ticker": co["ticker"], "period_end": end, "months": int(end[5:7])}
            for k, tags in FLOW.items():
                r[f"{k}_ytd"] = ytd_value(facts, tags, f"{year}-01-01", end)
                r[f"{k}_ytd_prior"] = ytd_value(facts, tags, f"{year - 1}-01-01", p_end)
            for k, tags in STOCK.items():
                r[f"{k}_now"] = instant(facts, tags, end)
                r[f"{k}_ye"] = instant(facts, tags, f"{year - 1}-12-31")
            rows.append(r)
        try:
            sub = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")["filings"]["recent"]
            time.sleep(0.15)
            df = pd.DataFrame(sub)
            df = df[(df["form"] == "8-K") & (df["filingDate"] >= f"{year - 2}-01-01")]
            for _, f in df.iterrows():
                items = [i.strip() for i in str(f.get("items", "")).split(",") if i.strip()]
                keep = [i for i in items if i in KEEP]
                if not keep:
                    continue
                acc = f["accessionNumber"].replace("-", "")
                ev.append({"ticker": co["ticker"], "date": f["filingDate"], "items": ",".join(keep),
                           "labels": "; ".join(ITEMS.get(i, i) for i in keep),
                           "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{f['primaryDocument']}"})
        except Exception as e:  # noqa: BLE001
            print(f"{co['ticker']}: submissions failed {e}", file=sys.stderr)
        print(f"{co['ticker']:5s} ytd={'yes' if end else 'no'}", file=sys.stderr)
    pd.DataFrame(rows).to_csv(ROOT / "data" / "fundamentals_ytd.csv", index=False)
    pd.DataFrame(ev).to_csv(ROOT / "data" / "filings_8k.csv", index=False)
    print(f"{len(rows)} YTD rows, {len(ev)} 8-K events", file=sys.stderr)


if __name__ == "__main__":
    main()
