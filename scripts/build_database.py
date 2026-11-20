"""Build data/pc_market.db: a provenance-tracked SQLite database of P&C industry and carrier data.

Sources
  * USC Risk & Uncertainty Management Center (Hartwig) presentations: chart data embedded in the
    published .pptx decks, every vintage kept (https://www.uscriskcenter.com/presentations/)
  * FRED (Federal Reserve Bank of St. Louis): live macro / price series used to reconcile the decks
  * SEC EDGAR XBRL (via data/fundamentals_raw.csv from scripts/build_fundamentals.py): carrier fundamentals
  * Yahoo Finance price snapshot (data/prices_*.csv from scripts/build_prices.py)

Usage:  python scripts/build_database.py            # download decks + FRED, rebuild DB
        python scripts/build_database.py --offline  # reuse cached decks / FRED pulls
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"
DB = DATA / "pc_market.db"
INDEX_URL = "https://www.uscriskcenter.com/presentations/"
UA = "Mozilla/5.0 (compatible; pc-tsr-tracker/1.0)"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------------------------------------------------------------------------
# Metric catalogue: how a chart in a deck maps to a canonical metric
#   (title regex, series regex or None for "the data series", metric, unit, scale, dimension mode)
#   scale "frac" -> decimal fractions converted to percent; "m2b" -> $ millions to $ billions
# ---------------------------------------------------------------------------
RULES = [
    (r"^P/C Insurance Industry Combined Ratio", None, "industry_combined_ratio", "%", None, None),
    (r"^P/C Direct Premiums Written Volume and Growth", r"^DPW Growth$", "industry_dpw_growth", "%", "frac", None),
    (r"^P/C Direct Premiums Written Volume and Growth", r"^DPW$", "industry_dpw", "$B", None, None),
    (r"^P/C Industry Net Income After Taxes", None, "industry_net_income", "$B", "m2b", None),
    (r"^ROE: Property/Casualty", None, "industry_roe", "%", "frac", None),
    (r"^Net Written Premium Growth \(All P/C", r"NPW Growth", "industry_npw_growth", "%", "frac", None),
    (r"^Commercial Lines Direct Premiums Written", None, "commercial_dpw_growth_h1", "%", "frac", None),
    (r"^CIAB: Average Commercial Rate Change, All Lines", None, "ciab_commercial_rate_change", "%", "frac", None),
    (r"^Change in Commercial Rate Renewals, by Line", None, "ciab_rate_change_by_line", "%", "frac", "category"),
    (r"^The Economy Drives P/C", r"DWP y-o-y", "industry_dwp_growth_yoy", "%", "frac", None),
    (r"^The Economy Drives P/C", r"nominal GDP", "deck_nominal_gdp_growth_yoy", "%", "frac", None),
    (r"Inflation-Adjusted Insured CAT Losses", r"^(Insured CAT Losses|Losses \$ B)$", "insured_cat_losses_real", "$B", None, None),
    (r"^Personal Auto Combined Ratio", None, "personal_auto_combined_ratio", "%", None, None),
    (r"^Personal Auto: Direct Premiums Written", r"^DPW Growth$", "personal_auto_dpw_growth", "%", "frac", None),
    (r"^Workers Comp Combined Ratios", None, "workers_comp_combined_ratio", "%", None, None),
    (r"^Combined Ratios by Line", None, "combined_ratio_by_line", "%", None, "category_series_year"),
    (r"^Collision Claim Severity", None, "collision_claim_severity", "$", None, None),
    (r"^New Private Housing Starts", None, "deck_housing_starts", "millions", None, None),
    (r"^Auto/Light Truck Sales", None, "deck_light_vehicle_sales", "millions", None, None),
    (r"^Employment: Insurance Carriers", None, "deck_insurance_carrier_employment", "thousands", None, None),
    (r"^Medical Cost Inflation vs. Overall CPI", r"Medical CPI", "deck_medical_cpi_yoy", "%", None, None),
    (r"^Medical Cost Inflation vs. Overall CPI", r"CPI-All Items", "deck_cpi_all_items_yoy", "%", None, None),
]
SKIP_SERIES = re.compile(r"^(shaded|Average for Decade|Recession)$", re.I)

CATALOG = {
    "industry_combined_ratio": ("Industry combined ratio (excl. mortgage & financial guaranty)", "A.M. Best; ISO/Verisk; S&P Global"),
    "industry_dpw_growth": ("Industry direct premiums written, growth", "A.M. Best; NAIC; S&P Global; Swiss Re (forecasts)"),
    "industry_dpw": ("Industry direct premiums written", "A.M. Best; NAIC; S&P Global; Swiss Re (forecasts)"),
    "industry_net_income": ("Industry net income after taxes", "A.M. Best; ISO/Verisk; APCIA"),
    "industry_roe": ("Industry return on equity (GAAP)", "A.M. Best; ISO; APCIA"),
    "industry_npw_growth": ("Industry net written premium growth", "A.M. Best; ISO"),
    "commercial_dpw_growth_h1": ("Commercial lines DPW growth (H1 y/y)", "S&P Capital IQ Pro; Insurance Insider US"),
    "ciab_commercial_rate_change": ("Average commercial rate change, all lines (quarterly)", "Council of Insurance Agents & Brokers survey"),
    "ciab_rate_change_by_line": ("Commercial renewal rate change by line", "Council of Insurance Agents & Brokers survey"),
    "industry_dwp_growth_yoy": ("Industry DWP growth y/y (quarterly to 2019, annual after)", "SNL/S&P; ISO; Swiss Re"),
    "deck_nominal_gdp_growth_yoy": ("Nominal GDP growth y/y as shown in deck", "US BEA (per deck)"),
    "insured_cat_losses_real": ("US insured catastrophe losses, inflation-adjusted", "Verisk PCS; Munich Re; Aon"),
    "personal_auto_combined_ratio": ("Personal auto combined ratio", "APCIA via S&P Global Market Intelligence"),
    "personal_auto_dpw_growth": ("Personal auto DPW growth", "A.M. Best; NAIC; S&P Global"),
    "workers_comp_combined_ratio": ("Workers compensation combined ratio", "NCCI; A.M. Best"),
    "combined_ratio_by_line": ("Combined ratio by line of business", "A.M. Best"),
    "collision_claim_severity": ("Collision claim severity (quarterly)", "ISO/PCI Fast Track"),
    "deck_housing_starts": ("New private housing starts as shown in deck", "US Commerce Dept.; Wells Fargo (forecasts)"),
    "deck_light_vehicle_sales": ("Auto/light truck sales as shown in deck", "US Commerce Dept.; Wells Fargo (forecasts)"),
    "deck_insurance_carrier_employment": ("Insurance carrier & related employment as shown in deck", "BLS CES"),
    "deck_medical_cpi_yoy": ("Medical care CPI y/y as shown in deck", "BLS via FRED (CPIMEDSL)"),
    "deck_cpi_all_items_yoy": ("CPI all items y/y as shown in deck", "BLS via FRED"),
}

FRED = {
    "GDP": "Nominal GDP (SAAR, $B)",
    "HOUST": "Housing starts (SAAR, thousands)",
    "ALTSALES": "Light vehicle sales (SAAR, millions)",
    "CPIAUCSL": "CPI all items (index)",
    "CPIMEDSL": "CPI medical care (index)",
    "CUSR0000SETE": "CPI motor vehicle insurance (index)",
    "CES5552410001": "Employment: insurance carriers (thousands)",
    "CES5552400001": "Employment: insurance carriers & related (thousands)",
    "DGS10": "10-year Treasury yield (%)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get(url: str, retries: int = 3, pause: float = 5.0) -> bytes:
    """Fetch with curl (honours the environment's proxy and CA settings), retrying with backoff."""
    import subprocess

    last = None
    for k in range(retries):
        # curl's own default User-Agent: honest, and accepted by FRED (which rejects some custom strings)
        r = subprocess.run(["curl", "-sSL", "--max-time", "60", url], capture_output=True)
        body = r.stdout
        if r.returncode == 0 and body and not (b"fred" in url.encode() and body[:15].lower().startswith((b"<!doctype", b"<html"))):
            return body
        last = r.stderr.decode(errors="ignore")[:200] or "bot-check page instead of CSV"
        time.sleep(pause * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def parse_period(cat: str) -> tuple[str | None, str | None, str]:
    """Returns (period, period_type, flag). flag: A=actual, E=estimate, F=forecast, P=preliminary."""
    c = cat.strip().replace("*", "")
    flag = "A"
    m = re.fullmatch(r"(\d{2}|\d{4})\s*([EFP])?", c)
    if m:
        y = int(m.group(1))
        y = y if y > 1900 else (1900 + y if y >= 50 else 2000 + y)
        return str(y), "year", m.group(2) or flag
    m = re.fullmatch(r"(\d{2}|\d{4}):\s*(H[12]|Q[1-4])", c)
    if m:
        y = int(m.group(1))
        y = y if y > 1900 else 2000 + y
        return f"{y}-{m.group(2)}", "half" if m.group(2).startswith("H") else "quarter", flag
    m = re.fullmatch(r"([1-4])Q(\d{2})", c)
    if m:
        return f"{2000 + int(m.group(2))}-Q{m.group(1)}", "quarter", flag
    m = re.fullmatch(r"([A-Z][a-z]{2})\s+(\d{2})", c)
    if m:
        dt = datetime.strptime(f"{m.group(1)} 20{m.group(2)}", "%b %Y")
        return dt.strftime("%Y-%m"), "month", flag
    m = re.fullmatch(r"(\d{2})-(\d{2})", c)
    if m:
        return f"19{m.group(1)}-19{m.group(2)}", "range", flag
    return None, None, flag


def deck_vintage(name: str, url: str) -> str:
    m = re.match(r"(\d{2})(\d{2})(\d{2})-", name)
    if m:
        return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"/uploads/(\d{4})/(\d{2})/", url)
    return f"{m.group(1)}-{m.group(2)}-15" if m else "1900-01-01"


# ---------------------------------------------------------------------------
# Ingest: Hartwig decks
# ---------------------------------------------------------------------------
def ingest_decks(offline: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pptx import Presentation

    CACHE.mkdir(parents=True, exist_ok=True)
    idx_file = CACHE / "presentations.html"
    if not offline or not idx_file.exists():
        idx_file.write_bytes(get(INDEX_URL))
    html = idx_file.read_text(errors="ignore")
    pptx_urls = sorted(set(re.findall(r'https://[^"\']+\.pptx', html)))
    pdf_urls = sorted(set(re.findall(r'https://[^"\']+\.pdf', html)))

    sources, raw = [], []
    for url in pptx_urls:
        name = Path(url).stem
        path = CACHE / f"{name}.pptx"
        if not path.exists() or not offline:
            if not path.exists():
                path.write_bytes(get(url))
        blob = path.read_bytes()
        pdf = next((p for p in pdf_urls if Path(p).stem.startswith(re.sub(r"-\d+$", "", name)[:20])), None)
        sid = f"usc:{name}"
        prs = Presentation(io.BytesIO(blob))
        title0 = ""
        for sh in prs.slides[0].shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                title0 = re.sub(r"\s+", " ", sh.text_frame.text.strip())[:200]
                break
        sources.append({"source_id": sid, "publisher": "USC Risk & Uncertainty Management Center (R. Hartwig)",
                        "title": title0 or name, "url": url, "pdf_url": pdf, "vintage_date": deck_vintage(name, url),
                        "retrieved_at": NOW, "sha256": hashlib.sha256(blob).hexdigest(), "kind": "presentation"})

        def walk(shapes):
            for s in shapes:
                if s.shape_type == 6:
                    yield from walk(s.shapes)
                else:
                    yield s

        for si, slide in enumerate(prs.slides, 1):
            shapes = list(walk(slide.shapes))
            texts = [s.text_frame.text.strip() for s in shapes if s.has_text_frame and s.text_frame.text.strip()]
            title = slide.shapes.title.text_frame.text.strip() if slide.shapes.title is not None else (texts[0] if texts else "")
            note = " | ".join(t for t in texts if re.search(r"(?i)sources?:|^\s*(\*|note)", t))
            for s in shapes:
                if not getattr(s, "has_chart", False):
                    continue
                for plot in s.chart.plots:
                    try:
                        cats = [str(c) for c in plot.categories]
                    except Exception:  # noqa: BLE001
                        cats = []
                    for ser in plot.series:
                        try:
                            vals = list(ser.values)
                        except Exception:  # noqa: BLE001
                            vals = []
                        for c, v in zip(cats, vals):
                            raw.append({"source_id": sid, "slide": si, "slide_title": re.sub(r"\s+", " ", title)[:250],
                                        "series": str(ser.name), "category": c, "value": v,
                                        "source_note": re.sub(r"\s+", " ", note)[:1000]})
        print(f"  deck {name}: ok", file=sys.stderr)
    return pd.DataFrame(sources), pd.DataFrame(raw)


def normalize(raw: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    vint = sources.set_index("source_id")["vintage_date"]
    out = []
    for (sid, slide, title, series), g in raw.groupby(["source_id", "slide", "slide_title", "series"], sort=False):
        if SKIP_SERIES.match(series or "") or g["value"].notna().sum() == 0:
            continue
        for t_re, s_re, metric, unit, scale, dim_mode in RULES:
            if not re.search(t_re, title):
                continue
            if s_re and not re.search(s_re, series or ""):
                continue
            vals = g["value"].astype(float)
            cats = [str(c) for c in g["category"]]
            fc_years = {re.sub(r"\D", "", c) for c in cats if re.fullmatch(r"\d{2,4}F\*?", c.strip())}
            if scale == "frac" and vals.abs().max() <= 1.5:
                vals = vals * 100
            elif scale == "m2b":
                vals = vals / 1000
            for cat, v in zip(g["category"], vals):
                if pd.isna(v):
                    continue
                if dim_mode == "category":
                    m = re.search(r"(\d{4}):\s*(Q[1-4])", title)
                    period, ptype, flag, dim = (f"{m.group(1)}-{m.group(2)}" if m else None), "quarter", "A", cat.strip()
                elif dim_mode == "category_series_year":
                    period, ptype, flag, dim = series.strip(), "year", "A", cat.strip()
                else:
                    period, ptype, flag = parse_period(cat)
                    dim = ""
                    if ptype == "year" and cat.strip().endswith("*") and re.sub(r"\D", "", cat) in fc_years:
                        period, ptype, flag = f"{period}-YTD", "partial", "A"
                if not period:
                    continue
                out.append({"metric": metric, "dimension": dim, "period": period, "period_type": ptype,
                            "value": float(v), "unit": unit, "flag": flag, "is_forecast": int(flag in ("F", "E")),
                            "source_id": sid, "slide": int(slide), "vintage_date": vint[sid]})
            break
    df = pd.DataFrame(out)
    # A deck can show the same metric on two slides (e.g. history and forecast views): keep one per vintage
    df = df.sort_values(["metric", "dimension", "period", "vintage_date", "source_id", "is_forecast", "slide"])
    return df.drop_duplicates(["metric", "dimension", "period", "source_id"], keep="first")


# ---------------------------------------------------------------------------
# Ingest: FRED
# ---------------------------------------------------------------------------
def ingest_fred(offline: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cat = [], []
    for sid, label in FRED.items():
        f = CACHE / f"fred_{sid}.csv"
        status = "ok"
        try:
            if not offline or not f.exists():
                f.write_bytes(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"))
                time.sleep(6)
            d = pd.read_csv(f)
            d.columns = ["date", "value"]
            d["value"] = pd.to_numeric(d["value"], errors="coerce")
            d = d.dropna()
            rows += [{"series_id": sid, "date": r.date, "value": r.value} for r in d.itertuples()]
        except Exception as e:  # noqa: BLE001
            status = f"unavailable: {str(e)[:120]}"
        cat.append({"series_id": sid, "title": label, "url": f"https://fred.stlouisfed.org/series/{sid}",
                    "retrieved_at": NOW, "status": status})
        print(f"  FRED {sid}: {status}", file=sys.stderr)
    return pd.DataFrame(rows, columns=["series_id", "date", "value"]), pd.DataFrame(cat)


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------
def latest(obs: pd.DataFrame) -> pd.DataFrame:
    o = obs.sort_values(["vintage_date", "is_forecast"], ascending=[True, False])
    return o.drop_duplicates(["metric", "dimension", "period"], keep="last")


def fred_annual(fred: pd.DataFrame, sid: str, how: str = "mean") -> pd.Series:
    s = fred[fred["series_id"] == sid].copy()
    if s.empty:
        return pd.Series(dtype=float)
    s["date"] = pd.to_datetime(s["date"])
    g = s.set_index("date")["value"].resample("YE")
    a = g.mean() if how == "mean" else g.last()
    full = s.groupby(pd.to_datetime(s["date"]).dt.year).size()
    a.index = a.index.year
    return a[full.reindex(a.index).fillna(0) >= (12 if sid != "GDP" else 4)]


def fred_yoy(fred: pd.DataFrame, sid: str, freq: str) -> pd.Series:
    s = fred[fred["series_id"] == sid].copy()
    if s.empty:
        return pd.Series(dtype=float)
    s = s.set_index(pd.to_datetime(s["date"]))["value"]
    lag = 4 if freq == "Q" else 12
    return (s / s.shift(lag) - 1) * 100


def reconcile(obs: pd.DataFrame, fred: pd.DataFrame, fund: pd.DataFrame, fred_status: dict) -> pd.DataFrame:
    L = latest(obs[obs["is_forecast"] == 0])
    checks = []

    def add(metric, period, hv, ref_src, rv, tol, method, note=""):
        if pd.isna(hv):
            return
        if rv is None or pd.isna(rv):
            status, diff = "reference unavailable", np.nan
            sid = ref_src.replace("FRED ", "")
            if sid in fred_status and fred_status[sid] != "ok":
                note = f"{sid} not retrievable at build time ({fred_status[sid][:60]})"
        else:
            diff = hv - rv
            status = "match" if abs(diff) <= tol else ("close" if abs(diff) <= 2 * tol else "variance")
        checks.append({"metric": metric, "period": period, "deck_value": hv, "reference_source": ref_src,
                       "reference_value": rv, "difference": diff, "tolerance": tol, "status": status,
                       "method": method, "note": note})

    # 1. Deck macro series vs FRED (same public series, should match)
    def series_check(metric, sid, conv, tol, method, ptype="year"):
        d = L[(L["metric"] == metric) & (L["period_type"] == ptype)]
        ref = conv()
        for r in d.itertuples():
            key = int(r.period) if ptype == "year" else r.period
            add(metric, r.period, r.value, f"FRED {sid}", ref.get(key, np.nan) if len(ref) else None, tol, method)

    series_check("deck_housing_starts", "HOUST", lambda: fred_annual(fred, "HOUST") / 1000, 0.03,
                 "annual average of monthly SAAR, millions")
    series_check("deck_light_vehicle_sales", "ALTSALES", lambda: fred_annual(fred, "ALTSALES"), 0.3,
                 "annual average of monthly SAAR, millions")
    # Compare only against the exact series the deck cites (never a broader substitute)
    series_check("deck_insurance_carrier_employment", "CES5552410001", lambda: fred_annual(fred, "CES5552410001"), 15,
                 "12-month average, thousands (series cited on the slide)")

    med = fred_yoy(fred, "CPIMEDSL", "M")
    allc = fred_yoy(fred, "CPIAUCSL", "M")
    for metric, ref, sid in [("deck_medical_cpi_yoy", med, "CPIMEDSL"), ("deck_cpi_all_items_yoy", allc, "CPIAUCSL")]:
        d = L[(L["metric"] == metric)]
        rr = {k.strftime("%Y-%m"): v for k, v in ref.items()} if len(ref) else {}
        for r in d.itertuples():
            add(metric, r.period, r.value, f"FRED {sid}", rr.get(r.period, np.nan) if rr else None, 0.15,
                "y/y % change of monthly index")

    gdp = fred_yoy(fred, "GDP", "Q")
    gq = {f"{k.year}-Q{(k.month - 1) // 3 + 1}": v for k, v in gdp.items()} if len(gdp) else {}
    ga = fred_annual(fred, "GDP")
    ga = (ga / ga.shift(1) - 1) * 100 if len(ga) else ga
    d = L[L["metric"] == "deck_nominal_gdp_growth_yoy"]
    for r in d.itertuples():
        rv = gq.get(r.period) if r.period_type == "quarter" else (ga.get(int(r.period)) if len(ga) else None)
        add("deck_nominal_gdp_growth_yoy", r.period, r.value, "FRED GDP", rv, 0.5,
            "y/y % change (current vintage; BEA revisions expected)",
            "gap reflects BEA benchmark revisions since the deck's data vintage" if rv is not None and pd.notna(rv) and abs(r.value - rv) > 0.5 else "")

    # 2. Industry figures vs the SEC-filing universe (different populations: directional test)
    f = fund.copy()
    agg = f.groupby("fiscal_year")[["premiums_earned", "losses_incurred", "net_income"]].sum(min_count=10)
    ok = f.dropna(subset=["losses_incurred", "premiums_earned"])
    ok = ok[(ok["losses_incurred"] / ok["premiums_earned"]).between(0.35, 1.2)]

    def panel(y, col_fn):
        """Premium-weighted value for carriers present in both y-1 and y (balanced panel)."""
        a, b = ok[ok["fiscal_year"] == y - 1].set_index("ticker"), ok[ok["fiscal_year"] == y].set_index("ticker")
        both = a.index.intersection(b.index)
        return (col_fn(a.loc[both]), col_fn(b.loc[both]), len(both)) if len(both) >= 10 else (np.nan, np.nan, len(both))

    lr_fn = lambda d: d["losses_incurred"].sum() / d["premiums_earned"].sum() * 100  # noqa: E731
    pe_fn = lambda d: d["premiums_earned"].sum()  # noqa: E731
    ind_cr = L[(L["metric"] == "industry_combined_ratio") & (L["period_type"] == "year")].set_index("period")["value"]
    ind_g = L[(L["metric"] == "industry_dpw_growth") & (L["period_type"] == "year")].set_index("period")["value"]
    ind_ni = L[(L["metric"] == "industry_net_income") & (L["period_type"] == "year")].set_index("period")["value"]
    for y in range(2021, int(f["fiscal_year"].max()) + 1):
        ys = str(y)
        l0, l1, n_lr = panel(y, lr_fn)
        if ys in ind_cr and str(y - 1) in ind_cr and pd.notna(l1):
            d_ind, d_sec = ind_cr[ys] - ind_cr[str(y - 1)], l1 - l0
            checks.append({"metric": "industry_combined_ratio (direction)", "period": ys, "deck_value": d_ind,
                           "reference_source": "SEC 10-K universe loss ratio change", "reference_value": d_sec,
                           "difference": d_ind - d_sec, "tolerance": np.nan,
                           "status": "consistent" if np.sign(d_ind) == np.sign(d_sec) or abs(d_ind) < 0.5 else "divergent",
                           "method": "sign of y/y change (pp); balanced panel, premium-weighted; populations differ",
                           "note": f"{n_lr} carriers in both years"})
        p0, p1, n_p = panel(y, pe_fn)
        if ys in ind_g and pd.notna(p1):
            g_sec = (p1 / p0 - 1) * 100
            checks.append({"metric": "industry_dpw_growth (level)", "period": ys, "deck_value": ind_g[ys],
                           "reference_source": "SEC 10-K universe earned-premium growth", "reference_value": g_sec,
                           "difference": ind_g[ys] - g_sec, "tolerance": 3.0,
                           "status": "match" if abs(ind_g[ys] - g_sec) <= 3 else "variance",
                           "method": "written vs earned premium; balanced panel vs industry", "note": f"{n_p} carriers"})
        if ys in ind_ni and y in agg.index:
            share = agg.loc[y, "net_income"] / 1e9 / ind_ni[ys] * 100
            checks.append({"metric": "industry_net_income (coverage)", "period": ys, "deck_value": ind_ni[ys],
                           "reference_source": "SEC 10-K universe net income ($B)", "reference_value": agg.loc[y, "net_income"] / 1e9,
                           "difference": np.nan, "tolerance": np.nan, "status": "info",
                           "method": "universe share of industry net income", "note": f"universe = {share:.0f}% of industry"})
    # 3. Cross-vintage definition checks documented in deck footnotes
    ni = obs[(obs["metric"] == "industry_net_income") & (obs["period"] == "2023")].sort_values("vintage_date")
    if len(ni) >= 2:
        first, last_ = ni.iloc[0], ni.iloc[-1]
        adj = 50.0  # one-off realized gain excluded in early-2026 decks (per footnote)
        if abs(last_["value"] - first["value"]) > 5:
            d = last_["value"] - (first["value"] + adj)
            checks.append({"metric": "industry_net_income (vintage definitions)", "period": "2023",
                           "deck_value": last_["value"], "reference_source": f"{first['source_id']} + $50B footnoted gain",
                           "reference_value": first["value"] + adj, "difference": d, "tolerance": 2.5,
                           "status": "match" if abs(d) <= 2.5 else "variance",
                           "method": "earlier vintage excludes a $50B realized gain; later vintage includes it",
                           "note": "definition change, not an operating revision"})
    return pd.DataFrame(checks)


def revisions(obs: pd.DataFrame) -> pd.DataFrame:
    o = obs.sort_values("vintage_date")
    g = o.groupby(["metric", "dimension", "period"])
    r = g.agg(first_vintage=("vintage_date", "first"), first_value=("value", "first"),
              latest_vintage=("vintage_date", "last"), latest_value=("value", "last"),
              n_vintages=("source_id", "nunique"), min_value=("value", "min"), max_value=("value", "max")).reset_index()
    r["revision"] = r["latest_value"] - r["first_value"]
    r["max_spread"] = r["max_value"] - r["min_value"]
    return r[r["n_vintages"] > 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE sources(source_id TEXT PRIMARY KEY, publisher TEXT, title TEXT, url TEXT, pdf_url TEXT,
                     vintage_date TEXT, retrieved_at TEXT, sha256 TEXT, kind TEXT);
CREATE TABLE chart_series_raw(source_id TEXT REFERENCES sources, slide INT, slide_title TEXT, series TEXT,
                              category TEXT, value REAL, source_note TEXT);
CREATE TABLE metric_catalog(metric TEXT PRIMARY KEY, label TEXT, primary_sources TEXT);
CREATE TABLE observations(metric TEXT REFERENCES metric_catalog, dimension TEXT, period TEXT, period_type TEXT,
                          value REAL, unit TEXT, flag TEXT, is_forecast INT, source_id TEXT REFERENCES sources,
                          slide INT, vintage_date TEXT);
CREATE INDEX ix_obs ON observations(metric, dimension, period);
CREATE TABLE fred_catalog(series_id TEXT PRIMARY KEY, title TEXT, url TEXT, retrieved_at TEXT, status TEXT);
CREATE TABLE fred_observations(series_id TEXT REFERENCES fred_catalog, date TEXT, value REAL);
CREATE TABLE reconciliation(metric TEXT, period TEXT, deck_value REAL, reference_source TEXT, reference_value REAL,
                            difference REAL, tolerance REAL, status TEXT, method TEXT, note TEXT);
CREATE TABLE revisions(metric TEXT, dimension TEXT, period TEXT, first_vintage TEXT, first_value REAL,
                       latest_vintage TEXT, latest_value REAL, n_vintages INT, min_value REAL, max_value REAL,
                       revision REAL, max_spread REAL);
CREATE TABLE carrier_universe(ticker TEXT PRIMARY KEY, name TEXT, segment TEXT, cik INT);
CREATE TABLE carrier_fundamentals_raw(ticker TEXT, fiscal_year INT, shares_outstanding REAL, shares_as_of TEXT,
                                      equity REAL, preferred REAL, goodwill REAL, intangibles REAL, aoci REAL,
                                      net_income REAL, premiums_earned REAL, losses_incurred REAL, py_development REAL,
                                      net_investment_income REAL, buybacks REAL, dividends_paid REAL);
CREATE TABLE prices(date TEXT, ticker TEXT, close REAL, total_return_close REAL);
CREATE TABLE splits(ticker TEXT, date TEXT, ratio REAL);
CREATE TABLE build_log(built_at TEXT, item TEXT, detail TEXT);
CREATE VIEW industry_latest AS
  SELECT o.* FROM observations o
  JOIN (SELECT metric, dimension, period, MAX(vintage_date) AS v FROM observations GROUP BY 1,2,3) m
    ON o.metric = m.metric AND o.dimension = m.dimension AND o.period = m.period AND o.vintage_date = m.v;
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from universe import UNIVERSE

    print("Ingesting presentations…", file=sys.stderr)
    sources, raw = ingest_decks(a.offline)
    obs = normalize(raw, sources)
    print("Ingesting FRED…", file=sys.stderr)
    fred, fred_cat = ingest_fred(a.offline)
    fund = pd.read_csv(DATA / "fundamentals_raw.csv")
    rec = reconcile(obs, fred, fund, dict(zip(fred_cat["series_id"], fred_cat["status"])))
    rev = revisions(obs)

    tr = pd.read_csv(DATA / "prices_total_return.csv", index_col=0, parse_dates=True)
    cl = pd.read_csv(DATA / "prices_close.csv", index_col=0, parse_dates=True)
    px = (cl.stack().rename("close").to_frame().join(tr.stack().rename("total_return_close"), how="outer").reset_index())
    px.columns = ["date", "ticker", "close", "total_return_close"]
    px["date"] = px["date"].dt.strftime("%Y-%m-%d")

    tmp = DB.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    sources.to_sql("sources", con, if_exists="append", index=False)
    raw.to_sql("chart_series_raw", con, if_exists="append", index=False)
    pd.DataFrame([{"metric": k, "label": v[0], "primary_sources": v[1]} for k, v in CATALOG.items()]).to_sql(
        "metric_catalog", con, if_exists="append", index=False)
    obs.to_sql("observations", con, if_exists="append", index=False)
    fred_cat.to_sql("fred_catalog", con, if_exists="append", index=False)
    fred.to_sql("fred_observations", con, if_exists="append", index=False)
    rec.to_sql("reconciliation", con, if_exists="append", index=False)
    rev.to_sql("revisions", con, if_exists="append", index=False)
    pd.DataFrame(UNIVERSE).to_sql("carrier_universe", con, if_exists="append", index=False)
    fund.to_sql("carrier_fundamentals_raw", con, if_exists="append", index=False)
    px.to_sql("prices", con, if_exists="append", index=False)
    pd.read_csv(DATA / "splits.csv").to_sql("splits", con, if_exists="append", index=False)
    log = [("decks", f"{len(sources)} presentations, {len(raw)} chart points, {len(obs)} observations"),
           ("fred", f"{fred_cat['status'].eq('ok').sum()}/{len(fred_cat)} series"),
           ("reconciliation", rec["status"].value_counts().to_dict().__repr__() if len(rec) else "none")]
    con.executemany("INSERT INTO build_log VALUES (?,?,?)", [(NOW, k, v) for k, v in log])
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp.replace(DB)
    for k, v in log:
        print(f"{k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
