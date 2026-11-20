"""Sub-industry and market-lens analytics: carriers, reinsurers, retail brokers, wholesale & MGA, insurtech."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import db
from universe import UNIVERSE

ORDER = ["Carrier", "Reinsurer", "Retail broker", "Wholesale & MGA", "Insurtech"]


def company_table(tr: pd.DataFrame, cl: pd.DataFrame, fund_raw: pd.DataFrame) -> pd.DataFrame:
    u = pd.DataFrame(UNIVERSE).set_index("ticker")
    last = tr.dropna(how="all").index[-1]
    m = tr["^GSPC"].pct_change(fill_method=None)
    # latest fiscal year reported by most companies (relative threshold: robust to universe size and partial data)
    vc = fund_raw.dropna(subset=["net_income"])["fiscal_year"].value_counts()
    full = vc[vc >= 0.6 * vc.max()] if len(vc) else vc
    fy_last = int(full.index.max()) if len(full) else int(pd.Timestamp.today().year - 1)
    rows = []
    for t in u.index:
        if t not in tr:
            continue
        s = tr[t].dropna()
        if len(s) < 60:
            continue

        def ret(start):
            x = s[s.index >= start]
            return (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 20 else np.nan

        base = s[s.index >= pd.Timestamp("2020-12-31")]
        yrs = (base.index[-1] - base.index[0]).days / 365.25
        ann = ((base.iloc[-1] / base.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs >= 1 else np.nan
        x = s.pct_change(fill_method=None)
        j = pd.concat([x, m], axis=1).dropna().iloc[-252:]
        beta = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var() if len(j) > 120 else np.nan
        vol = x.iloc[-252:].std() * np.sqrt(252) * 100
        f = fund_raw[fund_raw["ticker"] == t].set_index("fiscal_year")
        ni = f["net_income"].get(fy_last, np.nan) if "net_income" in f else np.nan
        sh = f["shares_outstanding"].get(fy_last, np.nan) if "shares_outstanding" in f else np.nan
        px = cl[t].dropna().iloc[-1] if t in cl else np.nan
        pe = px * sh / ni if pd.notna(ni) and ni > 0 and pd.notna(sh) else np.nan
        rv = f["revenue"] if "revenue" in f else pd.Series(dtype=float)
        g = (rv.get(fy_last) / rv.get(fy_last - 1) - 1) * 100 if pd.notna(rv.get(fy_last)) and pd.notna(rv.get(fy_last - 1)) else np.nan
        g3 = ((rv.get(fy_last) / rv.get(fy_last - 3)) ** (1 / 3) - 1) * 100 if pd.notna(rv.get(fy_last)) and pd.notna(rv.get(fy_last - 3)) and rv.get(fy_last - 3) > 0 else np.nan
        rows.append({
            "ticker": t, "name": u.loc[t, "name"], "sub_industry": u.loc[t, "sub_industry"], "lens": u.loc[t, "lens"],
            "market": u.loc[t, "market"], "listed_since": s.index[0].date() if s.index[0] > pd.Timestamp("2020-01-02") else None,
            "ret_1y": ret(last - pd.DateOffset(years=1)), "ret_ytd": ret(pd.Timestamp(f"{last.year - 1}-12-31")),
            "ann_since_2020": ann, "beta_1y": beta, "vol_1y": vol, "pe": pe, f"rev_growth_{fy_last}": g, "rev_cagr_3y": g3,
            "market_cap_bn": px * sh / 1e9 if pd.notna(sh) else np.nan,
        })
    return pd.DataFrame(rows)


def sub_index(tr: pd.DataFrame, start: str = "2020-12-31") -> pd.DataFrame:
    """Equal-weight total-return index per sub-industry (members join when they list)."""
    u = pd.DataFrame(UNIVERSE)
    out = {}
    for sub in ORDER:
        ts = [t for t in u.loc[u["sub_industry"] == sub, "ticker"] if t in tr]
        r = tr.loc[tr.index >= start, ts].pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0)
        out[sub] = (1 + r).cumprod() * 100
    return pd.DataFrame(out)


def tests(ct: pd.DataFrame) -> dict:
    out = {}
    for col in ["ret_1y", "beta_1y", "pe"]:
        groups = [g.dropna().values for _, g in ct.groupby("sub_industry")[col] if g.notna().sum() >= 2]
        if len(groups) >= 3:
            out[col] = stats.kruskal(*groups).pvalue
    a = ct[ct["sub_industry"] == "Retail broker"]["pe"].dropna()
    b = ct[ct["sub_industry"] == "Carrier"]["pe"].dropna()
    if len(a) >= 3 and len(b) >= 3:
        out["broker_vs_carrier_pe"] = stats.mannwhitneyu(a, b, alternative="greater").pvalue
    es = ct[(ct["sub_industry"] == "Carrier") & (ct["market"] == "Non-admitted (E&S)")]
    ad = ct[(ct["sub_industry"] == "Carrier") & (ct["market"] == "Admitted")]
    gcol = [c for c in ct.columns if c.startswith("rev_growth_")][0]
    if len(es) >= 3 and len(ad) >= 3:
        out["es_vs_admitted_growth"] = stats.mannwhitneyu(es[gcol].dropna(), ad[gcol].dropna(), alternative="greater").pvalue
        out["es_growth_median"] = es[gcol].median()
        out["admitted_growth_median"] = ad[gcol].median()
    out["growth_col"] = gcol
    return out


def facts(domain: str) -> pd.DataFrame:
    return db.q("SELECT * FROM context_facts WHERE domain = ? ORDER BY metric, period", (domain,))
