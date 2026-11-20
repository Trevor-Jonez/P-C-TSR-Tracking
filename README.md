# US P&C Value Signals

A Streamlit dashboard that tracks value creation across 18 US property & casualty insurers and
surfaces statistically tested market signals. All data lives in one provenance-tracked SQLite
database, [`data/pc_market.db`](data/pc_market.db):

| Source | What it provides | How it is verified |
|---|---|---|
| **USC Risk & Uncertainty Management Center** presentations ([index](https://www.uscriskcenter.com/presentations/)) | Industry combined ratio, premiums, net income, ROE, CIAB commercial pricing, cat losses, personal auto & WC results, ratios by line | Chart data read from each deck's embedded series; every vintage kept with deck URL, date and SHA-256; revisions tracked |
| **FRED** (St. Louis Fed) | GDP, housing starts, vehicle sales, CPI, medical CPI, employment, 10-yr Treasury | Used to reconcile the public series shown in the decks (243 matches; every gap explained) |
| **SEC EDGAR XBRL** (10-K filings) | Carrier equity, goodwill, intangibles, AOCI, net income, premiums, losses, shares | Cross-checked against industry premium growth and combined-ratio direction (balanced panel) |
| **Yahoo Finance** | Daily prices and dividend-adjusted total returns | Live on load; snapshot stored in the database |

Key tables: `sources`, `chart_series_raw`, `observations` (all vintages), view `industry_latest`,
`revisions`, `reconciliation`, `fred_observations`, `carrier_fundamentals_raw`, `prices`.

![Dashboard preview](docs/preview.png)

## What's inside

| Section | What it shows |
|---|---|
| **Insight deck** | Three cards drawn from every finding that passes a significance test (p < 0.10). Each states the statistic, test, sample size and source, and expands to its evidence chart and data. **Shuffle** deals a new set weighted by statistical strength. |
| Industry | Commercial pricing regimes (tested breaks), combined ratio with forecast, cat-loss step change, premiums vs GDP, deck figures against our SEC data and FRED |
| Market signals | Cumulative excess return vs the S&P 500 with beta-adjusted shock days, change-point test, rolling beta, segment total returns, shock-day log |
| Value drivers | TSR split into TBV/share growth, P/TBV re-rating and cash yield; which driver explains the ranking; outliers and leave-one-out influence |
| Fundamentals | Year-by-year distributions (TBV growth, AOCI, RoTE, P/TBV) with significance stars; loss-ratio heatmap |
| Year by year | Each fiscal year named only from statistically significant shifts, plus a signal matrix |
| Data & methods | Reconciliation results, sources with links and checksums, every dataset (filterable, downloadable), the full database, methodology |

Sidebar settings (exclude statistical outliers, shock threshold, segments) recompute every test, chart and card.

## Refreshing the data

```bash
SEC_USER_AGENT="Your Name you@example.com" python scripts/build_fundamentals.py
```

```bash
python scripts/build_prices.py
```

```bash
python scripts/build_database.py
```

`build_database.py` downloads the presentations listed on the USC Risk Center page, pulls FRED, folds in the
SEC and price staging files, runs every reconciliation check and writes `data/pc_market.db`.
Use `--offline` to rebuild from cached downloads in `data/cache/`.

The app pulls live prices on load and falls back to the committed snapshot if Yahoo is unavailable.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Sign in at https://share.streamlit.io with GitHub.
2. **Create app** → this repo, branch `main`, main file `app.py` → **Deploy**.

---
Independent analysis. Not investment advice.
