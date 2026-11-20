# US P&C Value Signals

A Streamlit dashboard that tracks value creation across 18 US property & casualty insurers and
surfaces statistically tested market signals, built entirely from primary data:

- **SEC EDGAR XBRL** (10-K filings): equity, goodwill, intangibles, AOCI, net income, earned premium,
  losses incurred, share counts
- **Yahoo Finance**: daily prices and dividend-adjusted total returns

![Dashboard preview](docs/preview.png)

## What's inside

| Section | What it shows |
|---|---|
| **Insight deck** | Three cards drawn from every finding that passes a significance test (p < 0.10). Each states the statistic, test, sample size and source, and expands to its evidence chart and data. **Shuffle** deals a new set weighted by statistical strength. |
| Market signals | Cumulative excess return vs the S&P 500 with beta-adjusted shock days, change-point test, rolling beta, segment total returns, shock-day log |
| Value drivers | TSR split into TBV/share growth, P/TBV re-rating and cash yield; which driver explains the ranking; outliers and leave-one-out influence |
| Fundamentals | Year-by-year distributions (TBV growth, AOCI, RoTE, P/TBV) with significance stars; loss-ratio heatmap |
| Year by year | Each fiscal year named only from statistically significant shifts, plus a signal matrix |
| Data & methods | Every dataset, filterable and downloadable, with full methodology |

Sidebar settings (exclude statistical outliers, shock threshold, segments) recompute every test, chart and card.

## Refreshing the data

```bash
SEC_USER_AGENT="Your Name you@example.com" python scripts/build_fundamentals.py
```

```bash
python scripts/build_prices.py
```

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
