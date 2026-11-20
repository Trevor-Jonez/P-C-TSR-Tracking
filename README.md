# US P&C Insurance Value Creators: Live TSR Tracker

A Streamlit dashboard that tracks live total shareholder return (TSR) for major US
property & casualty insurers against BCG's 2021–2025 Insurance Value Creators benchmarks.

## What it shows

1. **Live annualized TSR** for PGR, CB, TRV, ALL, HIG, WRB and ACGL, plus the S&P 500,
   the S&P 500 P&C index and the KIE insurance ETF. Choose a trailing 5-year window or
   since year-end 2020. Prices are dividend-adjusted from Yahoo Finance and cached for an hour.
2. **Upcoming data releases** that will move the benchmark (manually maintained).
3. **BCG TSR decomposition**: TBV growth, cash-flow yield and multiple change by cohort
   (figures transcribed manually from BCG; verify against the source report before citing).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Sign in at https://share.streamlit.io with GitHub.
2. **Create app** → choose this repo, branch `main`, main file `app.py` → **Deploy**.

Every push to `main` redeploys automatically.
