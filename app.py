import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="US P&C Value Creators & BCG TSR Engine", layout="wide")

st.title("US P&C Insurance Value Creators: Live TSR & BCG Gold-Standard Tracker")
st.caption("Benchmarked to BCG's 2026 Value Thesis: Softer Rates, Structural Risks & Platform AI")

# 1. Live ticker & index ingestion
TICKERS = {
    "S&P 500 P&C Index": "^SP500-40301040",
    "SPDR S&P Insurance ETF (KIE, P&C proxy)": "KIE",
    "S&P 500 Benchmark": "^GSPC",
    "Progressive (Personal Outlier)": "PGR",
    "Chubb (Global Commercial/Specialty)": "CB",
    "Travelers (Commercial/Personal)": "TRV",
    "Allstate (Personal Lines)": "ALL",
    "Hartford (Commercial/Personal)": "HIG",
    "W. R. Berkley (Specialty/E&S)": "WRB",
    "Arch Capital (Specialty/Reinsurance)": "ACGL",
}
TICKER_TO_NAME = {v: k for k, v in TICKERS.items()}
START_DATE = "2020-12-31"


@st.cache_data(ttl=3600, show_spinner="Fetching live prices from Yahoo Finance...")
def fetch_prices() -> pd.DataFrame:
    """Daily total-return (dividend-adjusted) closes, columns named by entity."""
    raw = yf.download(
        list(TICKERS.values()),
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    closes = raw["Close"]
    if isinstance(closes, pd.Series):  # single ticker edge case
        closes = closes.to_frame()
    # yfinance sorts columns alphabetically by ticker, so map by ticker, never by position
    closes = closes.rename(columns=TICKER_TO_NAME)
    return closes.sort_index()


def annualized_return(series: pd.Series, start: pd.Timestamp) -> float:
    """Annualized return from the last close on/before `start` to the latest close."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    start_px = s.asof(start)
    if pd.isna(start_px):
        return float("nan")
    start_dt = s.index[s.index <= start][-1]
    end_dt, end_px = s.index[-1], s.iloc[-1]
    years = (end_dt - start_dt).days / 365.25
    if years <= 0:
        return float("nan")
    return ((end_px / start_px) ** (1 / years) - 1) * 100


try:
    prices = fetch_prices()
except Exception as exc:  # network / rate-limit failures should not crash the page
    st.error(
        "Could not load live market data from Yahoo Finance right now "
        f"({type(exc).__name__}). This is usually a temporary rate limit — "
        "please refresh in a few minutes."
    )
    st.stop()

missing = [name for name in TICKERS if name not in prices.columns or prices[name].dropna().empty]
if missing:
    st.warning("No data returned for: " + ", ".join(missing) + ". They are excluded from the charts below.")
prices = prices.drop(columns=[m for m in missing if m in prices.columns])

last_date = prices.dropna(how="all").index[-1]

# 2. Annualized TSR
st.subheader("1. Live Annualized TSR vs. BCG 2021–2025 Benchmarks")
window = st.radio(
    "Measurement window",
    ["Trailing 5 years", "Since year-end 2020"],
    horizontal=True,
)
if window == "Trailing 5 years":
    window_start = last_date - pd.DateOffset(years=5)
else:
    window_start = pd.Timestamp(START_DATE)

tsr = {name: annualized_return(prices[name], window_start) for name in prices.columns}
df_cagr = (
    pd.DataFrame({"Entity": list(tsr.keys()), "Live_Annualized_TSR_%": list(tsr.values())})
    .dropna()
    .sort_values(by="Live_Annualized_TSR_%", ascending=False)
)
st.caption(
    f"Dividend-adjusted price data from {window_start:%b %d, %Y} to {last_date:%b %d, %Y} "
    "(latest available close)."
)

col1, col2 = st.columns([1.2, 1])
with col1:
    fig_bar = px.bar(
        df_cagr,
        x="Live_Annualized_TSR_%",
        y="Entity",
        orientation="h",
        title=f"Live Annualized TSR — {window} (%)",
        text=df_cagr["Live_Annualized_TSR_%"].map(lambda x: f"{x:.1f}%"),
        color="Live_Annualized_TSR_%",
        color_continuous_scale="Tealgrn",
    )
    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_bar.add_vline(x=18.0, line_dash="dash", line_color="red", annotation_text="BCG US P&C 2021–2025 (18%)")
    st.plotly_chart(fig_bar, width="stretch")

with col2:
    st.markdown("#### Next Statutory & Industry Data Drop Countdown")
    drops = pd.DataFrame([
        {"Dataset": "CIAB Q3/Q4 Commercial Rate Survey", "Expected Window": "Dec 2026 / Feb 2027", "Primary Signal": "Soft-market rate deceleration (-1.2% Q1 '26)"},
        {"Dataset": "FY 2026 Rolling TSR Window Close", "Expected Window": "Dec 31, 2026", "Primary Signal": "Shifts 5-yr window to 2022–2026 (drops 2021)"},
        {"Dataset": "NAIC / III FY26 Statutory Combined Ratio", "Expected Window": "March 2027", "Primary Signal": "Tests if 92% Auto / 88% Home CRs hold"},
        {"Dataset": "BCG 2027 Insurance Value Creators", "Expected Window": "April & June 2027", "Primary Signal": "Updated TBV, Cash Flow & Multiple splits"},
    ])
    st.dataframe(drops, hide_index=True, width="stretch")
    st.caption("Release windows and signals are manually maintained estimates, not live data.")

# 3. BCG Exhibit 1 & 4 TSR decomposition emulator
st.subheader("2. BCG Exhibit 1 & 4 Emulator: TSR Driver Split (2021–2025 Gold Standard)")
bcg_decomp = pd.DataFrame([
    {"Cohort": "US P&C Composite", "TBV Growth": 13.0, "Cash Flow Yield": 4.2, "Multiple Change": 0.8},
    {"Cohort": "US Commercial Lines", "TBV Growth": 13.2, "Cash Flow Yield": 4.3, "Multiple Change": 0.5},
    {"Cohort": "US Personal Lines (Incl. PGR)", "TBV Growth": 12.0, "Cash Flow Yield": 3.5, "Multiple Change": 1.5},
    {"Cohort": "US Personal Lines (Excl. PGR)", "TBV Growth": 9.2, "Cash Flow Yield": 3.3, "Multiple Change": 0.5},
    {"Cohort": "Global Reinsurance", "TBV Growth": 13.8, "Cash Flow Yield": 4.5, "Multiple Change": 0.8},
    {"Cohort": "Global Insurance Avg", "TBV Growth": 9.5, "Cash Flow Yield": 4.8, "Multiple Change": 0.7},
])

fig_stack = go.Figure()
for col, color in [("TBV Growth", "#006644"), ("Cash Flow Yield", "#2ECC71"), ("Multiple Change", "#A3E4D7")]:
    fig_stack.add_trace(go.Bar(name=col, x=bcg_decomp["Cohort"], y=bcg_decomp[col], marker_color=color))
fig_stack.update_layout(barmode="stack", title="BCG 2021–2025 TSR Decomposition (%)", yaxis_title="Annualized Contribution (%)")
st.plotly_chart(fig_stack, width="stretch")
st.caption(
    "Source: figures manually transcribed from BCG's Insurance Value Creators research; "
    "illustrative and not independently verified. Check the original report before citing."
)
