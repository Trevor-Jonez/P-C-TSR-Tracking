"""Visual theme: palette taken from westmonroe.com's published CSS custom properties."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY = "#070154"        # --grounded-blue
BLUE = "#0047FF"        # --highlight-blue
BLUE_HOVER = "#1433EB"  # --highlight-blue-hover
MAGENTA = "#F900D3"     # --highlight-magenta
YELLOW = "#F6EB21"      # --highlight-yellow (use on navy only; low contrast on white)
RED = "#ED002D"         # --accent-red
SLATE = "#50658E"       # --dark-gray
LIGHT = "#E8EEF8"       # --light-gray
MID = "#D9D9D9"         # --medium-gray
INK = "#1B1B33"
WHITE = "#FFFFFF"

# Categorical order: navy, blue, magenta, slate, then lighter tints
SERIES = [NAVY, BLUE, MAGENTA, SLATE, "#8FA3FF", "#B9B6D6", "#FF8AEC", "#9AA8C3"]
# TSR building blocks used consistently on every decomposition chart
COMPONENT_COLORS = {
    "Growth of TBV": NAVY,
    "Change in P/TBV multiple": BLUE,
    "Cash flow contribution": MAGENTA,
}
POS, NEG = BLUE, RED
FONT = "IBM Plex Sans, -apple-system, Segoe UI, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, monospace"


def register_plotly_template() -> None:
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family=FONT, color=INK, size=13),
        title=dict(font=dict(family=FONT, size=16, color=NAVY), x=0, xanchor="left"),
        colorway=SERIES,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        margin=dict(l=10, r=10, t=56, b=10),
        xaxis=dict(gridcolor=LIGHT, zerolinecolor=MID, linecolor=MID, tickfont=dict(color=SLATE)),
        yaxis=dict(gridcolor=LIGHT, zerolinecolor=MID, linecolor=MID, tickfont=dict(color=SLATE)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=SLATE)),
        hoverlabel=dict(bgcolor=NAVY, font=dict(family=FONT, color=WHITE), bordercolor=NAVY),
        bargap=0.28,
    )
    pio.templates["wm"] = tpl
    pio.templates.default = "plotly_white+wm"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame, button, input, select, textarea {{
  font-family: {FONT};
}}
.block-container {{ padding-top: 1.2rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ color: {NAVY}; font-weight: 600; letter-spacing: -0.01em; }}

.hero {{
  background: {NAVY}; color: {WHITE}; border-radius: 4px;
  padding: 2.2rem 2.4rem 2rem; margin-bottom: 1.4rem; position: relative; overflow: hidden;
}}
.hero:after {{
  content: ""; position: absolute; right: -80px; top: -80px; width: 280px; height: 280px;
  border-radius: 50%; background: radial-gradient(circle, {BLUE} 0%, rgba(0,71,255,0) 70%); opacity: .55;
}}
.hero .eyebrow {{ font-family: {MONO}; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: {YELLOW}; }}
.hero h1 {{ color: {WHITE}; font-size: 2.2rem; line-height: 1.15; margin: .5rem 0 .6rem; font-weight: 600; }}
.hero p {{ color: #C9CBE6; max-width: 820px; font-size: 1.02rem; margin: 0; }}

.chapter {{ font-family: {MONO}; color: {BLUE}; font-size: .8rem; letter-spacing: .14em; text-transform: uppercase; margin-bottom: -.4rem; }}
.lede {{ color: {SLATE}; font-size: 1.02rem; max-width: 900px; }}

.kpi {{ background: {LIGHT}; border-left: 4px solid {BLUE}; border-radius: 2px; padding: .9rem 1rem; height: 100%; }}
.kpi.navy {{ background: {NAVY}; border-left-color: {YELLOW}; }}
.kpi.navy .v, .kpi.navy .l {{ color: {WHITE}; }}
.kpi.navy .s {{ color: #B9BCE0; }}
.kpi.alert {{ border-left-color: {RED}; }}
.kpi.mag {{ border-left-color: {MAGENTA}; }}
.kpi .v {{ font-size: 1.9rem; font-weight: 600; color: {NAVY}; line-height: 1.1; }}
.kpi .l {{ font-size: .88rem; color: {INK}; font-weight: 500; margin-top: .25rem; }}
.kpi .s {{ font-family: {MONO}; font-size: .72rem; color: {SLATE}; margin-top: .35rem; }}

.theme-card {{ border-top: 3px solid {BLUE}; padding: .7rem .2rem 0; }}
.theme-card b {{ color: {NAVY}; display: block; margin-bottom: .25rem; }}
.theme-card span {{ color: {SLATE}; font-size: .92rem; }}

.source {{ font-family: {MONO}; font-size: .72rem; color: {SLATE}; border-top: 1px solid {MID}; padding-top: .4rem; margin-top: .2rem; }}

.stTabs [data-baseweb="tab-list"] {{ gap: .2rem; border-bottom: 1px solid {MID}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {MONO}; font-size: .8rem; letter-spacing: .04em; padding: .6rem .9rem; color: {SLATE}; }}
.stTabs [aria-selected="true"] {{ color: {NAVY}; font-weight: 500; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {BLUE}; height: 3px; }}
.stButton>button, .stDownloadButton>button {{ background: {BLUE}; color: {WHITE}; border: 0; border-radius: 2px; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background: {BLUE_HOVER}; color: {WHITE}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(value: str, label: str, sub: str = "", variant: str = "") -> str:
    return (
        f'<div class="kpi {variant}"><div class="v">{value}</div>'
        f'<div class="l">{label}</div><div class="s">{sub}</div></div>'
    )


def chapter(num: str, title: str, lede: str) -> None:
    st.markdown(f'<div class="chapter">Chapter {num}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


def source(text: str) -> None:
    st.markdown(f'<div class="source">{text}</div>', unsafe_allow_html=True)
