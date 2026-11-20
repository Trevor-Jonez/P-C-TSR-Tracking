"""Visual theme. Palette from westmonroe.com's published CSS custom properties (yellow intentionally unused)."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY = "#070154"
BLUE = "#0047FF"
BLUE_HOVER = "#1433EB"
MAGENTA = "#F900D3"
RED = "#ED002D"
SLATE = "#50658E"
LIGHT = "#E8EEF8"
MID = "#D9D9D9"
INK = "#14123A"
WHITE = "#FFFFFF"
BLUE_TINT = "#8FA3FF"
NAVY_TINT = "#B9B6D6"

SERIES = [NAVY, BLUE, MAGENTA, SLATE, BLUE_TINT, NAVY_TINT, RED]
SEGMENT_COLORS = {
    "Personal": MAGENTA,
    "Personal & commercial": SLATE,
    "Commercial (standard)": NAVY,
    "Commercial (specialty)": BLUE,
}
# Decomposition blocks: (color, text color that passes contrast on that fill)
COMPONENTS = {
    "TBV/share growth": (NAVY, WHITE),
    "P/TBV re-rating": (BLUE, WHITE),
    "Cash yield": (MAGENTA, NAVY),
}
POS, NEG = BLUE, RED
FONT = "IBM Plex Sans, -apple-system, Segoe UI, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, monospace"


def register_plotly_template() -> None:
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family=FONT, color=INK, size=12),
        title=dict(font=dict(family=FONT, size=15, color=NAVY), x=0, xanchor="left", y=0.98, yanchor="top"),
        colorway=SERIES,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        margin=dict(l=8, r=16, t=48, b=64),
        xaxis=dict(gridcolor=LIGHT, zerolinecolor=MID, linecolor=MID, tickfont=dict(color=SLATE, size=11),
                   title=dict(font=dict(color=SLATE, size=12))),
        yaxis=dict(gridcolor=LIGHT, zerolinecolor=MID, linecolor=MID, tickfont=dict(color=SLATE, size=11),
                   title=dict(font=dict(color=SLATE, size=12))),
        # Legends sit under the plot so they can never collide with the title
        legend=dict(orientation="h", yref="container", y=0.005, yanchor="bottom", xanchor="left", x=0,
                    font=dict(color=INK, size=11)),
        hoverlabel=dict(bgcolor=NAVY, font=dict(family=FONT, color=WHITE, size=12), bordercolor=NAVY),
        bargap=0.28,
    )
    pio.templates["wm"] = tpl
    pio.templates.default = "plotly_white+wm"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame, button, input, select, textarea, label {{
  font-family: {FONT};
}}
.block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1380px; }}
h1, h2, h3, h4, h5 {{ color: {NAVY}; font-weight: 600; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.55rem; }}
h4 {{ font-size: 1.05rem; margin-top: .6rem; }}
p, li {{ color: {INK}; }}

/* Slim header bar */
.topbar {{ display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  border-bottom: 3px solid {NAVY}; padding: .2rem 0 .6rem; margin-bottom: .9rem; }}
.topbar .t {{ font-size: 1.45rem; font-weight: 700; color: {NAVY}; letter-spacing: -0.015em; }}
.topbar .t span {{ color: {BLUE}; }}
.topbar .m {{ font-family: {MONO}; font-size: .78rem; color: {SLATE}; }}

/* Insight cards */
.icard {{ background: {WHITE}; border: 1px solid {MID}; border-top: 4px solid var(--accent, {BLUE});
  border-radius: 3px; padding: .85rem 1rem .8rem; min-height: 262px; display: flex; flex-direction: column; gap: .4rem; }}
.icard .row {{ display: flex; justify-content: space-between; align-items: center; gap: .5rem; }}
.chip {{ font-family: {MONO}; font-size: .7rem; letter-spacing: .06em; text-transform: uppercase;
  color: {NAVY}; background: {LIGHT}; padding: .12rem .45rem; border-radius: 2px; white-space: nowrap; }}
.badge {{ font-family: {MONO}; font-size: .7rem; padding: .12rem .45rem; border-radius: 2px; white-space: nowrap; }}
.badge.strong {{ background: {NAVY}; color: {WHITE}; }}
.badge.sig {{ background: {BLUE}; color: {WHITE}; }}
.badge.sugg {{ background: {WHITE}; color: {SLATE}; border: 1px solid {SLATE}; }}
.icard .h {{ font-size: 1.02rem; font-weight: 600; color: {NAVY}; line-height: 1.3; }}
.icard .num {{ font-size: 1.75rem; font-weight: 700; color: var(--accent, {BLUE}); line-height: 1.05; }}
.icard .lbl {{ font-size: .8rem; color: {SLATE}; margin-top: -.2rem; }}
.icard .ev {{ font-size: .86rem; color: {INK}; line-height: 1.4; }}
.icard .src {{ font-family: {MONO}; font-size: .7rem; color: {SLATE}; margin-top: auto; padding-top: .3rem; border-top: 1px dashed {MID}; }}

.kpi {{ background: {LIGHT}; border-left: 4px solid {BLUE}; padding: .7rem .9rem; border-radius: 2px; height: 100%; }}
.kpi .v {{ font-size: 1.5rem; font-weight: 700; color: {NAVY}; line-height: 1.1; }}
.kpi .l {{ font-size: .85rem; color: {INK}; font-weight: 500; margin-top: .15rem; }}
.kpi .s {{ font-family: {MONO}; font-size: .74rem; color: {SLATE}; margin-top: .25rem; }}

.ycard {{ border-left: 4px solid var(--accent, {BLUE}); background: {LIGHT}; padding: .75rem .9rem; border-radius: 2px; min-height: 250px; }}
.ycard .y {{ font-family: {MONO}; font-size: .8rem; color: {SLATE}; }}
.ycard .h {{ font-weight: 600; color: {NAVY}; font-size: 1rem; margin: .15rem 0 .35rem; }}
.ycard ul {{ margin: 0 !important; padding-left: .95rem !important; }}
.ycard li {{ font-size: .84rem; color: {INK}; margin: 0 0 .25rem 0 !important; padding-left: 0 !important; line-height: 1.35; }}

.lede {{ color: {SLATE}; font-size: .98rem; max-width: 920px; margin-top: -.3rem; }}
.note {{ font-family: {MONO}; font-size: .74rem; color: {SLATE}; border-top: 1px solid {MID}; padding-top: .4rem; margin-top: .1rem; }}

.stTabs [data-baseweb="tab-list"], [role="tablist"] {{ gap: .25rem; border-bottom: 1px solid {MID}; }}
.stTabs [data-baseweb="tab"], [role="tab"] {{ font-family: {MONO}; font-size: .84rem; color: {SLATE}; padding: .55rem 1rem !important; }}
[role="tab"][aria-selected="true"] {{ color: {NAVY}; font-weight: 500; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {BLUE}; height: 3px; }}
.stButton>button, .stDownloadButton>button {{ background: {BLUE}; color: {WHITE}; border: 0; border-radius: 2px; font-weight: 500; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background: {BLUE_HOVER}; color: {WHITE}; }}
.stButton>button p {{ color: {WHITE}; }}
[data-testid="stExpander"] summary p {{ font-size: .85rem; color: {NAVY}; font-weight: 500; }}
[data-testid="stSidebar"] {{ background: {LIGHT}; }}

/* "So what" block: compact, executive, sits under every chart */
.sw {{ border-left: 3px solid var(--accent, {BLUE}); background: #F7F9FD; padding: .55rem .8rem .6rem; margin: .1rem 0 .2rem; border-radius: 2px; }}
.sw .tag {{ font-family: {MONO}; font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; color: var(--accent, {BLUE}); }}
.sw .hd {{ font-weight: 600; color: {NAVY}; font-size: .95rem; line-height: 1.3; margin: .1rem 0 .2rem; }}
.sw .bd {{ font-size: .86rem; color: {INK}; line-height: 1.45; }}
.sw .bd b {{ color: {NAVY}; }}
.roles {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin-top: .2rem; }}
@media (max-width: 900px) {{ .roles {{ grid-template-columns: 1fr; }} }}
.role {{ background: {WHITE}; border: 1px solid {MID}; border-top: 3px solid {NAVY}; padding: .5rem .65rem; border-radius: 2px; }}
.role .who {{ font-family: {MONO}; font-size: .7rem; letter-spacing: .08em; color: {SLATE}; text-transform: uppercase; }}
.role .why {{ font-size: .84rem; color: {INK}; margin: .2rem 0 .3rem; line-height: 1.4; }}
.role .do {{ font-size: .82rem; color: {NAVY}; line-height: 1.4; }}
.role .do:before {{ content: "→ "; color: {BLUE}; font-weight: 600; }}
.watch {{ font-size: .8rem; color: {SLATE}; margin-top: .35rem; }}
.watch b {{ color: {NAVY}; font-weight: 600; }}
/* Freshness stamp: deliberately quiet */
.fresh {{ font-family: {MONO}; font-size: .66rem; color: #8A97B2; text-align: right; margin: -.35rem 0 .3rem; letter-spacing: .02em; }}
.fresh:before {{ content: "● "; color: #9FD9B0; font-size: .6rem; }}
.fresh.stale:before {{ color: #F2B8C6; }}
.theme-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: .6rem; }}
@media (max-width: 1100px) {{ .theme-row {{ grid-template-columns: repeat(2, 1fr); }} }}
.tcard {{ border-top: 4px solid var(--accent, {BLUE}); background: {LIGHT}; padding: .7rem .8rem; border-radius: 2px; height: 100%; }}
.tcard .n {{ font-family: {MONO}; font-size: .7rem; color: {SLATE}; }}
.tcard .t {{ font-weight: 650; color: {NAVY}; font-size: .98rem; line-height: 1.25; margin: .15rem 0 .3rem; }}
.tcard .e {{ font-size: .82rem; color: {INK}; line-height: 1.4; }}
.tcard .k {{ font-size: 1.25rem; font-weight: 700; color: var(--accent, {BLUE}); margin-top: .3rem; }}
.story p {{ font-size: 1.0rem; line-height: 1.6; color: {INK}; max-width: 980px; }}
.story h3 {{ margin-top: 1.1rem; }}
.pill {{ display: inline-block; font-family: {MONO}; font-size: .7rem; padding: .05rem .4rem; border-radius: 10px; background: {LIGHT}; color: {NAVY}; margin-right: .25rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def strength(p: float) -> tuple[str, str]:
    if p < 0.01:
        return "strong", "p<0.01 · strong"
    if p < 0.05:
        return "sig", f"p={p:.2f} · significant"
    return "sugg", f"p={p:.2f} · suggestive"


def kpi(value: str, label: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div>'


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


THEME_ACCENT = {
    "The cycle has turned": RED,
    "Underwriting precision wins": BLUE,
    "Volatility is structural": MAGENTA,
    "Value is migrating along the chain": NAVY,
    "Cyber & AI: growth line, accumulation line": SLATE,
}


def sowhat(sw: dict, key: str) -> None:
    """Render a chart's executive 'so what': visible headline + body, role guidance on demand."""
    if not sw:
        return
    acc = THEME_ACCENT.get(sw.get("theme", ""), BLUE)
    st.markdown(
        f'<div class="sw" style="--accent:{acc}"><div class="tag">{sw.get("theme", "")}</div>'
        f'<div class="hd">{sw["headline"]}</div><div class="bd">{sw["body"]}</div></div>',
        unsafe_allow_html=True,
    )
    roles = sw.get("roles")
    if roles:
        with st.expander("What it means for the CFO, CUO and CEO", expanded=False):
            cells = "".join(
                f'<div class="role"><div class="who">{who}</div><div class="why">{r[0]}</div><div class="do">{r[1]}</div></div>'
                for who, r in roles.items()
            )
            watch = sw.get("watch")
            w = f'<div class="watch"><b>Watch:</b> {watch}</div>' if watch else ""
            st.markdown(f'<div class="roles">{cells}</div>{w}', unsafe_allow_html=True)


def fresh(text: str, stale: bool = False) -> None:
    st.markdown(f'<div class="fresh{" stale" if stale else ""}">{text}</div>', unsafe_allow_html=True)
