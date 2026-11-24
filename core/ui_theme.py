"""Gaya visual bersama untuk seluruh halaman Streamlit.

Tanpa ikon dan tanpa emoji. Hanya tipografi, jarak, kartu netral, dan grafik seragam.
"""
import math

import altair as alt
import pandas as pd
import streamlit as st

ACCENT = "#7aa2f7"
PALETTE = ["#7aa2f7", "#4fd1c5", "#e0af68", "#f7768e", "#bb9af7", "#9ece6a"]
MAX_POINTS = 4000

_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}

/* ikon tautan di samping judul dan tombol +/- pada input angka disembunyikan */
[data-testid="stHeaderActionElements"] {display: none !important;}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {display: none !important;}

.block-container {
    padding: 2.5rem 3rem 4rem 3rem;
    max-width: 100%;
}

h1 {font-size: 1.9rem !important; font-weight: 600 !important; letter-spacing: -0.02em;}
h2, h3, h4 {font-weight: 600 !important; letter-spacing: -0.01em;}
h4 {margin-top: 0.4rem;}

[data-testid="stSidebar"] {border-right: 1px solid #21262d;}
[data-testid="stSidebarNav"] a {border-radius: 8px;}

[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] p {color: #8b949e; font-size: 0.85rem; font-weight: 500;}
[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 600;}

[data-testid="stTabs"] button[role="tab"] {font-weight: 500; font-size: 0.95rem;}

[data-testid="stExpander"] {border-radius: 10px; border-color: #21262d;}
[data-testid="stDataFrame"] {border-radius: 10px;}

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: 8px;
    font-weight: 500;
}

hr {margin: 2rem 0 !important; border-color: #21262d !important;}

.app-muted {color: #8b949e; font-size: 0.9rem; margin: -0.5rem 0 1.5rem 0;}
.app-step-title {font-weight: 600; font-size: 1rem; margin-bottom: 0.15rem;}
.app-step-desc {color: #8b949e; font-size: 0.9rem;}

.app-steps {display: flex; gap: 8px; margin: 0.25rem 0 0.35rem 0;}
.app-step {
    flex: 1;
    font-size: 0.85rem;
    padding-top: 10px;
    border-top: 3px solid #21262d;
    color: #8b949e;
}
.app-step.done {border-color: #7aa2f7; color: #e6edf3;}
.app-step.active {border-color: #e0af68; color: #e6edf3; font-weight: 600;}
.app-steps-note {color: #8b949e; font-size: 0.85rem; margin-bottom: 1.25rem;}
</style>
"""


def inject_css():
    """Sisipkan CSS global. Dipanggil sekali per run dari app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="app-muted">{subtitle}</p>', unsafe_allow_html=True)


def section(title, caption=None):
    st.subheader(title)
    if caption:
        st.caption(caption)


def card_title(title, desc=None):
    """Judul kecil di dalam kartu (st.container with border)."""
    st.markdown(f'<div class="app-step-title">{title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="app-step-desc">{desc}</div>', unsafe_allow_html=True)


def steps_html(labels, done_flags):
    """Indikator langkah: langkah selesai biru, langkah berikutnya kuning, sisanya abu-abu."""
    first_todo = next((i for i, d in enumerate(done_flags) if not d), None)
    items = []
    for i, (lbl, d) in enumerate(zip(labels, done_flags)):
        cls = "done" if d else ("active" if i == first_todo else "")
        items.append(f'<div class="app-step {cls}">{lbl}</div>')
    n_done = sum(bool(d) for d in done_flags)
    return (
        f'<div class="app-steps">{"".join(items)}</div>'
        f'<div class="app-steps-note">{n_done} dari {len(labels)} langkah selesai</div>'
    )


# ---------------- Grafik dengan gaya seragam ----------------

def _style(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#21262d",
            domainColor="#30363d",
            tickColor="#30363d",
            labelColor="#8b949e",
            titleColor="#8b949e",
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(labelColor="#8b949e", titleColor="#8b949e", orient="top")
        .properties(height=260)
    )


def _downsample(df):
    if len(df) <= MAX_POINTS:
        return df
    step = math.ceil(len(df) / MAX_POINTS)
    return df.iloc[::step]


def bar_chart(df, x, y, color=None, offset=None):
    """Grafik batang. df dalam bentuk long (satu baris per nilai)."""
    enc = {
        "x": alt.X(x, sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        "y": alt.Y(y, title=None),
        "tooltip": [x, y] + ([color] if color else []),
    }
    if color:
        enc["color"] = alt.Color(color, title=None, scale=alt.Scale(range=PALETTE))
    if offset:
        enc["xOffset"] = alt.XOffset(offset)
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(**enc)
    st.altair_chart(_style(chart), width="stretch")


def line_chart(df, x, y, color=None):
    """Grafik garis. df dalam bentuk long."""
    df = _downsample(df)
    enc = {
        "x": alt.X(x, title=None),
        "y": alt.Y(y, title=None),
        "tooltip": [x, y] + ([color] if color else []),
    }
    if color:
        enc["color"] = alt.Color(color, title=None, scale=alt.Scale(range=PALETTE))
    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(**enc)
    st.altair_chart(_style(chart), width="stretch")
