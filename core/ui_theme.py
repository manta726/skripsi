"""Gaya visual bersama untuk seluruh halaman Streamlit.

Tanpa ikon dan tanpa emoji. Hanya tipografi, jarak, dan kartu netral.
"""
import streamlit as st

_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}

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
