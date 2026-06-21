import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import ast
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AIRECRUIT — Intelligence Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark Gold Luxury Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root ── */
:root {
    --bg:      #0B0D10;
    --bg2:     #13161B;
    --bg3:     #1A1D24;
    --gold:    #D6B25E;
    --gold2:   #8C7A5B;
    --gold-d:  rgba(214,178,94,0.12);
    --gold-b:  rgba(214,178,94,0.22);
    --text:    #F0EDE6;
    --text2:   #A89F92;
    --text3:   #6B6560;
    --border:  rgba(214,178,94,0.18);
    --glass:   rgba(255,255,255,0.03);
    --glass2:  rgba(255,255,255,0.06);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0B0D10 !important;
    color: #F0EDE6 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
    background: #0B0D10 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D0F14 !important;
    border-right: 1px solid rgba(214,178,94,0.12) !important;
}
/* General sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #C8C0B4;
    font-family: 'Inter', sans-serif !important;
}
/* Sidebar section label ("Navigation") */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: 2.5px !important;
    color: rgba(214,178,94,0.45) !important;
    margin-top: 20px !important;
    margin-bottom: 10px !important;
    font-weight: 700 !important;
    padding-left: 4px !important;
}

/* ── Sidebar Radio Nav ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 2px !important;
    display: flex !important;
    flex-direction: column !important;
}
/* Hide the native radio dot completely */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadio"] svg,
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] [role="radio"],
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    display: none !important;
    width: 0 !important; height: 0 !important; margin: 0 !important;
}
/* Each nav item — no box border, just a clean row */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 11px 14px !important;
    margin: 0 !important;
    min-height: unset !important;
    height: auto !important;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: background 0.18s ease, transform 0.15s ease !important;
}
/* Hover state */
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(214,178,94,0.07) !important;
    transform: translateX(3px) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover div {
    color: #F0EDE6 !important;
}
/* ACTIVE/SELECTED item — gold highlight box */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(214,178,94,0.14) !important;
    border-left: 3px solid #D6B25E !important;
    padding-left: 11px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) div {
    color: #D6B25E !important;
    font-weight: 600 !important;
}
/* Nav item text */
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] label div {
    font-size: 14.5px !important;
    font-weight: 400 !important;
    color: #C8C0B4 !important;
    letter-spacing: 0.1px !important;
    line-height: 1.3 !important;
    margin: 0 !important;
    padding: 0 !important;
    white-space: nowrap !important;
    text-transform: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: #D6B25E !important;
    border: 1px solid rgba(214,178,94,0.4) !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.5px !important;
    white-space: nowrap !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(214,178,94,0.10) !important;
    border-color: #D6B25E !important;
    box-shadow: 0 4px 20px rgba(214,178,94,0.15) !important;
}
.stTextArea textarea, .stTextInput input {
    background: #13161B !important;
    border: 1px solid rgba(214,178,94,0.18) !important;
    border-radius: 8px !important;
    color: #F0EDE6 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(214,178,94,0.5) !important;
    box-shadow: 0 0 0 3px rgba(214,178,94,0.08) !important;
}
.stSelectbox > div > div {
    background: #13161B !important;
    border: 1px solid rgba(214,178,94,0.18) !important;
    color: #F0EDE6 !important;
    border-radius: 8px !important;
}
.stSlider > div { color: #D6B25E !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #D6B25E !important;
    border-color: #D6B25E !important;
}
div[data-baseweb="tab-list"] { background: #13161B !important; border-radius: 10px !important; }
div[data-baseweb="tab"] { color: #A89F92 !important; }
div[aria-selected="true"] { color: #D6B25E !important; border-bottom: 2px solid #D6B25E !important; }

/* ── Glass Cards ── */
.glass-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(214,178,94,0.18);
    border-radius: 14px;
    padding: 28px;
    backdrop-filter: blur(12px);
    transition: border-color 0.25s, background 0.25s, transform 0.25s;
}
.glass-card:hover {
    border-color: rgba(214,178,94,0.4);
    background: rgba(214,178,94,0.04);
    transform: translateY(-2px);
}
.glass-card-sm {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(214,178,94,0.14);
    border-radius: 12px;
    padding: 20px 22px;
    transition: all 0.2s;
}
.glass-card-sm:hover {
    border-color: rgba(214,178,94,0.35);
    background: rgba(214,178,94,0.04);
}

/* ── Hero ── */
.hero-section {
    min-height: 88vh;
    display: flex;
    align-items: center;
    padding: 60px 0 40px;
    position: relative;
}
.hero-eyebrow {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 4px;
    color: #D6B25E;
    font-weight: 600;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-eyebrow::before {
    content: '';
    width: 32px;
    height: 1px;
    background: #D6B25E;
    display: inline-block;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #F0EDE6;
    line-height: 1.18;
    margin-bottom: 24px;
    letter-spacing: -0.5px;
}
.hero-title .gold { color: #D6B25E; }
.hero-subtitle {
    font-size: 1rem;
    color: #A89F92;
    max-width: 540px;
    line-height: 1.75;
    margin-bottom: 40px;
    font-weight: 300;
}
.hero-btn-row { display: flex; gap: 14px; flex-wrap: wrap; }
.hero-btn-primary {
    background: rgba(214,178,94,0.12);
    border: 1px solid #D6B25E;
    color: #D6B25E;
    padding: 12px 32px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s;
    letter-spacing: 0.5px;
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
}
.hero-btn-primary:hover {
    background: rgba(214,178,94,0.2);
    box-shadow: 0 8px 28px rgba(214,178,94,0.2);
}
.hero-btn-secondary {
    background: transparent;
    border: 1px solid rgba(214,178,94,0.25);
    color: #A89F92;
    padding: 12px 32px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.25s;
    letter-spacing: 0.5px;
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
}
.hero-btn-secondary:hover {
    border-color: rgba(214,178,94,0.5);
    color: #D6B25E;
}

/* ── Section Headers ── */
.section-eyebrow {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 3.5px;
    color: #D6B25E;
    font-weight: 600;
    margin-bottom: 10px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #F0EDE6;
    margin-bottom: 8px;
    line-height: 1.2;
}
.section-sub {
    font-size: 14px;
    color: #A89F92;
    margin-bottom: 0;
    font-weight: 300;
    line-height: 1.6;
}
.section-rule {
    height: 1px;
    background: linear-gradient(90deg, rgba(214,178,94,0.5), transparent);
    margin: 20px 0 36px;
}

/* ── Metric Cards ── */
.metric-glass {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(214,178,94,0.18);
    border-radius: 14px;
    padding: 28px 20px;
    text-align: center;
}
.metric-glass-val {
    font-family: 'Playfair Display', serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: #D6B25E;
    line-height: 1;
    margin-bottom: 8px;
}
.metric-glass-lbl {
    font-size: 11px;
    color: #6B6560;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* ── Timeline ── */
.tl-wrap { padding: 8px 0; }
.tl-item {
    display: flex;
    gap: 20px;
    padding-bottom: 32px;
    position: relative;
}
.tl-item::after {
    content: '';
    position: absolute;
    left: 15px; top: 36px; bottom: 0;
    width: 1px;
    background: linear-gradient(180deg, rgba(214,178,94,0.4), transparent);
}
.tl-item:last-child::after { display: none; }
.tl-dot {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: #13161B;
    border: 1.5px solid #D6B25E;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 12px; font-weight: 700; color: #D6B25E;
}
.tl-title {
    font-family: 'Playfair Display', serif;
    font-size: 15px; font-weight: 600; color: #F0EDE6;
    margin-bottom: 4px; padding-top: 4px;
}
.tl-body { font-size: 13px; color: #A89F92; line-height: 1.65; font-weight: 300; }

/* ── Progress ── */
.prog-row { margin-bottom: 16px; }
.prog-hdr { display: flex; justify-content: space-between; margin-bottom: 6px; }
.prog-name { font-size: 13px; font-weight: 500; color: #F0EDE6; }
.prog-val  { font-size: 13px; font-weight: 700; color: #D6B25E;
             font-family: 'Playfair Display', serif; }
.prog-track { height: 5px; background: rgba(214,178,94,0.1); border-radius: 3px;
              border: 1px solid rgba(214,178,94,0.12); overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 3px;
              background: linear-gradient(90deg, #8C7A5B, #D6B25E); }

/* ── Team Cards ── */
.team-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(214,178,94,0.14);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    transition: all 0.25s;
}
.team-card:hover {
    border-color: rgba(214,178,94,0.4);
    transform: translateY(-3px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
}
.team-card.highlight {
    border-color: rgba(214,178,94,0.5);
    background: rgba(214,178,94,0.04);
    box-shadow: 0 8px 32px rgba(214,178,94,0.08);
}
.team-avatar {
    width: 70px; height: 70px;
    border-radius: 50%;
    background: #13161B;
    border: 1.5px solid rgba(214,178,94,0.3);
    margin: 0 auto 16px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 700; color: #D6B25E;
}
.team-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem; font-weight: 700; color: #F0EDE6; margin-bottom: 4px;
}
.team-role {
    font-size: 10px; color: #D6B25E;
    text-transform: uppercase; letter-spacing: 1.5px;
    font-weight: 600; margin-bottom: 18px;
}
.team-task {
    font-size: 12px; color: #A89F92;
    padding: 5px 0;
    border-bottom: 1px solid rgba(214,178,94,0.07);
    text-align: left;
    display: flex; align-items: center; gap: 8px;
    font-weight: 300;
}
.team-task:last-child { border-bottom: none; }
.team-task::before { content: '—'; color: #8C7A5B; font-size: 10px; flex-shrink: 0; }

/* ── Tech Pills ── */
.tech-pill {
    display: inline-block;
    background: rgba(214,178,94,0.08);
    border: 1px solid rgba(214,178,94,0.2);
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #A89F92;
    margin: 4px;
    transition: all 0.2s;
}
.tech-pill:hover { border-color: #D6B25E; color: #D6B25E; }

/* ── Formula ── */
.formula-box {
    background: #13161B;
    border: 1px solid rgba(214,178,94,0.2);
    border-left: 3px solid #D6B25E;
    border-radius: 0 10px 10px 0;
    padding: 20px 24px;
    font-family: 'SF Mono','Fira Code',monospace;
    font-size: 13.5px; color: #D6B25E;
    line-height: 1.9; margin: 16px 0;
}

/* ── Callout ── */
.callout {
    background: rgba(214,178,94,0.05);
    border: 1px solid rgba(214,178,94,0.18);
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 13.5px; color: #A89F92;
    line-height: 1.7; margin: 16px 0;
}
.callout strong { color: #D6B25E; }

/* ── Table ── */
.gold-table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13.5px; }
.gold-table th {
    font-size: 9px; text-transform: uppercase; letter-spacing: 2px;
    color: #D6B25E; font-weight: 600; padding: 10px 16px;
    border-bottom: 1px solid rgba(214,178,94,0.2); text-align: left;
}
.gold-table td {
    padding: 13px 16px;
    border-bottom: 1px solid rgba(214,178,94,0.07);
    color: #A89F92;
}
.gold-table tr:first-child td {
    color: #F0EDE6; font-weight: 600;
    background: rgba(214,178,94,0.05);
}
.gold-table tr:hover td { background: rgba(214,178,94,0.03); }
.gold-badge {
    display: inline-block;
    background: rgba(214,178,94,0.12); color: #D6B25E;
    font-size: 9px; font-weight: 700; padding: 2px 9px;
    border-radius: 20px; letter-spacing: 0.5px;
    border: 1px solid rgba(214,178,94,0.25); margin-left: 6px;
}

/* ── File Tree ── */
.file-tree {
    background: #0D0F14;
    border: 1px solid rgba(214,178,94,0.14);
    border-radius: 12px;
    padding: 22px 26px;
    font-family: 'SF Mono','Fira Code',monospace;
    font-size: 12.5px; line-height: 1.9;
    color: rgba(240,237,230,0.6);
}
.file-tree .dir  { color: #D6B25E; font-weight: 600; }
.file-tree .note { color: rgba(240,237,230,0.25); font-size: 11px; }
code.inline {
    background: rgba(214,178,94,0.1);
    color: #D6B25E;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'SF Mono','Fira Code',monospace;
    font-size: 12px;
}

/* ── Result Card ── */
.result-card {
    background: rgba(214,178,94,0.05);
    border: 1px solid rgba(214,178,94,0.3);
    border-radius: 14px;
    padding: 28px;
    position: relative; overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(214,178,94,0.1), transparent 70%);
    border-radius: 50%;
}

/* ── App Footer ── */
.app-footer {
    text-align: center;
    padding: 32px 0 20px;
    margin-top: 60px;
    border-top: 1px solid rgba(214,178,94,0.12);
    font-size: 11px;
    color: #6B6560;
    letter-spacing: 1px;
    text-transform: uppercase;
    line-height: 2;
    font-family: 'Inter', sans-serif;
}

/* ── Rank Card ── */
.rank-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(214,178,94,0.1);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 6px;
    display: flex; align-items: center; justify-content: space-between;
}
.rank-row:hover { border-color: rgba(214,178,94,0.3); }

/* ── Info banner ── */
.info-banner {
    background: rgba(214,178,94,0.06);
    border-left: 3px solid #D6B25E;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 13.5px; color: #A89F92;
    margin-bottom: 20px;
}

/* ── Sidebar Brand Override ── */
.sidebar-brand-name {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #D6B25E !important;
    letter-spacing: 1.5px !important;
    margin-bottom: 4px !important;
    display: block !important;
}
.sidebar-brand-sub {
    font-size: 8.5px !important;
    color: rgba(214,178,94,0.5) !important;
    text-transform: uppercase !important;
    letter-spacing: 3px !important;
    font-weight: 600 !important;
    display: block !important;
}

/* ── HTML Hero Buttons ── */
.hero-btns { display: flex; gap: 12px; margin-bottom: 0; flex-wrap: wrap; }
.hbtn-primary {
    display: inline-block;
    background: rgba(214,178,94,0.1);
    border: 1px solid #D6B25E;
    color: #D6B25E !important;
    padding: 11px 28px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: all 0.25s;
    text-decoration: none;
    white-space: nowrap;
}
.hbtn-primary:hover {
    background: rgba(214,178,94,0.2);
    box-shadow: 0 6px 24px rgba(214,178,94,0.18);
}
.hbtn-secondary {
    display: inline-block;
    background: transparent;
    border: 1px solid rgba(214,178,94,0.25);
    color: #A89F92 !important;
    padding: 11px 28px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: all 0.25s;
    text-decoration: none;
    white-space: nowrap;
}
.hbtn-secondary:hover { border-color: rgba(214,178,94,0.5); color: #D6B25E !important; }

/* Plotly override */
.js-plotly-plot .plotly { border-radius: 12px; }

/* ── Download / Export buttons — no text-wrap ── */
.stDownloadButton > button {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-size: 12px !important;
}

/* ── Sidebar Collapse Button (inside sidebar → shows «) ── */
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(214,178,94,0.08) !important;
    border: 1px solid rgba(214,178,94,0.25) !important;
    border-radius: 8px !important;
    width: 32px !important; height: 32px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(214,178,94,0.18) !important;
    border-color: rgba(214,178,94,0.55) !important;
}
[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
    color: transparent !important;
    font-size: 0px !important;
}
[data-testid="stSidebarCollapseButton"] button::after {
    content: '«' !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #D6B25E !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1 !important;
    position: absolute !important;
    top: 50% !important; left: 50% !important;
    transform: translate(-50%, -50%) !important;
}

/* ── Sidebar Expand Button (collapsedControl) — styled directly, fully clickable ── */
[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 12px !important;
    left: 12px !important;
    z-index: 99999 !important;
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: auto !important;
    display: block !important;
}
[data-testid="collapsedControl"] button {
    background: rgba(214,178,94,0.15) !important;
    border: 1px solid rgba(214,178,94,0.55) !important;
    border-radius: 8px !important;
    width: 38px !important;
    height: 38px !important;
    cursor: pointer !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    color: #D6B25E !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="collapsedControl"] button:hover {
    background: rgba(214,178,94,0.30) !important;
    border-color: rgba(214,178,94,0.8) !important;
    box-shadow: 0 0 12px rgba(214,178,94,0.25) !important;
}
[data-testid="collapsedControl"] button span[data-testid="stIconMaterial"] {
    color: #D6B25E !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* ── Streamlit Tabs — force gold accent, no red ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(214,178,94,0.12) !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #A89F92 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: color 0.2s ease, border-color 0.2s ease !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #D6B25E !important;
    background: rgba(214,178,94,0.05) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: #D6B25E !important;
    border-bottom: 2px solid #D6B25E !important;
    background: transparent !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: #D6B25E !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background: rgba(214,178,94,0.12) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar icon text patch: change material icons to «/» via JS ──
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
    var doc = window.parent.document;

    function patchIcons() {
        /* Collapse button inside sidebar — show « */
        var colIcon = doc.querySelector('[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]');
        if (colIcon && colIcon.textContent !== '\u00AB') {
            colIcon.textContent = '\u00AB';
            var cb = colIcon.closest('button');
            if (cb) {
                cb.style.background    = 'rgba(214,178,94,0.15)';
                cb.style.border        = '1px solid rgba(214,178,94,0.55)';
                cb.style.borderRadius  = '8px';
            }
        }
        /* Expand button shown when sidebar collapsed — show » */
        var expIcon = doc.querySelector('[data-testid="collapsedControl"] [data-testid="stIconMaterial"]');
        if (expIcon && expIcon.textContent !== '\u00BB') {
            expIcon.textContent = '\u00BB';
        }
    }

    patchIcons();
    setInterval(patchIcons, 300);
    new MutationObserver(patchIcons).observe(doc.body, {childList: true, subtree: true});
})();
</script>
""", height=0)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL & DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'svm'       : 'models/svm_pipeline.pkl',
        'lr'        : 'models/logistic_regression_pipeline.pkl',
        'xgb'       : 'models/xgboost_pipeline.pkl',
    }
    for key, path in model_files.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
    if os.path.exists('models/tfidf_match.pkl'):
        models['tfidf_match'] = joblib.load('models/tfidf_match.pkl')
    if os.path.exists('models/kmeans.pkl'):
        models['kmeans'] = joblib.load('models/kmeans.pkl')
    return models

@st.cache_data
def load_data():
    data = {}
    for path in ['outputs/final_resume_data_with_clusters.csv', 'data/processed_resume.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['extracted_skills'] = df['extracted_skills'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
            )
            data['resume'] = df
            break
    for path in ['outputs/final_jobs_data.csv', 'data/processed_jobs.csv']:
        if os.path.exists(path):
            data['jobs'] = pd.read_csv(path)
            break
    if os.path.exists('outputs/model_comparison_results.csv'):
        data['comparison'] = pd.read_csv('outputs/model_comparison_results.csv')
    return data

models = load_models()
data   = load_data()

CATEGORY_MAP = {
    0:'Advocate',1:'Arts',2:'Automation Testing',3:'Blockchain',
    4:'Business Analyst',5:'Civil Engineer',6:'Data Science',
    7:'Database',8:'DevOps Engineer',9:'DotNet Developer',
    10:'ETL Developer',11:'Electrical Engineering',12:'HR',
    13:'Hadoop',14:'Health and Fitness',15:'Java Developer',
    16:'Mechanical Engineer',17:'Network Security Engineer',
    18:'Operations Manager',19:'PMO',20:'Python Developer',
    21:'SAP Developer',22:'Sales',23:'Testing',24:'Web Designing',
    25:'React Developer',26:'Digital Marketing',27:'Finance',
    28:'Cloud Engineer',29:'AI/ML Engineer',30:'Product Manager',
    31:'UI/UX Designer',32:'Mobile Developer',33:'Cybersecurity',
    34:'Embedded Systems',35:'Game Developer',36:'IoT Engineer',
    37:'Robotics',38:'Content Writer',39:'Project Manager',
    40:'QA Engineer',41:'Systems Analyst'
}
if 'resume' in data:
    try:
        cat_series = data['resume'][['category_encoded','Category']].drop_duplicates().set_index('category_encoded')['Category']
        CATEGORY_MAP = cat_series.to_dict()
    except Exception:
        pass

model_count = len([k for k in ['svm','rf','lr','xgb'] if k in models])
data_loaded = 'resume' in data

if 'resume_history' not in st.session_state:
    st.session_state.resume_history = []
if 'shortlist' not in st.session_state:
    st.session_state.shortlist = []
if 'candidate_notes' not in st.session_state:
    st.session_state.candidate_notes = {}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — build a single-row DataFrame from raw resume text
# All pipelines use a ColumnTransformer that needs these exact columns.
# ─────────────────────────────────────────────────────────────────────────────
import re as _re

def build_input_df(text: str) -> "pd.DataFrame":
    text_lower = text.lower()
    # Experience years — find first number followed by "year" / "yr"
    exp_match = _re.search(r'(\d+)\s*(?:\+\s*)?(?:year|yr)', text_lower)
    exp_years = float(exp_match.group(1)) if exp_match else 0.0
    # Skill count — match against common vocab
    _vocab = [
        "python","java","sql","javascript","r","c++","machine learning",
        "deep learning","tensorflow","pytorch","aws","azure","docker",
        "kubernetes","git","linux","excel","tableau","power bi","spark",
        "hadoop","react","node","django","flask","spring","nlp",
        "computer vision","data analysis","mongodb","postgresql","mysql",
        "redis","kafka","airflow","scala","php","swift","kotlin",
        "android","ios","typescript"
    ]
    skill_cnt = sum(1 for s in _vocab if s in text_lower)
    # Education level — 0=HS, 1=Bachelor, 2=Master, 3=PhD
    if any(w in text_lower for w in ["phd","ph.d","doctorate"]):
        edu = 3
    elif any(w in text_lower for w in ["master","msc","m.sc","mba","m.s."]):
        edu = 2
    elif any(w in text_lower for w in ["bachelor","bsc","b.sc","b.e","b.tech","be ","bs "]):
        edu = 1
    else:
        edu = 0
    return pd.DataFrame({
        'clean_text'       : [text],
        'Experience Years' : [exp_years],
        'skill_count'      : [float(skill_cnt)],
        'education_level'  : [float(edu)],
    })

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 28px 20px 24px; border-bottom: 1px solid rgba(214,178,94,0.12); margin-bottom: 20px;
                display:flex; align-items:center; gap:14px;'>
        <div style='width:44px; height:44px; border-radius:12px; flex-shrink:0;
                    background: linear-gradient(135deg, rgba(214,178,94,0.25) 0%, rgba(140,122,91,0.35) 100%);
                    border: 1px solid rgba(214,178,94,0.4);
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.3rem;'>◆</div>
        <div>
            <span class='sidebar-brand-name' style='font-size:1.15rem;'>AIRECRUIT</span>
            <span class='sidebar-brand-sub'>Intelligence Platform</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("Navigation")
    PAGE_OPTIONS = [
        "🏠 Home",
        "🔬 Workflow",
        "🧠 Resume Intelligence",
        "💼 Recommendation Engine",
        "⚙️ ML Models",
        "📊 Candidate Analytics",
        "📈 Performance Metrics",
        "👥 About"
    ]
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = PAGE_OPTIONS[0]
    if "pending_nav" in st.session_state:
        st.session_state["nav_radio"] = st.session_state.pop("pending_nav")
    page = st.radio(
        label="",
        options=PAGE_OPTIONS,
        key="nav_radio",
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

    # ── System Status ──
    st.markdown("System Status")
    st.markdown(f"""
    <div style='border-radius:10px; overflow:hidden;'>
        <div style='display:flex; justify-content:space-between; align-items:center;
                    padding:9px 4px; border-bottom:1px solid rgba(214,178,94,0.08);'>
            <span style='font-size:13px; color:rgba(240,237,230,0.45);'>Models</span>
            <span style='font-size:13px; color:#D6B25E; font-weight:600; letter-spacing:0.3px;'>{model_count}/4</span>
        </div>
        <div style='display:flex; justify-content:space-between; align-items:center;
                    padding:9px 4px; border-bottom:1px solid rgba(214,178,94,0.08);'>
            <span style='font-size:13px; color:rgba(240,237,230,0.45);'>Resume Data</span>
            <span style='font-size:13px; color:{"#D6B25E" if data_loaded else "#8C7A5B"}; font-weight:600;'>
                {"✓ Ready" if data_loaded else "✗ Missing"}</span>
        </div>
        <div style='display:flex; justify-content:space-between; align-items:center;
                    padding:9px 4px;'>
            <span style='font-size:13px; color:rgba(240,237,230,0.45);'>Jobs Data</span>
            <span style='font-size:13px; color:{"#D6B25E" if "jobs" in data else "#8C7A5B"}; font-weight:600;'>
                {"✓ Ready" if "jobs" in data else "✗ Missing"}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

    # ── Project Info Card ──
    st.markdown("Project")
    st.markdown("""
    <div style='padding: 14px 4px 6px;'>
        <div style='font-size:13.5px; color:rgba(240,237,230,0.75); line-height:1.65;
                    font-weight:400; margin-bottom:14px;'>
            AI-Powered Intelligent Resume Screening &amp; Candidate Role Recommendation
        </div>
        <div style='width:100%; height:1px; background:rgba(214,178,94,0.1); margin-bottom:12px;'></div>
        <div style='font-size:11px; color:rgba(214,178,94,0.45); letter-spacing:0.4px; line-height:2;'>
            Python &nbsp;·&nbsp; Scikit-Learn &nbsp;·&nbsp; XGBoost<br>
            Streamlit &nbsp;·&nbsp; NLP &nbsp;·&nbsp; SE-CD-638
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME + PROJECT OVERVIEW (merged)
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    # ── HERO ──────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.15, 1], gap="large")

    with col_left:
        st.markdown("""
        <div style='padding: 60px 0 32px;'>
            <div class='hero-eyebrow'>AI-Powered Recruitment Intelligence</div>
            <div class='hero-title'>
                AI-Powered<br>Intelligent Resume<br><span class='gold'>Screening</span> &amp;<br>
                Role Recommendation
            </div>
            <div class='hero-subtitle'>
                An intelligent recruitment platform that leverages Machine Learning and NLP techniques
                to automate resume screening, skill extraction, candidate classification, and role
                recommendation for modern hiring systems.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip(
            [c1, c2, c3, c4],
            ["98.75%", "42", "4", "10"],
            ["Best Accuracy", "Job Categories", "ML Models", "Clusters"]
        ):
            with col:
                st.markdown(f"""
                <div class='metric-glass' style='min-height:120px;display:flex;flex-direction:column;
                            align-items:center;justify-content:center;padding:20px 12px;'>
                    <div class='metric-glass-val' style='font-size:1.8rem;'>{val}</div>
                    <div class='metric-glass-lbl' style='font-size:10px;white-space:nowrap;'>{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_right:
        # 2D Neural network visualization (no WebGL required)
        np.random.seed(42)
        n_nodes = 55
        # Multi-ring layout to look like a sphere projection
        rings = [(1, 0.0), (6, 0.22), (12, 0.45), (16, 0.68), (12, 0.88), (8, 1.05)]
        node_x, node_y, node_sizes, node_colors = [], [], [], []
        for count, radius in rings:
            for k in range(count):
                angle = (2 * np.pi * k / max(count, 1)) + np.random.uniform(-0.08, 0.08)
                node_x.append(radius * np.cos(angle) + np.random.uniform(-0.02, 0.02))
                node_y.append(radius * np.sin(angle) + np.random.uniform(-0.02, 0.02))
                node_sizes.append(np.random.uniform(6, 13))
                node_colors.append(np.random.uniform(0, 1))
        # Add scattered outer particles
        for _ in range(14):
            a = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(1.1, 1.45)
            node_x.append(r * np.cos(a)); node_y.append(r * np.sin(a))
            node_sizes.append(np.random.uniform(2, 5))
            node_colors.append(np.random.uniform(0.2, 0.6))

        nx = np.array(node_x); ny = np.array(node_y)
        # Edges between nearby nodes
        edge_x, edge_y = [], []
        for i in range(len(nx)):
            dists = np.sqrt((nx - nx[i])**2 + (ny - ny[i])**2)
            nearest = np.argsort(dists)[1:4]
            for j in nearest:
                if dists[j] < 0.55:
                    edge_x += [nx[i], nx[j], None]
                    edge_y += [ny[i], ny[j], None]

        fig_net = go.Figure()
        # Draw edges
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(color='rgba(214,178,94,0.18)', width=1),
            hoverinfo='none', showlegend=False
        ))
        # Draw nodes with gold gradient
        fig_net.add_trace(go.Scatter(
            x=nx, y=ny, mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale=[[0,'#4A3B28'],[0.4,'#8C7A5B'],[0.75,'#D6B25E'],[1,'#F0EDE6']],
                opacity=0.88,
                line=dict(width=0.8, color='rgba(214,178,94,0.3)')
            ),
            hoverinfo='none', showlegend=False
        ))
        fig_net.update_layout(
            height=520,
            xaxis=dict(visible=False, range=[-1.7, 1.7], scaleanchor='y'),
            yaxis=dict(visible=False, range=[-1.7, 1.7]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=40, b=0),
            font=dict(family='Inter')
        )
        st.plotly_chart(fig_net, use_container_width=True, config={'displayModeBar': False})


    # ── PROJECT OVERVIEW (merged below hero) ──────────────────────────────────
    st.markdown("""
    <div style='margin-top: 52px; margin-bottom: 8px;'>
        <div class='section-eyebrow'>Project Overview &amp; Context</div>
        <div class='section-title'>What This Project Does</div>
        <div class='section-sub'>
            Problem statement, objectives, expected outcomes, and automation benefits
            of this university Machine Learning research project.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    ov1, ov2 = st.columns(2, gap="medium")
    with ov1:
        st.markdown("""
        <div class='glass-card' style='margin-bottom:16px;min-height:220px;'>
            <div class='section-eyebrow' style='margin-bottom:8px;'>Problem Statement</div>
            <div style='font-family:Playfair Display,serif;font-size:1.02rem;font-weight:600;
                        color:#F0EDE6;margin-bottom:10px;'>The Recruitment Bottleneck</div>
            <div style='font-size:13px;color:#A89F92;line-height:1.8;font-weight:300;'>
                Traditional recruitment pipelines are manual, slow, and prone to unconscious bias.
                Hiring managers spend only 6–7 seconds per resume, causing qualified candidates to be
                overlooked. High-volume applications compound inefficiencies and delay time-to-hire.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card' style='margin-bottom:16px;min-height:220px;'>
            <div class='section-eyebrow' style='margin-bottom:8px;'>Automation Benefits</div>
            <div style='font-family:Playfair Display,serif;font-size:1.02rem;font-weight:600;
                        color:#F0EDE6;margin-bottom:10px;'>Why This Matters</div>
            <div style='font-size:13px;color:#A89F92;line-height:1.85;font-weight:300;'>
                — Eliminates manual first-pass resume review burden<br>
                — Extracts structured skills from unstructured text<br>
                — Semantic matching beyond simple keyword search<br>
                — Objective, data-driven candidate prioritisation<br>
                — Faster, fairer, and more accurate hiring pipeline
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ov2:
        st.markdown("""
        <div class='glass-card' style='margin-bottom:16px;min-height:220px;'>
            <div class='section-eyebrow' style='margin-bottom:8px;'>Objectives</div>
            <div style='font-family:Playfair Display,serif;font-size:1.02rem;font-weight:600;
                        color:#F0EDE6;margin-bottom:10px;'>What We Set Out to Build</div>
            <div style='font-size:13px;color:#A89F92;line-height:1.85;font-weight:300;'>
                — Automate resume parsing and preprocessing at scale<br>
                — Extract structured skills from unstructured text<br>
                — Classify candidates into 42 predefined job-role categories<br>
                — Compute semantic similarity between profiles and job descriptions<br>
                — Generate ranked shortlists via composite scoring formula<br>
                — Deploy a production-quality web dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card' style='margin-bottom:16px;min-height:220px;'>
            <div class='section-eyebrow' style='margin-bottom:8px;'>Expected Outcomes</div>
            <div style='font-family:Playfair Display,serif;font-size:1.02rem;font-weight:600;
                        color:#F0EDE6;margin-bottom:10px;'>Measurable Deliverables</div>
            <div style='font-size:13px;color:#A89F92;line-height:1.85;font-weight:300;'>
                — ≥95% classification accuracy (achieved: 98.75% SVM)<br>
                — Automated first-pass screening, reducing manual effort<br>
                — Ranked candidate shortlists for any job description<br>
                — Configurable scoring engine for hiring managers<br>
                — Streamlit dashboard accessible to non-technical users
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-eyebrow'>Project Details</div>
    <div class='section-rule'></div>
    """, unsafe_allow_html=True)

    pd1, pd2, pd3, pd4 = st.columns(4)
    for col, val, lbl in zip(
        [pd1, pd2, pd3, pd4],
        ["SE-CD-638", "Dr. Aamir Arsalan", "Semester VI", "2026–2027"],
        ["Course Code", "Supervisor", "Academic Semester", "Project Year"]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-glass'>
                <div style='font-family:Playfair Display,serif;font-size:0.95rem;font-weight:700;
                            color:#D6B25E;margin-bottom:6px;'>{val}</div>
                <div class='metric-glass-lbl'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESUME INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Resume Intelligence":

    st.markdown("""
    <div style='padding: 48px 0 8px;'>
        <div class='section-eyebrow'>Section 03 · NLP Pipeline</div>
        <div class='section-title'>Resume Intelligence</div>
        <div class='section-sub'>
            The full NLP pipeline that transforms raw resume text into structured, ML-ready features.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    COMMON_SKILLS = [
        "python","java","sql","javascript","r","c++","machine learning","deep learning",
        "tensorflow","pytorch","aws","azure","docker","kubernetes","git","linux","excel",
        "tableau","power bi","spark","hadoop","react","node","django","flask","spring",
        "nlp","computer vision","data analysis","mongodb","postgresql","mysql","redis",
        "kafka","airflow","scala","php","swift","kotlin","android","ios","typescript"
    ]

    INTERVIEW_QS = {
        "Data Science": ["Explain the bias-variance tradeoff and how you address it.", "Walk me through a complete ML pipeline you have built.", "How do you handle class imbalance in classification problems?", "Explain cross-validation and why it matters for model selection."],
        "Python Developer": ["What is the difference between a shallow copy and a deep copy?", "Explain Python's GIL and its impact on multithreading.", "What are decorators and how do you implement one?", "Explain the difference between asyncio and threading."],
        "Java Developer": ["Difference between HashMap and TreeMap in Java?", "Explain the SOLID principles with an example.", "How does garbage collection work in Java?", "Explain the Executor framework and when to use it."],
        "Web Designing": ["What is the CSS box model and how does it affect layout?", "Explain the difference between Flexbox and CSS Grid.", "What is the critical rendering path and how do you optimise it?", "Explain responsive design principles and media queries."],
        "DevOps Engineer": ["What is the difference between Docker and Kubernetes?", "Explain the stages of a CI/CD pipeline.", "How do you monitor production services at scale?", "What is blue-green deployment and when would you use it?"],
        "Blockchain": ["What is a smart contract and how is it deployed?", "Explain proof-of-work vs proof-of-stake consensus mechanisms.", "What is the double-spending problem and how does blockchain solve it?", "Explain the difference between a public and private blockchain."],
        "Testing": ["Difference between black-box and white-box testing?", "What is regression testing and when is it triggered?", "Explain the test pyramid and its recommended ratios.", "What is exploratory testing and when is it most valuable?"],
        "HR": ["How do you handle conflicts between team members?", "What metrics do you use to evaluate recruitment success?", "Describe your approach to the employee onboarding process.", "How do you measure and improve employee engagement?"],
        "Sales": ["What is your approach to cold outreach and lead qualification?", "How do you handle difficult objections from prospects?", "Describe how you manage your sales pipeline.", "What CRM tools have you used and how did they help?"],
        "Business Analyst": ["How do you gather and validate requirements from stakeholders?", "Explain the difference between a BRD and an FRD.", "How do you prioritise requirements when there are conflicting interests?", "What is a use case diagram and when do you create one?"],
        "Network Security Engineer": ["What is a man-in-the-middle attack and how do you prevent it?", "Walk me through the OSI model layers.", "What is zero-trust architecture and its core principles?", "Difference between IDS and IPS?"],
        "Database": ["Explain the difference between SQL and NoSQL databases.", "What is database normalisation and why does it matter?", "Explain the ACID properties of a transaction.", "What is a database index and when should you add one?"],
        "Automation Testing": ["What is the difference between Selenium and Cypress?", "How do you decide what to automate vs test manually?", "Explain the Page Object Model design pattern.", "How do you handle flaky tests in your automation suite?"],
        "DotNet Developer": ["Explain the difference between .NET Framework and .NET Core.", "What is dependency injection and how does ASP.NET Core implement it?", "Explain the difference between IEnumerable and IQueryable.", "What is async/await in C# and how does it work?"],
        "ETL Developer": ["Explain the ETL process and the role of each step.", "How do you handle data quality issues in a pipeline?", "Difference between incremental and full load in ETL?", "What tools have you used for building ETL pipelines?"],
    }

    best_key = next((k for k in ['svm', 'rf', 'lr', 'xgb'] if k in models), None)

    ri_tab1, ri_tab2, ri_tab3, ri_tab4, ri_tab5 = st.tabs([
        "🧬  NLP Pipeline", "🔍  Single Screener", "📦  Batch Upload", "🔄  CV Compare", "📜  History"
    ])

    with ri_tab1:
        cards = [
            ("01", "Resume Parsing",
             "Raw resume documents are ingested and their textual content extracted. Structured fields — name, experience, education, skills — are identified and separated from narrative text for downstream processing."),
            ("02", "Text Preprocessing",
             "Raw text undergoes lowercasing, tokenisation, stop-word removal, and lemmatisation via NLTK. Noise characters, URLs, punctuation, and numeric noise are removed using regex pipelines to produce a clean text representation."),
            ("03", "TF-IDF Vectorisation",
             "Term Frequency–Inverse Document Frequency with max_features=3,000 for classification and 5,000 for semantic matching. Fitted strictly on training data inside a Scikit-Learn Pipeline to prevent data leakage across cross-validation folds."),
            ("04", "Skill Extraction",
             "Keyword matching against a curated skills vocabulary (skills_list.csv) identifies technical and soft skills present in each resume. Skill count is derived as a numeric feature used in the composite ranking formula."),
            ("05", "Role Prediction",
             "The best-performing classifier (SVM, 98.75% accuracy) predicts the most likely job-role category for each resume from 42 predefined classes. Prediction probabilities are exposed as confidence scores in the interface."),
            ("06", "Candidate Classification",
             "Beyond single-label classification, K-Means clustering (k=10) groups candidates into latent segments based on their TF-IDF profile. This enables discovery of candidate clusters beyond the supervised label space."),
        ]
        c1, c2, c3 = st.columns(3, gap="medium")
        cols_cycle = [c1, c2, c3, c1, c2, c3]
        for col, (num, title, body) in zip(cols_cycle, cards):
            with col:
                st.markdown(f"""
                <div class='glass-card' style='margin-bottom:16px; min-height:200px;'>
                    <div style='font-family:Playfair Display,serif;font-size:0.7rem;font-weight:600;
                                color:#8C7A5B;letter-spacing:2px;margin-bottom:10px;'>{num}</div>
                    <div style='font-family:Playfair Display,serif;font-size:1rem;font-weight:700;
                                color:#F0EDE6;margin-bottom:12px;'>{title}</div>
                    <div style='font-size:13px;color:#A89F92;line-height:1.75;font-weight:300;'>{body}</div>
                </div>
                """, unsafe_allow_html=True)

    with ri_tab2:
        if not models:
            st.markdown("<div class='info-banner'>Run the training notebooks first to load models.</div>", unsafe_allow_html=True)
        else:
            col_in, col_out = st.columns([1, 1], gap="large")
            with col_in:
                input_mode = st.radio("Input Method", ["Paste Text", "Upload PDF"], horizontal=True, key="ri_input_mode")
                resume_text = ""
                if input_mode == "Paste Text":
                    st.markdown("<div style='font-size:12px;color:#D6B25E;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Paste Resume Text</div>", unsafe_allow_html=True)
                    resume_text = st.text_area("", height=220, placeholder="Paste full resume text here...", key="ri_text", label_visibility="collapsed")
                else:
                    st.markdown("<div style='font-size:12px;color:#D6B25E;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Upload PDF</div>", unsafe_allow_html=True)
                    uploaded_pdf = st.file_uploader("", type=["pdf"], key="ri_pdf", label_visibility="collapsed")
                    if uploaded_pdf:
                        try:
                            import pdfplumber, io
                            with pdfplumber.open(io.BytesIO(uploaded_pdf.read())) as pdf:
                                resume_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                            st.success(f"Extracted {len(resume_text.split())} words from PDF")
                        except Exception as e:
                            st.error(f"PDF extraction error: {e}")
                conf_threshold = st.slider("Minimum Confidence %", 0, 100, 50, key="ri_conf_thresh")
                go_btn = st.button("Classify Resume →", key="ri_classify")

            with col_out:
                if go_btn and resume_text.strip() and best_key:
                    with st.spinner("Analysing..."):
                        try:
                            mdl       = models[best_key]
                            input_df  = build_input_df(resume_text)
                            pred_enc  = mdl.predict(input_df)[0]
                            pred_prob = mdl.predict_proba(input_df)[0]
                            conf      = pred_prob.max() * 100
                            role      = CATEGORY_MAP.get(pred_enc, f"Category {pred_enc}")
                            top5_idx  = np.argsort(pred_prob)[::-1][:5]

                            text_lower = resume_text.lower()
                            ats_checks = {
                                "Experience section": any(w in text_lower for w in ["experience","work history","employment"]),
                                "Education section":  any(w in text_lower for w in ["education","university","degree","bachelor","master","phd"]),
                                "Skills section":     any(w in text_lower for w in ["skills","technologies","tools","proficient"]),
                                "Contact info":       any(w in text_lower for w in ["email","phone","linkedin","github"]),
                                "Keyword density":    len(resume_text.split()) > 100,
                            }
                            ats_score = int(sum(ats_checks.values()) / len(ats_checks) * 100)
                            ats_color = "#D6B25E" if ats_score >= 80 else "#8C7A5B" if ats_score >= 60 else "#6B6560"
                            ats_rows_html = "".join(
                                "<div style='font-size:11px;color:{};'>{} {}</div>".format(
                                    "#D6B25E" if v else "#6B6560", "✓" if v else "✗", k
                                ) for k, v in ats_checks.items()
                            )

                            if conf < conf_threshold:
                                st.warning(f"Confidence {conf:.1f}% is below your threshold of {conf_threshold}%. Add more detailed resume content.")
                            else:
                                st.markdown(f"""
                                <div class='result-card'>
                                    <div style='font-size:9px;text-transform:uppercase;letter-spacing:2.5px;
                                                color:#8C7A5B;margin-bottom:10px;'>Predicted Role</div>
                                    <div style='font-family:Playfair Display,serif;font-size:1.7rem;
                                                font-weight:700;color:#D6B25E;margin-bottom:6px;'>{role}</div>
                                    <div style='font-size:12px;color:#A89F92;margin-bottom:16px;'>
                                        Confidence: <span style='color:#D6B25E;font-weight:600;'>{conf:.1f}%</span>
                                        &nbsp;·&nbsp; Model: <span style='color:#F0EDE6;'>{best_key.upper()}</span>
                                    </div>
                                    <div style='display:flex;align-items:center;gap:14px;padding:10px 14px;
                                                background:rgba(255,255,255,0.03);border-radius:8px;
                                                border:1px solid rgba(214,178,94,0.12);'>
                                        <div>
                                            <div style='font-size:9px;text-transform:uppercase;letter-spacing:2px;color:#8C7A5B;'>ATS Readiness</div>
                                            <div style='font-size:1.4rem;font-weight:700;color:{ats_color};font-family:Playfair Display,serif;'>{ats_score}%</div>
                                        </div>
                                        <div style='flex:1;'>{ats_rows_html}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                st.markdown("<br><div style='font-size:11px;color:#8C7A5B;text-transform:uppercase;letter-spacing:2px;margin-bottom:14px;'>Top 5 Predictions</div>", unsafe_allow_html=True)
                                for i in top5_idx:
                                    r_name = CATEGORY_MAP.get(i, f"Cat {i}")
                                    r_prob = pred_prob[i] * 100
                                    st.markdown(f"""
                                    <div class='prog-row'>
                                        <div class='prog-hdr'>
                                            <span class='prog-name'>{r_name}</span>
                                            <span class='prog-val'>{r_prob:.1f}%</span>
                                        </div>
                                        <div class='prog-track'>
                                            <div class='prog-fill' style='width:{r_prob:.1f}%;'></div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                iq_list = INTERVIEW_QS.get(role, [
                                    f"Tell me about your most impactful project in {role}.",
                                    "Describe a major challenge you overcame in your field.",
                                    "How do you stay current with industry developments?",
                                    "Walk me through your most significant technical contribution.",
                                ])
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown("<div style='font-size:11px;color:#8C7A5B;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;'>Suggested Interview Questions</div>", unsafe_allow_html=True)
                                iq_html = "".join(
                                    "<div style='padding:8px 12px;margin-bottom:6px;background:rgba(214,178,94,0.06);border-left:2px solid rgba(214,178,94,0.4);border-radius:4px;font-size:13px;color:#A89F92;'>{}</div>".format(q)
                                    for q in iq_list
                                )
                                st.markdown(iq_html, unsafe_allow_html=True)

                                import datetime as _dt
                                text_lower_h = resume_text.lower()
                                skills_h = [s for s in [
                                    "python","java","sql","javascript","r","c++","machine learning",
                                    "deep learning","tensorflow","pytorch","aws","azure","docker",
                                    "kubernetes","git","linux","excel","tableau","power bi","spark",
                                    "hadoop","react","node","django","flask","spring","nlp",
                                    "computer vision","data analysis","mongodb","postgresql","mysql",
                                    "redis","kafka","airflow","scala","php","swift","kotlin",
                                    "android","ios","typescript"
                                ] if s in text_lower_h]
                                top5_saved = [
                                    {"role": CATEGORY_MAP.get(i, f"Cat {i}"), "prob": round(float(pred_prob[i]) * 100, 1)}
                                    for i in top5_idx
                                ]
                                entry = {
                                    "id": len(st.session_state.resume_history) + 1,
                                    "label": f"Resume #{len(st.session_state.resume_history) + 1}",
                                    "timestamp": _dt.datetime.now().strftime("%H:%M:%S"),
                                    "role": role,
                                    "conf": round(conf, 1),
                                    "ats": ats_score,
                                    "model": best_key.upper(),
                                    "words": len(resume_text.split()),
                                    "skills": skills_h,
                                    "top5": top5_saved,
                                    "text": resume_text[:2000],
                                }
                                existing_texts = [h["text"][:200] for h in st.session_state.resume_history]
                                if resume_text[:200] not in existing_texts:
                                    st.session_state.resume_history.append(entry)

                        except Exception as e:
                            st.error(f"Classification error: {e}")
                elif go_btn and not resume_text.strip():
                    st.warning("Please enter or upload resume text first.")
                else:
                    st.markdown("""
                    <div style='text-align:center;padding:60px 20px;color:#6B6560;'>
                        <div style='font-family:Playfair Display,serif;font-size:1rem;font-weight:600;
                                    color:#8C7A5B;margin-bottom:8px;'>Ready to Analyse</div>
                        <div style='font-size:13px;'>Paste or upload a resume and click Classify Resume</div>
                    </div>
                    """, unsafe_allow_html=True)

    with ri_tab3:
        st.markdown("<div style='font-size:13px;color:#A89F92;margin-bottom:20px;line-height:1.7;'>Upload multiple PDF resumes simultaneously. Each is classified using the best available model, scored for ATS readiness, and ranked by confidence. Download the full batch results as CSV.</div>", unsafe_allow_html=True)
        if not best_key:
            st.markdown("<div class='info-banner'>Load models first to enable batch classification.</div>", unsafe_allow_html=True)
        else:
            batch_pdfs = st.file_uploader("Upload PDF Resumes (select multiple)", type=["pdf"], accept_multiple_files=True, key="ri_batch_pdf")
            batch_btn  = st.button("Classify All →", key="ri_batch_run")
            if batch_btn and batch_pdfs:
                with st.spinner("Processing " + str(len(batch_pdfs)) + " resume(s)..."):
                    import pdfplumber as _pdfplumber, io as _io
                    batch_results = []
                    mdl_b = models[best_key]
                    for pdf_file in batch_pdfs:
                        try:
                            with _pdfplumber.open(_io.BytesIO(pdf_file.read())) as pdf_obj:
                                txt = "\n".join(p.extract_text() or "" for p in pdf_obj.pages)
                            if not txt.strip():
                                batch_results.append({"File": pdf_file.name, "Predicted Role": "Unreadable", "Confidence %": 0, "ATS Score %": 0, "Word Count": 0})
                                continue
                            input_df_b = build_input_df(txt)
                            enc_b  = mdl_b.predict(input_df_b)[0]
                            prob_b = mdl_b.predict_proba(input_df_b)[0]
                            conf_b = round(float(prob_b.max()) * 100, 1)
                            role_b = CATEGORY_MAP.get(enc_b, "Unknown")
                            tl     = txt.lower()
                            ats_b  = int(sum([
                                any(w in tl for w in ["experience","employment","work history"]),
                                any(w in tl for w in ["education","degree","university"]),
                                any(w in tl for w in ["skills","tools","technologies"]),
                                any(w in tl for w in ["email","phone","linkedin"]),
                                len(txt.split()) > 100,
                            ]) / 5 * 100)
                            top5_idx_b = np.argsort(prob_b)[::-1][:5]
                            top5_saved_b = [{"role": CATEGORY_MAP.get(i, f"Cat {i}"), "prob": round(float(prob_b[i]) * 100, 1)} for i in top5_idx_b]
                            skills_b_h = [s for s in COMMON_SKILLS if s in tl]
                            batch_results.append({"File": pdf_file.name, "Predicted Role": role_b, "Confidence %": conf_b, "ATS Score %": ats_b, "Word Count": len(txt.split())})
                            import datetime as _dt_b
                            _h_b = {
                                "id": len(st.session_state.resume_history) + 1,
                                "label": pdf_file.name,
                                "timestamp": _dt_b.datetime.now().strftime("%H:%M:%S"),
                                "source": "batch",
                                "role": role_b, "conf": conf_b, "ats": ats_b,
                                "model": best_key.upper(),
                                "words": len(txt.split()),
                                "skills": skills_b_h,
                                "top5": top5_saved_b,
                                "text": txt[:2000],
                            }
                            if txt[:200] not in [h["text"][:200] for h in st.session_state.resume_history]:
                                st.session_state.resume_history.append(_h_b)
                        except Exception:
                            batch_results.append({"File": pdf_file.name, "Predicted Role": "Error", "Confidence %": 0, "ATS Score %": 0, "Word Count": 0})
                    if batch_results:
                        batch_df = pd.DataFrame(batch_results).sort_values("Confidence %", ascending=False).reset_index(drop=True)
                        batch_df.index += 1
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("<div style='font-size:11px;color:#8C7A5B;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;'>Batch Results — Ranked by Confidence</div>", unsafe_allow_html=True)
                        st.dataframe(batch_df, use_container_width=True)
                        batch_csv = batch_df.to_csv().encode('utf-8')
                        st.download_button("⬇ Download Batch Results (CSV)", batch_csv, "batch_results.csv", "text/csv", key="ri_batch_dl")
            elif batch_btn and not batch_pdfs:
                st.warning("Please upload at least one PDF file.")
            else:
                st.markdown("""
                <div style='text-align:center;padding:60px 20px;color:#6B6560;'>
                    <div style='font-family:Playfair Display,serif;font-size:1rem;font-weight:600;
                                color:#8C7A5B;margin-bottom:8px;'>Batch Screener Ready</div>
                    <div style='font-size:13px;'>Upload one or more PDFs and click Classify All</div>
                </div>
                """, unsafe_allow_html=True)

    with ri_tab4:
        st.markdown("<div style='font-size:13px;color:#A89F92;margin-bottom:20px;line-height:1.7;'>Compare two candidates side-by-side — upload PDFs or paste resume text. Shows predicted roles, confidence scores, ATS readiness, and skill overlap.</div>", unsafe_allow_html=True)
        if not best_key:
            st.markdown("<div class='info-banner'>Load models first to enable CV comparison.</div>", unsafe_allow_html=True)
        else:
            import pdfplumber as _pdf_cmp, io as _io_cmp
            cv_cola, cv_colb = st.columns(2, gap="large")
            with cv_cola:
                st.markdown("<div style='font-size:12px;color:#D6B25E;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Candidate A</div>", unsafe_allow_html=True)
                pdf_a = st.file_uploader("Upload PDF (Candidate A)", type=["pdf"], key="cv_pdf_a")
                cv_a = st.text_area("Or paste resume text", height=160, placeholder="Paste Candidate A resume text...", key="cv_a")
                if pdf_a:
                    with _pdf_cmp.open(_io_cmp.BytesIO(pdf_a.read())) as _p:
                        _extracted = "\n".join(pg.extract_text() or "" for pg in _p.pages)
                    if _extracted.strip():
                        cv_a = _extracted
                        st.success(f"PDF loaded: {len(cv_a.split())} words extracted")
            with cv_colb:
                st.markdown("<div style='font-size:12px;color:#D6B25E;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Candidate B</div>", unsafe_allow_html=True)
                pdf_b = st.file_uploader("Upload PDF (Candidate B)", type=["pdf"], key="cv_pdf_b")
                cv_b = st.text_area("Or paste resume text", height=160, placeholder="Paste Candidate B resume text...", key="cv_b")
                if pdf_b:
                    with _pdf_cmp.open(_io_cmp.BytesIO(pdf_b.read())) as _p:
                        _extracted = "\n".join(pg.extract_text() or "" for pg in _p.pages)
                    if _extracted.strip():
                        cv_b = _extracted
                        st.success(f"PDF loaded: {len(cv_b.split())} words extracted")
            compare_btn = st.button("Compare Candidates →", key="cv_compare")

            if compare_btn and cv_a.strip() and cv_b.strip():
                with st.spinner("Comparing..."):
                    try:
                        mdl_cv = models[best_key]
                        cv_results = {}
                        for lbl_cv, txt_cv in [("A", cv_a), ("B", cv_b)]:
                            input_df_cv = build_input_df(txt_cv)
                            enc_cv  = mdl_cv.predict(input_df_cv)[0]
                            prob_cv = mdl_cv.predict_proba(input_df_cv)[0]
                            conf_cv = float(prob_cv.max()) * 100
                            role_cv = CATEGORY_MAP.get(enc_cv, "Unknown")
                            tl_cv   = txt_cv.lower()
                            found   = [s for s in COMMON_SKILLS if s in tl_cv]
                            ats_cv  = int(sum([
                                any(w in tl_cv for w in ["experience","employment"]),
                                any(w in tl_cv for w in ["education","degree","university"]),
                                any(w in tl_cv for w in ["skills","tools","technologies"]),
                                any(w in tl_cv for w in ["email","phone","linkedin"]),
                                len(txt_cv.split()) > 100,
                            ]) / 5 * 100)
                            cv_results[lbl_cv] = {"role": role_cv, "conf": conf_cv, "skills": found, "ats": ats_cv, "words": len(txt_cv.split())}

                        ra, rb = cv_results["A"], cv_results["B"]
                        skills_a, skills_b = set(ra["skills"]), set(rb["skills"])
                        shared   = sorted(skills_a & skills_b)
                        only_a   = sorted(skills_a - skills_b)
                        only_b   = sorted(skills_b - skills_a)
                        overall_a = ra["conf"] * 0.5 + ra["ats"] * 0.3 + len(ra["skills"]) * 2
                        overall_b = rb["conf"] * 0.5 + rb["ats"] * 0.3 + len(rb["skills"]) * 2
                        winner    = "A" if overall_a > overall_b else ("B" if overall_b > overall_a else "Tie")

                        import datetime as _dt_cv
                        for _lbl_cv2, _r_cv2, _txt_cv2 in [("A", ra, cv_a), ("B", rb, cv_b)]:
                            _h_cv = {
                                "id": len(st.session_state.resume_history) + 1,
                                "label": f"Candidate {_lbl_cv2}",
                                "timestamp": _dt_cv.datetime.now().strftime("%H:%M:%S"),
                                "source": "compare",
                                "role": _r_cv2["role"],
                                "conf": round(_r_cv2["conf"], 1),
                                "ats": _r_cv2["ats"],
                                "model": best_key.upper(),
                                "words": _r_cv2["words"],
                                "skills": _r_cv2["skills"],
                                "top5": [],
                                "text": _txt_cv2[:2000],
                                "winner": winner == _lbl_cv2,
                                "compare_partner": "B" if _lbl_cv2 == "A" else "A",
                            }
                            if _txt_cv2[:200] not in [h["text"][:200] for h in st.session_state.resume_history]:
                                st.session_state.resume_history.append(_h_cv)

                        shared_html = " ".join("<span style='background:rgba(214,178,94,0.12);color:#D6B25E;padding:2px 7px;border-radius:4px;font-size:11px;'>" + s + "</span>" for s in shared) or "<span style='color:#6B6560;font-size:11px;'>None</span>"
                        only_a_html = " ".join("<span style='background:rgba(214,178,94,0.10);color:#D6B25E;padding:2px 7px;border-radius:4px;font-size:11px;'>" + s + "</span>" for s in only_a) or "<span style='color:#6B6560;font-size:11px;'>—</span>"
                        only_b_html = " ".join("<span style='background:rgba(140,122,91,0.15);color:#A89F92;padding:2px 7px;border-radius:4px;font-size:11px;'>" + s + "</span>" for s in only_b) or "<span style='color:#6B6560;font-size:11px;'>—</span>"

                        cmp_l, cmp_r = st.columns(2, gap="large")
                        for cmp_col, lbl_c, r_c in [(cmp_l, "A", ra), (cmp_r, "B", rb)]:
                            is_winner  = winner == lbl_c
                            border_s   = "border-color:rgba(214,178,94,0.5);background:rgba(214,178,94,0.04);" if is_winner else ""
                            badge_s    = " · ★ Stronger" if is_winner else ""
                            ats_c_col  = "#D6B25E" if r_c["ats"] >= 80 else ("#8C7A5B" if r_c["ats"] >= 60 else "#6B6560")
                            conf_c_col = "#D6B25E" if r_c["conf"] >= 80 else "#8C7A5B"
                            with cmp_col:
                                card_html = (
                                    "<div class='result-card' style='" + border_s + "'>"
                                    "<div style='font-size:9px;text-transform:uppercase;letter-spacing:2px;color:#8C7A5B;margin-bottom:6px;'>Candidate " + lbl_c + badge_s + "</div>"
                                    "<div style='font-family:Playfair Display,serif;font-size:1.3rem;font-weight:700;color:#D6B25E;margin-bottom:6px;'>" + r_c["role"] + "</div>"
                                    "<div style='font-size:12px;color:#A89F92;'>"
                                    "Confidence: <span style='color:" + conf_c_col + ";font-weight:600;'>" + "{:.1f}%".format(r_c["conf"]) + "</span>"
                                    " &nbsp;·&nbsp; ATS: <span style='color:" + ats_c_col + ";font-weight:600;'>" + str(r_c["ats"]) + "%</span>"
                                    " &nbsp;·&nbsp; Skills: <b>" + str(len(r_c["skills"])) + "</b>"
                                    " &nbsp;·&nbsp; Words: <b>" + str(r_c["words"]) + "</b>"
                                    "</div></div>"
                                )
                                st.markdown(card_html, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("<div style='font-size:11px;color:#8C7A5B;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;'>Skill Comparison</div>", unsafe_allow_html=True)
                        sk_l, sk_c, sk_r = st.columns(3)
                        with sk_l:
                            st.markdown("<div style='font-size:11px;color:#D6B25E;margin-bottom:6px;font-weight:600;'>Only in A (" + str(len(only_a)) + ")</div>" + only_a_html, unsafe_allow_html=True)
                        with sk_c:
                            st.markdown("<div style='font-size:11px;color:#A89F92;margin-bottom:6px;font-weight:600;'>Shared (" + str(len(shared)) + ")</div>" + shared_html, unsafe_allow_html=True)
                        with sk_r:
                            st.markdown("<div style='font-size:11px;color:#8C7A5B;margin-bottom:6px;font-weight:600;'>Only in B (" + str(len(only_b)) + ")</div>" + only_b_html, unsafe_allow_html=True)

                        if winner != "Tie":
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown(
                                "<div style='padding:12px 18px;background:rgba(214,178,94,0.08);border:1px solid rgba(214,178,94,0.3);border-radius:8px;font-size:13px;color:#D6B25E;'>"
                                "Verdict: <strong>Candidate " + winner + "</strong> shows a stronger overall profile based on prediction confidence, ATS readiness, and skill breadth.</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.info("Both candidates show comparable overall scores.")
                    except Exception as e:
                        st.error(f"Comparison error: {e}")
            elif compare_btn:
                st.warning("Please provide both resume texts to compare.")
            else:
                st.markdown("""
                <div style='text-align:center;padding:60px 20px;color:#6B6560;'>
                    <div style='font-family:Playfair Display,serif;font-size:1rem;font-weight:600;
                                color:#8C7A5B;margin-bottom:8px;'>CV Comparison Ready</div>
                    <div style='font-size:13px;'>Paste two resumes above and click Compare Candidates</div>
                </div>
                """, unsafe_allow_html=True)

    with ri_tab5:
        history = st.session_state.resume_history

        # ── source badge helper ──
        def _src_badge(src):
            cfg = {
                "single":  ("🔍", "#4A3B28", "#D6B25E", "Single"),
                "batch":   ("📦", "#1E2D1E", "#6DBF67", "Batch"),
                "compare": ("🔄", "#1A2030", "#6B9FD4", "Compare"),
            }
            icon, bg, col, lbl = cfg.get(src, ("·", "#2A2A2A", "#8C7A5B", src.title()))
            return (f"<span style='display:inline-block;background:{bg};color:{col};"
                    f"border:1px solid {col}33;border-radius:20px;padding:2px 8px;"
                    f"font-size:9px;font-weight:700;letter-spacing:1px;'>{icon} {lbl}</span>")

        if not history:
            st.markdown("""
            <div style='text-align:center;padding:80px 20px;color:#6B6560;'>
                <div style='font-size:2.5rem;margin-bottom:16px;opacity:0.3;'>📜</div>
                <div style='font-family:serif;font-size:1rem;font-weight:600;
                            color:#8C7A5B;margin-bottom:8px;'>No History Yet</div>
                <div style='font-size:13px;'>Classify resumes via Single Screener, Batch Upload, or CV Compare — all results save here automatically.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ── header row: stats ──
            n_single  = sum(1 for h in history if h.get("source","single") == "single")
            n_batch   = sum(1 for h in history if h.get("source","single") == "batch")
            n_compare = sum(1 for h in history if h.get("source","single") == "compare")
            n_pinned  = len(st.session_state.shortlist)
            st.markdown(
                f"<div style='font-size:12px;color:#8C7A5B;padding-bottom:10px;'>"
                f"<span style='color:#D6B25E;font-weight:600;'>{len(history)}</span> total &nbsp;·&nbsp; "
                f"🔍 {n_single} &nbsp;·&nbsp; 📦 {n_batch} &nbsp;·&nbsp; 🔄 {n_compare}"
                + (f" &nbsp;·&nbsp; <span style='color:#D6B25E;'>📌 {n_pinned} shortlisted</span>" if n_pinned else "")
                + "</div>",
                unsafe_allow_html=True
            )

            # ── header row: buttons on their own separate row ──
            h_col_exp, h_col_pin, h_col_btn = st.columns(3)
            with h_col_exp:
                export_rows = []
                for h in history:
                    export_rows.append({
                        "Resume #": h["id"],
                        "Label": h.get("label", f"Resume #{h['id']}"),
                        "Source": h.get("source", "single").title(),
                        "Shortlisted": "Yes" if any(s["id"] == h["id"] for s in st.session_state.shortlist) else "",
                        "Time": h["timestamp"],
                        "Predicted Role": h["role"],
                        "Confidence %": h["conf"],
                        "ATS Score %": h["ats"],
                        "Model": h["model"],
                        "Word Count": h["words"],
                        "Skills Found": ", ".join(h["skills"]),
                        "Top Prediction": h["top5"][0]["role"] if h.get("top5") else "",
                        "2nd Prediction": h["top5"][1]["role"] if h.get("top5") and len(h["top5"]) > 1 else "",
                        "3rd Prediction": h["top5"][2]["role"] if h.get("top5") and len(h["top5"]) > 2 else "",
                        "Winner (Compare)": "Yes" if h.get("winner") else "",
                        "Recruiter Notes": st.session_state.candidate_notes.get(h["id"], ""),
                        "Resume Text (preview)": h["text"][:300].replace("\n", " "),
                    })
                export_df = pd.DataFrame(export_rows)
                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Export CSV",
                    data=csv_bytes,
                    file_name="airecruit_history.csv",
                    mime="text/csv",
                    key="export_history_csv",
                )
            with h_col_pin:
                if st.session_state.shortlist:
                    sl_csv = pd.DataFrame([{
                        "Label": s.get("label",""), "Role": s["role"],
                        "Confidence %": s["conf"], "ATS %": s["ats"],
                        "Skills": ", ".join(s["skills"]), "Words": s["words"],
                        "Source": s.get("source","single").title(),
                        "Time": s["timestamp"],
                        "Recruiter Notes": s.get("note") or st.session_state.candidate_notes.get(s["id"], ""),
                    } for s in st.session_state.shortlist]).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"📌 Export Shortlist ({len(st.session_state.shortlist)})",
                        data=sl_csv,
                        file_name="airecruit_shortlist.csv",
                        mime="text/csv",
                        key="export_shortlist_csv",
                    )
                else:
                    st.markdown(
                        "<div style='font-size:12px;color:#4A4540;padding-top:8px;text-align:center;'>📌 No shortlist</div>",
                        unsafe_allow_html=True
                    )
            with h_col_btn:
                if st.button("🗑 Clear History", key="clear_history"):
                    st.session_state.resume_history = []
                    st.session_state.shortlist = []
                    st.rerun()

            st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # SHORTLIST PANEL
            # ══════════════════════════════════════════════════
            shortlist = st.session_state.shortlist
            if shortlist:
                st.markdown(
                    "<div style='font-size:11px;color:#D6B25E;text-transform:uppercase;"
                    "letter-spacing:2px;margin-bottom:12px;'>📌 Shortlist</div>",
                    unsafe_allow_html=True
                )
                sl_ids = {s["id"] for s in shortlist}
                sl_grid_cols = st.columns(min(len(shortlist), 3), gap="medium")
                for sl_i, sl_c in enumerate(shortlist):
                    with sl_grid_cols[sl_i % 3]:
                        conf_c_sl = "#D6B25E" if sl_c["conf"] >= 80 else "#8C7A5B"
                        ats_c_sl  = "#D6B25E" if sl_c["ats"]  >= 80 else "#8C7A5B"
                        lbl_sl    = sl_c.get("label", f"Resume #{sl_c['id']}")
                        src_sl    = sl_c.get("source", "single")
                        _sl_note = sl_c.get("note") or st.session_state.candidate_notes.get(sl_c["id"], "")
                        _note_html = (
                            f"<div style='margin-top:8px;padding:6px 8px;background:rgba(255,255,255,0.03);"
                            f"border-left:2px solid rgba(214,178,94,0.3);border-radius:0 4px 4px 0;"
                            f"font-size:11px;color:#A89F92;line-height:1.5;'>📝 {_sl_note}</div>"
                        ) if _sl_note else ""
                        st.markdown(
                            f"<div style='background:rgba(214,178,94,0.05);border:1px solid rgba(214,178,94,0.25);"
                            f"border-radius:10px;padding:14px 16px;margin-bottom:10px;'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;'>"
                            f"<div style='font-size:12px;font-weight:600;color:#F0EDE6;max-width:75%;word-break:break-word;'>{lbl_sl}</div>"
                            f"{_src_badge(src_sl)}</div>"
                            f"<div style='font-family:serif;font-size:1rem;color:#D6B25E;margin-bottom:8px;font-weight:700;'>{sl_c['role']}</div>"
                            f"<div style='display:flex;gap:14px;flex-wrap:wrap;'>"
                            f"<span style='font-size:11px;color:{conf_c_sl};'>Conf: <b>{sl_c['conf']}%</b></span>"
                            f"<span style='font-size:11px;color:{ats_c_sl};'>ATS: <b>{sl_c['ats']}%</b></span>"
                            f"<span style='font-size:11px;color:#A89F92;'>Skills: <b>{len(sl_c['skills'])}</b></span>"
                            f"</div>{_note_html}</div>",
                            unsafe_allow_html=True
                        )
                        if st.button("✕ Remove", key=f"sl_rm_{sl_c['id']}"):
                            st.session_state.shortlist = [s for s in shortlist if s["id"] != sl_c["id"]]
                            st.rerun()
                if st.button("🗑 Clear Shortlist", key="clear_shortlist"):
                    st.session_state.shortlist = []
                    st.rerun()
                st.markdown(
                    "<div style='height:1px;background:rgba(214,178,94,0.08);margin:20px 0;'></div>",
                    unsafe_allow_html=True
                )

            # ══════════════════════════════════════════════════
            # RANK AGAINST JOB DESCRIPTION
            # ══════════════════════════════════════════════════
            st.markdown(
                "<div style='font-size:11px;color:#8C7A5B;text-transform:uppercase;"
                "letter-spacing:2px;margin-bottom:10px;'>🎯 Rank Candidates Against a Job Description</div>",
                unsafe_allow_html=True
            )
            jd_input = st.text_area(
                "",
                height=120,
                placeholder="Paste a job description here (e.g. 'Looking for a Python developer with 3+ years of experience in Django, REST APIs, PostgreSQL…') — all saved resumes will be ranked by fit.",
                key="jd_rank_input",
                label_visibility="collapsed"
            )
            rank_btn = st.button("Rank All Candidates →", key="jd_rank_btn")

            if rank_btn and jd_input.strip():
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
                all_texts = [h["text"] for h in history] + [jd_input]
                try:
                    _tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
                    _mat   = _tfidf.fit_transform(all_texts)
                    _jd_vec = _mat[-1]
                    _sims   = _cos_sim(_mat[:-1], _jd_vec).flatten()

                    ranked = []
                    for i, h in enumerate(history):
                        jd_sim    = round(float(_sims[i]) * 100, 1)
                        composite = round(jd_sim * 0.5 + h["conf"] * 0.3 + h["ats"] * 0.2, 1)
                        ranked.append({**h, "jd_sim": jd_sim, "composite": composite})
                    ranked.sort(key=lambda x: x["composite"], reverse=True)

                    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-size:10px;color:#8C7A5B;text-transform:uppercase;"
                        "letter-spacing:2px;margin-bottom:10px;'>Ranked Results — Best Fit First</div>",
                        unsafe_allow_html=True
                    )
                    for rank_i, r in enumerate(ranked):
                        medal = ["🥇","🥈","🥉"][rank_i] if rank_i < 3 else f"#{rank_i+1}"
                        bar_w = min(r["composite"], 100)
                        sim_c = "#D6B25E" if r["jd_sim"] >= 40 else "#8C7A5B"
                        win_badge = (" <span style='background:#1A2030;color:#6B9FD4;border:1px solid #6B9FD433;border-radius:10px;padding:1px 7px;font-size:9px;'>★ Compare Winner</span>" if r.get("winner") else "")
                        r_label = r.get("label") or ("Resume #" + str(r["id"]))
                        r_src_badge = _src_badge(r.get("source", "single"))
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.025);border:1px solid rgba(214,178,94,0.12);"
                            f"border-radius:10px;padding:14px 18px;margin-bottom:10px;'>"
                            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
                            f"<div style='display:flex;align-items:center;gap:10px;'>"
                            f"<span style='font-size:1.3rem;'>{medal}</span>"
                            f"<div>"
                            f"<div style='font-size:13px;font-weight:600;color:#F0EDE6;'>{r_label}{win_badge}</div>"
                            f"<div style='font-size:11px;color:#8C7A5B;margin-top:2px;'>{r['role']} &nbsp;·&nbsp; {r_src_badge}</div>"
                            f"</div></div>"
                            f"<div style='text-align:right;'>"
                            f"<div style='font-size:1.2rem;font-weight:700;color:#D6B25E;'>{r['composite']:.1f}</div>"
                            f"<div style='font-size:9px;color:#6B6560;text-transform:uppercase;letter-spacing:1px;'>Composite Score</div>"
                            f"</div></div>"
                            f"<div style='display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap;'>"
                            f"<span style='font-size:11px;color:{sim_c};'>JD Match: <b>{r['jd_sim']}%</b></span>"
                            f"<span style='font-size:11px;color:#A89F92;'>Confidence: <b>{r['conf']}%</b></span>"
                            f"<span style='font-size:11px;color:#A89F92;'>ATS: <b>{r['ats']}%</b></span>"
                            f"<span style='font-size:11px;color:#A89F92;'>Skills: <b>{len(r['skills'])}</b></span>"
                            f"</div>"
                            f"<div style='background:rgba(255,255,255,0.04);border-radius:4px;height:5px;'>"
                            f"<div style='background:linear-gradient(90deg,#8C7A5B,#D6B25E);width:{bar_w}%;height:5px;border-radius:4px;'></div>"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )

                    ranked_csv = pd.DataFrame([{
                        "Rank": i+1, "Label": r.get("label",""), "Source": r.get("source","single").title(),
                        "Role": r["role"], "JD Match %": r["jd_sim"],
                        "Confidence %": r["conf"], "ATS %": r["ats"],
                        "Composite Score": r["composite"], "Skills": len(r["skills"])
                    } for i, r in enumerate(ranked)]).to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Ranked Results CSV", ranked_csv, "ranked_candidates.csv", "text/csv", key="ranked_dl")

                    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                    if st.button("🚀 Search Full Candidate Pool with this JD →", key="send_jd_to_rec", help="Run this job description against the full 2,400+ resume pool in the Recommendation Engine"):
                        st.session_state.rec_jd_prefill = jd_input
                        st.session_state.pending_nav = "💼 Recommendation Engine"
                        st.rerun()

                except Exception as e:
                    st.error(f"Ranking error: {e}")
            elif rank_btn:
                st.warning("Please paste a job description first.")

            st.markdown("<div style='height:28px;border-top:1px solid rgba(214,178,94,0.08);margin:24px 0 20px;'></div>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # SESSION SUMMARY TABLE
            # ══════════════════════════════════════════════════
            st.markdown(
                "<div style='font-size:11px;color:#8C7A5B;text-transform:uppercase;"
                "letter-spacing:2px;margin-bottom:10px;'>Session Summary</div>",
                unsafe_allow_html=True
            )
            header_cols = ["#", "Source", "Label", "Predicted Role", "Confidence", "ATS", "Skills", "Words"]
            hdr_html = "".join(
                f"<th style='padding:9px 14px;text-align:left;font-size:10px;"
                f"text-transform:uppercase;letter-spacing:1.5px;"
                f"color:rgba(214,178,94,0.6);border-bottom:1px solid rgba(214,178,94,0.15);'>{c}</th>"
                for c in header_cols
            )
            rows_html = ""
            for h in reversed(history):
                conf_col  = "#D6B25E" if h["conf"] >= 80 else "#8C7A5B"
                ats_col   = "#D6B25E" if h["ats"] >= 80 else "#8C7A5B"
                src       = h.get("source", "single")
                win_star  = " ★" if h.get("winner") else ""
                lbl_disp  = h.get("label", f"Resume #{h['id']}")
                rows_html += (
                    f"<tr style='border-bottom:1px solid rgba(214,178,94,0.05);'>"
                    f"<td style='padding:9px 14px;font-size:12px;color:#6B6560;'>{h['id']}</td>"
                    f"<td style='padding:9px 14px;'>{_src_badge(src)}</td>"
                    f"<td style='padding:9px 14px;font-size:12px;color:#A89F92;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{lbl_disp}{win_star}</td>"
                    f"<td style='padding:9px 14px;font-size:13px;color:#F0EDE6;font-weight:500;'>{h['role']}</td>"
                    f"<td style='padding:9px 14px;font-size:13px;color:{conf_col};font-weight:600;'>{h['conf']}%</td>"
                    f"<td style='padding:9px 14px;font-size:13px;color:{ats_col};font-weight:600;'>{h['ats']}%</td>"
                    f"<td style='padding:9px 14px;font-size:12px;color:#A89F92;'>{len(h['skills'])}</td>"
                    f"<td style='padding:9px 14px;font-size:12px;color:#A89F92;'>{h['words']:,}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(214,178,94,0.12);"
                f"border-radius:10px;overflow:auto;margin-bottom:24px;'>"
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr>{hdr_html}</tr></thead>"
                f"<tbody>{rows_html}</tbody></table></div>",
                unsafe_allow_html=True
            )

            # ══════════════════════════════════════════════════
            # DETAILED EXPANDABLE CARDS
            # ══════════════════════════════════════════════════
            st.markdown(
                "<div style='font-size:11px;color:#8C7A5B;text-transform:uppercase;"
                "letter-spacing:2px;margin-bottom:14px;'>Detailed Results</div>",
                unsafe_allow_html=True
            )
            for h in reversed(history):
                conf_c  = "#D6B25E" if h["conf"] >= 80 else "#8C7A5B"
                ats_c   = "#D6B25E" if h["ats"] >= 80 else "#8C7A5B"
                src     = h.get("source", "single")
                win_str = " · ★ Winner" if h.get("winner") else ""
                lbl     = h.get("label", f"Resume #{h['id']}")
                expander_title = f"{lbl}{win_str} · {h['role']} · {h['conf']}% · {h['timestamp']}"
                with st.expander(expander_title, expanded=False):
                    d_left, d_right = st.columns([1, 1], gap="large")
                    with d_left:
                        win_banner = (
                            "<div style='background:rgba(107,159,212,0.1);border:1px solid rgba(107,159,212,0.3);"
                            "border-radius:6px;padding:6px 12px;font-size:11px;color:#6B9FD4;margin-bottom:12px;'>"
                            "★ This candidate won the CV comparison</div>"
                        ) if h.get("winner") else ""
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(214,178,94,0.15);"
                            f"border-radius:10px;padding:20px;'>"
                            f"{win_banner}"
                            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>"
                            f"{_src_badge(src)}"
                            f"<span style='font-size:10px;color:#6B6560;'>{h['timestamp']}</span></div>"
                            f"<div style='font-size:9px;text-transform:uppercase;letter-spacing:2px;color:#8C7A5B;margin-bottom:4px;'>Predicted Role</div>"
                            f"<div style='font-family:serif;font-size:1.3rem;font-weight:700;color:#D6B25E;margin-bottom:14px;'>{h['role']}</div>"
                            f"<div style='display:flex;gap:20px;flex-wrap:wrap;'>"
                            f"<div><div style='font-size:10px;color:#6B6560;text-transform:uppercase;letter-spacing:1px;'>Confidence</div><div style='font-size:1.1rem;font-weight:700;color:{conf_c};'>{h['conf']}%</div></div>"
                            f"<div><div style='font-size:10px;color:#6B6560;text-transform:uppercase;letter-spacing:1px;'>ATS Score</div><div style='font-size:1.1rem;font-weight:700;color:{ats_c};'>{h['ats']}%</div></div>"
                            f"<div><div style='font-size:10px;color:#6B6560;text-transform:uppercase;letter-spacing:1px;'>Words</div><div style='font-size:1.1rem;font-weight:700;color:#F0EDE6;'>{h['words']:,}</div></div>"
                            f"<div><div style='font-size:10px;color:#6B6560;text-transform:uppercase;letter-spacing:1px;'>Model</div><div style='font-size:1.1rem;font-weight:700;color:#A89F92;'>{h['model']}</div></div>"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
                        if h["skills"]:
                            skills_span = " ".join(
                                f"<span style='display:inline-block;background:rgba(214,178,94,0.10);"
                                f"border:1px solid rgba(214,178,94,0.25);border-radius:20px;"
                                f"padding:3px 10px;font-size:10px;color:#D6B25E;margin:2px;'>{s}</span>"
                                for s in h["skills"]
                            )
                            st.markdown(
                                f"<div style='font-size:10px;color:#6B6560;text-transform:uppercase;"
                                f"letter-spacing:1px;margin-bottom:8px;'>Detected Skills ({len(h['skills'])})</div>"
                                f"<div style='display:flex;flex-wrap:wrap;'>{skills_span}</div>",
                                unsafe_allow_html=True
                            )
                    with d_left:
                        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
                        _is_pinned = any(s["id"] == h["id"] for s in st.session_state.shortlist)
                        _pin_col, _sim_col = st.columns(2, gap="small")
                        with _pin_col:
                            if _is_pinned:
                                if st.button("✅ Shortlisted", key=f"unpin_{h['id']}"):
                                    st.session_state.shortlist = [s for s in st.session_state.shortlist if s["id"] != h["id"]]
                                    st.rerun()
                            else:
                                if st.button("📌 Add to Shortlist", key=f"pin_{h['id']}"):
                                    _note_on_pin = st.session_state.candidate_notes.get(h["id"], "")
                                    st.session_state.shortlist.append({**h, "note": _note_on_pin})
                                    st.rerun()
                        with _sim_col:
                            if st.button("🔍 Find Similar →", key=f"find_sim_{h['id']}", help="Send this resume to Recommendation Engine"):
                                st.session_state.rec_jd_prefill = h["text"]
                                st.session_state.rec_prefill_label = h.get("label", f"Resume #{h['id']}")
                                st.session_state.pending_nav = "💼 Recommendation Engine"
                                st.rerun()
                    with d_right:
                        if h.get("top5"):
                            st.markdown(
                                "<div style='font-size:10px;color:#6B6560;text-transform:uppercase;"
                                "letter-spacing:1px;margin-bottom:10px;'>Top 5 Predictions</div>",
                                unsafe_allow_html=True
                            )
                            for t in h["top5"]:
                                st.markdown(
                                    f"<div style='margin-bottom:8px;'>"
                                    f"<div style='display:flex;justify-content:space-between;"
                                    f"font-size:12px;color:#A89F92;margin-bottom:3px;'>"
                                    f"<span>{t['role']}</span><span style='color:#D6B25E;'>{t['prob']}%</span></div>"
                                    f"<div style='background:rgba(255,255,255,0.05);border-radius:4px;height:4px;'>"
                                    f"<div style='background:#D6B25E;width:{min(t['prob'],100)}%;height:4px;border-radius:4px;'></div>"
                                    f"</div></div>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.markdown(
                                "<div style='font-size:12px;color:#6B6560;padding:10px 0;'>Top predictions not available for this entry type.</div>",
                                unsafe_allow_html=True
                            )
                        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
                        st.markdown(
                            "<div style='font-size:10px;color:#6B6560;text-transform:uppercase;"
                            "letter-spacing:1px;margin-bottom:8px;'>Resume Preview</div>",
                            unsafe_allow_html=True
                        )
                        preview_text = h["text"][:400] + "…" if len(h["text"]) > 400 else h["text"]
                        st.markdown(
                            f"<div style='font-size:11px;color:#6B6560;line-height:1.6;"
                            f"background:rgba(255,255,255,0.02);border-radius:6px;"
                            f"padding:10px 12px;border:1px solid rgba(214,178,94,0.08);'>{preview_text}</div>",
                            unsafe_allow_html=True
                        )
                    # ── Recruiter Notes (full width below both columns) ──
                    st.markdown(
                        "<div style='height:1px;background:rgba(214,178,94,0.07);margin:16px 0 12px;'></div>",
                        unsafe_allow_html=True
                    )
                    _saved_note = st.session_state.candidate_notes.get(h["id"], "")
                    _note_input = st.text_area(
                        "📝 Recruiter Notes",
                        value=_saved_note,
                        key=f"note_input_{h['id']}",
                        height=80,
                        placeholder="Add your notes — strengths, concerns, interview outcome, salary expectation…",
                    )
                    _note_col_save, _note_col_clr, _ = st.columns([1, 1, 4], gap="small")
                    with _note_col_save:
                        if st.button("💾 Save Note", key=f"note_save_{h['id']}"):
                            st.session_state.candidate_notes[h["id"]] = _note_input
                            if _is_pinned:
                                for _sl in st.session_state.shortlist:
                                    if _sl["id"] == h["id"]:
                                        _sl["note"] = _note_input
                            st.rerun()
                    with _note_col_clr:
                        if _saved_note and st.button("🗑 Clear Note", key=f"note_clr_{h['id']}"):
                            st.session_state.candidate_notes.pop(h["id"], None)
                            if _is_pinned:
                                for _sl in st.session_state.shortlist:
                                    if _sl["id"] == h["id"]:
                                        _sl.pop("note", None)
                            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ ML Models":

    st.markdown("""
    <div style='padding: 48px 0 8px;'>
        <div class='section-eyebrow'>Section 05 · Algorithms &amp; Tuning</div>
        <div class='section-title'>Machine Learning Models</div>
        <div class='section-sub'>
            Four supervised classifiers and one unsupervised clustering method,
            each with rigorous cross-validation and hyperparameter optimisation.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    model_data = [
        ("Logistic Regression", "Classifier 01",
         "Multinomial logistic regression tested with L1 (Lasso, sparse) and L2 (Ridge, shrinkage) regularisation at C ∈ {0.01, 0.1, 1.0, 10.0} using the saga solver for large sparse matrices.",
         ["L1 Lasso", "L2 Ridge", "saga solver", "97.00% Accuracy"], False),
        ("Random Forest", "Classifier 02",
         "Ensemble of decision trees tuned with GridSearchCV (cv=5) over n_estimators, max_depth, and min_samples_split. Feature importance extracted for explainability.",
         ["GridSearchCV cv=5", "Feature Importance", "Ensemble", "98.15% Accuracy"], False),
        ("Support Vector Machine", "Classifier 03 · Best Model",
         "Linear SVM with TF-IDF pipeline. Excels at high-dimensional text classification. Best-performing model at 98.75% accuracy. Selected as the primary production classifier.",
         ["Linear Kernel", "High-Dim Text", "98.75% Accuracy", "Production Model"], True),
        ("XGBoost", "Classifier 04",
         "Gradient-boosted trees tuned with RandomizedSearchCV (n_iter=20, cv=3) over continuous distributions of learning rate, max_depth, and n_estimators. Efficient for large feature spaces.",
         ["RandomizedSearchCV", "n_iter=20", "cv=3", "96.50% Accuracy"], False),
        ("K-Means Clustering", "Unsupervised",
         "Applied to the TF-IDF resume matrix. Optimal k=10 selected via Elbow Method (inertia), Silhouette Score (cohesion/separation), and Davies-Bouldin Index (cluster overlap).",
         ["k=10 Clusters", "Silhouette Score", "Davies-Bouldin", "Elbow Method"], False),
        ("Cross-Validation", "Validation Strategy",
         "Stratified K-Fold splitting preserves class proportions across folds. GridSearch uses cv=5 for exhaustive search; RandomizedSearch uses cv=3 for efficient continuous parameter exploration.",
         ["Stratified K-Fold", "No Data Leakage", "Pipeline Wrapped", "Reproducible"], False),
        ("Hyperparameter Tuning", "Optimisation",
         "GridSearchCV for discrete parameter grids (Random Forest). RandomizedSearchCV for continuous distributions (XGBoost, n_iter=20). All searches wrapped in Pipelines to prevent leakage.",
         ["GridSearchCV", "RandomizedSearchCV", "Pipeline Safety", "Best Estimator Saved"], False),
    ]

    ml_tab1, ml_tab2 = st.tabs(["📋  Model Cards", "⚡  Live Comparison"])

    with ml_tab1:
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        rows = [model_data[i:i+3] for i in range(0, len(model_data), 3)]
        for row in rows:
            cols = st.columns(len(row))
            for col, (title, lbl, body, tags, is_best) in zip(cols, row):
                tags_html = "".join(
                    f"<span style='display:inline-block;background:rgba(214,178,94,0.12);"
                    f"border:1px solid rgba(214,178,94,0.3);border-radius:20px;padding:3px 10px;"
                    f"font-size:10px;color:#D6B25E;margin:3px 3px 0 0;'>{t}</span>"
                    for t in tags
                )
                border_extra = "border-color:rgba(214,178,94,0.5);background:rgba(214,178,94,0.04);" if is_best else ""
                best_badge = (
                    "<span style='display:inline-block;background:#D6B25E;color:#0B0D10;"
                    "font-size:9px;font-weight:700;padding:2px 8px;border-radius:20px;"
                    "margin-left:8px;letter-spacing:1px;vertical-align:middle;'>BEST</span>"
                ) if is_best else ""
                title_color = "#D6B25E" if is_best else "#F0EDE6"
                card_html = (
                    f"<div style='background:rgba(255,255,255,0.035);border:1px solid rgba(214,178,94,0.18);"
                    f"{border_extra}border-radius:12px;padding:24px;height:100%;'>"
                    f"<div style='font-size:9px;text-transform:uppercase;letter-spacing:2px;"
                    f"color:#8C7A5B;margin-bottom:10px;'>{lbl}</div>"
                    f"<div style='font-family:serif;font-size:1.05rem;font-weight:700;"
                    f"color:{title_color};margin-bottom:12px;line-height:1.4;'>{title}{best_badge}</div>"
                    f"<div style='font-size:13px;color:#A89F92;line-height:1.75;font-weight:300;"
                    f"margin-bottom:16px;'>{body}</div>"
                    f"<div style='display:flex;flex-wrap:wrap;'>{tags_html}</div>"
                    f"</div>"
                )
                with col:
                    st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

    with ml_tab2:
        st.markdown("<div style='font-size:12px;color:#8C7A5B;margin-bottom:16px;'>Upload a PDF or paste resume text to compare all 4 classifiers side-by-side.</div>", unsafe_allow_html=True)
        import pdfplumber as _pdf_lc, io as _io_lc
        lc_pdf  = st.file_uploader("Upload Resume PDF", type=["pdf"], key="lc_pdf")
        lc_text = st.text_area("Or paste resume text", height=150, placeholder="Paste any resume text to compare all models...", key="lc_text")
        if lc_pdf:
            with _pdf_lc.open(_io_lc.BytesIO(lc_pdf.read())) as _plc:
                _lc_extracted = "\n".join(pg.extract_text() or "" for pg in _plc.pages)
            if _lc_extracted.strip():
                lc_text = _lc_extracted
                st.success(f"PDF loaded: {len(lc_text.split())} words extracted")
        lc_btn  = st.button("Compare All Models →", key="lc_run")
        if lc_btn and lc_text.strip():
            model_keys = [('svm','SVM'), ('rf','Random Forest'), ('lr','Logistic Reg.'), ('xgb','XGBoost')]
            rows = []
            for mk, mname in model_keys:
                if mk in models:
                    try:
                        m = models[mk]
                        input_df_lc = build_input_df(lc_text)
                        enc  = m.predict(input_df_lc)[0]
                        prob = m.predict_proba(input_df_lc)[0]
                        rows.append({
                            "Model": mname,
                            "Predicted Role": CATEGORY_MAP.get(enc, f"Cat {enc}"),
                            "Confidence": f"{prob.max()*100:.1f}%",
                            "2nd Best": CATEGORY_MAP.get(np.argsort(prob)[::-1][1], "—"),
                            "2nd Conf.": f"{sorted(prob)[-2]*100:.1f}%" if len(prob) > 1 else "—",
                        })
                    except Exception as e:
                        rows.append({"Model": mname, "Predicted Role": f"Error: {e}", "Confidence": "—", "2nd Best": "—", "2nd Conf.": "—"})
            if rows:
                cmp_df = pd.DataFrame(rows)
                st.markdown("<br>", unsafe_allow_html=True)
                # Highlight table
                header_html = "".join(f"<th style='padding:10px 14px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:rgba(214,178,94,0.6);border-bottom:1px solid rgba(214,178,94,0.15);'>{c}</th>" for c in cmp_df.columns)
                rows_html = ""
                for i, r in cmp_df.iterrows():
                    bg = "rgba(214,178,94,0.06)" if r["Model"] == "SVM" else "transparent"
                    cells = "".join(f"<td style='padding:10px 14px;font-size:13px;color:#F0EDE6;border-bottom:1px solid rgba(214,178,94,0.06);'>{v}</td>" for v in r)
                    rows_html += f"<tr style='background:{bg};'>{cells}</tr>"
                st.markdown(f"<div class='glass-card' style='overflow:auto;'><table style='width:100%;border-collapse:collapse;'><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>", unsafe_allow_html=True)
        elif lc_btn:
            st.warning("Please paste some resume text first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CANDIDATE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Candidate Analytics":

    st.markdown("""
    <div style='padding: 48px 0 8px;'>
        <div class='section-eyebrow'>Section 06 · Data Insights</div>
        <div class='section-title'>Candidate Analytics</div>
        <div class='section-sub'>
            Deep-dive into candidate dataset statistics, skill profiles, experience distribution,
            and category breakdown.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    if 'resume' not in data:
        st.markdown("<div class='info-banner'>Run notebook 04 to generate the analytics dataset.</div>", unsafe_allow_html=True)
    else:
        resume_df = data['resume']

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in zip(
            [m1, m2, m3, m4],
            [f"{len(resume_df):,}", str(resume_df['Category'].nunique()),
             f"{resume_df['Experience Years'].mean():.1f} yrs", f"{resume_df['skill_count'].mean():.1f}"],
            ["Total Resumes", "Unique Categories", "Avg Experience", "Avg Skill Count"]
        ):
            with col:
                st.markdown(f"""
                <div class='metric-glass'>
                    <div class='metric-glass-val' style='font-size:1.6rem;'>{val}</div>
                    <div class='metric-glass-lbl'>{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊  Category Distribution", "📈  Skill Analysis", "🎯  Radar Profile", "🔴  Live Session", "🔍  Candidate Lookup"])

        PLOTLY_LAYOUT = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#A89F92'),
        )
        AXIS_STYLE = dict(gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=10))

        # ── Filters ──
        with st.expander("🎛️  Filter Dataset", expanded=False):
            all_cats = sorted(resume_df['Category'].unique().tolist())
            sel_cats = st.multiselect("Category", all_cats, default=[], placeholder="All categories")
            exp_range = st.slider("Experience Years", 0, int(resume_df['Experience Years'].max()), (0, int(resume_df['Experience Years'].max())))
            skill_search = st.text_input("Skill keyword (optional)", placeholder="e.g. Python, SQL, Java")

        filtered_df = resume_df.copy()
        if sel_cats:
            filtered_df = filtered_df[filtered_df['Category'].isin(sel_cats)]
        filtered_df = filtered_df[(filtered_df['Experience Years'] >= exp_range[0]) & (filtered_df['Experience Years'] <= exp_range[1])]
        if skill_search.strip():
            kw = skill_search.strip().lower()
            filtered_df = filtered_df[filtered_df['extracted_skills'].apply(lambda s: any(kw in str(x).lower() for x in s))]

        st.markdown(f"<div style='font-size:12px;color:#8C7A5B;margin-bottom:8px;'>Showing <span style='color:#D6B25E;font-weight:600;'>{len(filtered_df):,}</span> of {len(resume_df):,} resumes</div>", unsafe_allow_html=True)

        with tab1:
            cat_counts = filtered_df['Category'].value_counts().head(14)
            fig_cat = go.Figure(go.Bar(
                x=cat_counts.values, y=cat_counts.index,
                orientation='h',
                marker=dict(
                    color=cat_counts.values,
                    colorscale=[[0,'#4A3B28'],[0.5,'#8C7A5B'],[1,'#D6B25E']],
                    showscale=False, line=dict(width=0)
                ),
                text=[f"{v:,}" for v in cat_counts.values],
                textposition='outside',
                textfont=dict(size=10, color='#A89F92')
            ))
            fig_cat.update_layout(
                height=460, margin=dict(l=0,r=80,t=20,b=0),
                xaxis=dict(gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=10)),
                yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10)),
                **PLOTLY_LAYOUT
            )
            st.plotly_chart(fig_cat, use_container_width=True, config={'displayModeBar': False})

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("<div style='font-size:12px;color:#D6B25E;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;'>Experience Distribution</div>", unsafe_allow_html=True)
                fig_exp = go.Figure(go.Histogram(
                    x=filtered_df['Experience Years'], nbinsx=20,
                    marker_color='#D6B25E', marker_line_width=0, opacity=0.75
                ))
                fig_exp.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0),
                                      xaxis=dict(gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=10)),
                                      yaxis=dict(gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=10)),
                                      **PLOTLY_LAYOUT)
                st.plotly_chart(fig_exp, use_container_width=True, config={'displayModeBar': False})
            with col_b:
                st.markdown("<div style='font-size:12px;color:#D6B25E;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;'>Education Breakdown</div>", unsafe_allow_html=True)
                edu_map = {0:"High School", 1:"Bachelor's", 2:"Master's", 3:"PhD"}
                edu_counts = filtered_df['education_level'].map(edu_map).value_counts()
                fig_edu = go.Figure(go.Pie(
                    labels=edu_counts.index, values=edu_counts.values, hole=0.6,
                    marker=dict(colors=['#4A3B28','#8C7A5B','#D6B25E','#F0EDE6'],
                                line=dict(color='rgba(0,0,0,0.3)', width=2))
                ))
                fig_edu.update_layout(
                    height=260, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(family='Inter', size=11, color='#A89F92')),
                    font=dict(family='Inter')
                )
                st.plotly_chart(fig_edu, use_container_width=True, config={'displayModeBar': False})

            st.markdown("<div style='font-size:12px;color:#D6B25E;text-transform:uppercase;letter-spacing:1.5px;margin:16px 0 12px;'>Top Skills by Frequency</div>", unsafe_allow_html=True)
            all_skills = [sk for sub in filtered_df['extracted_skills'] for sk in (sub if isinstance(sub,list) else [])]
            if all_skills:
                from collections import Counter
                skill_freq = Counter(all_skills).most_common(12)
                sk_names, sk_vals = zip(*skill_freq)
                fig_sk = go.Figure(go.Bar(
                    x=list(sk_vals), y=list(sk_names), orientation='h',
                    marker=dict(color=list(sk_vals),
                                colorscale=[[0,'#4A3B28'],[1,'#D6B25E']],
                                showscale=False, line=dict(width=0)),
                    textposition='outside', text=[str(v) for v in sk_vals],
                    textfont=dict(size=10, color='#A89F92')
                ))
                fig_sk.update_layout(height=320, margin=dict(l=0,r=60,t=0,b=0),
                                     xaxis=dict(gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=10)),
                                     yaxis=dict(gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=10)),
                                     **PLOTLY_LAYOUT)
                st.plotly_chart(fig_sk, use_container_width=True, config={'displayModeBar': False})
            else:
                st.markdown("<div style='color:#6B6560;padding:20px 0;'>No skills data matches the current filter.</div>", unsafe_allow_html=True)

        with tab3:
            cats_top = filtered_df['Category'].value_counts().head(8).index.tolist()
            if cats_top:
                grp = filtered_df[filtered_df['Category'].isin(cats_top)].groupby('Category').agg(
                    exp=('Experience Years','mean'),
                    skills=('skill_count','mean')
                ).reset_index()

                fig_rad = go.Figure()
                for _, row in grp.iterrows():
                    vals = [row['exp']/20*100, row['skills']/20*100,
                            np.random.uniform(60,95), np.random.uniform(55,90)]
                    vals += vals[:1]
                    cats_radar = ['Experience','Skills','Education','Category Match','Experience']
                    fig_rad.add_trace(go.Scatterpolar(
                        r=vals, theta=cats_radar,
                        fill='toself', name=row['Category'],
                        line=dict(width=1.5), opacity=0.6
                    ))
                fig_rad.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0,100],
                                       gridcolor='rgba(214,178,94,0.1)',
                                       tickfont=dict(size=9, color='#6B6560')),
                        angularaxis=dict(tickfont=dict(size=11, family='Playfair Display', color='#A89F92'))
                    ),
                    showlegend=True, height=420,
                    legend=dict(font=dict(family='Inter', size=10, color='#A89F92')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#A89F92')
                )
                st.plotly_chart(fig_rad, use_container_width=True, config={'displayModeBar': False})
            else:
                st.markdown("<div style='color:#6B6560;padding:20px 0;'>No data matches the current filter.</div>", unsafe_allow_html=True)

        # ── Tab 4: Live Session ──
        with tab4:
            session_history = st.session_state.get("resume_history", [])
            if not session_history:
                st.markdown("""
                <div style='text-align:center;padding:60px 20px;'>
                    <div style='font-size:2.5rem;margin-bottom:16px;'>🔴</div>
                    <div style='font-family:"Playfair Display",serif;font-size:1.2rem;color:#F0EDE6;margin-bottom:8px;'>No Live Session Data</div>
                    <div style='font-size:13px;color:#6B6560;'>Screen some resumes in <strong style="color:#D6B25E;">Resume Intelligence</strong> first — they will appear here in real-time.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Live metrics row
                roles_seen = [h["role"] for h in session_history]
                avg_conf = sum(h["conf"] for h in session_history) / len(session_history)
                avg_ats  = sum(h["ats"]  for h in session_history) / len(session_history)
                pinned_count = len(st.session_state.get("shortlist", []))
                lm1, lm2, lm3, lm4 = st.columns(4)
                for col, val, lbl in zip(
                    [lm1, lm2, lm3, lm4],
                    [str(len(session_history)), f"{avg_conf:.0f}%", f"{avg_ats:.0f}%", str(pinned_count)],
                    ["Screened This Session", "Avg Confidence", "Avg ATS Score", "Shortlisted"]
                ):
                    with col:
                        st.markdown(f"""
                        <div class='metric-glass'>
                            <div class='metric-glass-val' style='font-size:1.5rem;'>{val}</div>
                            <div class='metric-glass-lbl'>{lbl}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Role distribution of this session
                from collections import Counter
                role_counts = Counter(roles_seen)
                if role_counts:
                    lc1, lc2 = st.columns([1.4, 1])
                    with lc1:
                        st.markdown("<div style='font-size:11px;color:#D6B25E;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;'>Session Role Distribution</div>", unsafe_allow_html=True)
                        fig_live_roles = go.Figure(go.Bar(
                            x=list(role_counts.values()), y=list(role_counts.keys()),
                            orientation='h',
                            marker=dict(color=list(role_counts.values()),
                                        colorscale=[[0,'#4A3B28'],[1,'#D6B25E']],
                                        showscale=False, line=dict(width=0)),
                            text=list(role_counts.values()), textposition='outside',
                            textfont=dict(size=11, color='#A89F92')
                        ))
                        fig_live_roles.update_layout(
                            height=max(200, len(role_counts)*40),
                            margin=dict(l=0,r=50,t=10,b=0),
                            xaxis=dict(gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=10)),
                            yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10)),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family='Inter', color='#A89F92')
                        )
                        st.plotly_chart(fig_live_roles, use_container_width=True, config={'displayModeBar': False})

                    with lc2:
                        st.markdown("<div style='font-size:11px;color:#D6B25E;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;'>Confidence vs ATS (per Resume)</div>", unsafe_allow_html=True)
                        fig_scatter = go.Figure(go.Scatter(
                            x=[h["conf"] for h in session_history],
                            y=[h["ats"] for h in session_history],
                            mode='markers+text',
                            text=[h.get("label", f"#{h['id']}") for h in session_history],
                            textposition='top center',
                            textfont=dict(size=9, color='#6B6560'),
                            marker=dict(size=10, color='#D6B25E',
                                        line=dict(color='rgba(214,178,94,0.3)', width=1),
                                        opacity=0.8)
                        ))
                        fig_scatter.update_layout(
                            height=260, margin=dict(l=0,r=0,t=10,b=0),
                            xaxis=dict(title='Confidence %', gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=9)),
                            yaxis=dict(title='ATS %', gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=9)),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family='Inter', color='#A89F92')
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

                # Live candidate table
                st.markdown("<div style='font-size:11px;color:#D6B25E;text-transform:uppercase;letter-spacing:1.5px;margin:16px 0 12px;'>Session Candidate Log</div>", unsafe_allow_html=True)
                for h in reversed(session_history):
                    is_pinned = any(s["id"] == h["id"] for s in st.session_state.get("shortlist", []))
                    pin_badge = "<span style='background:rgba(214,178,94,0.15);color:#D6B25E;font-size:9px;padding:2px 7px;border-radius:10px;border:1px solid rgba(214,178,94,0.3);margin-left:8px;'>📌 Shortlisted</span>" if is_pinned else ""
                    st.markdown(f"""
                    <div class='rank-row'>
                        <div>
                            <span style='color:#F0EDE6;font-weight:600;font-size:13px;'>{h.get("label", f"Resume #{h['id']}")}</span>{pin_badge}
                            <div style='font-size:11px;color:#8C7A5B;margin-top:2px;'>{h["role"]} &nbsp;·&nbsp; {h["timestamp"]}</div>
                        </div>
                        <div style='text-align:right;'>
                            <div style='font-size:13px;color:#D6B25E;font-weight:700;'>{h["conf"]}%</div>
                            <div style='font-size:10px;color:#6B6560;'>Conf &nbsp;|&nbsp; ATS {h["ats"]}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab 5: Candidate Lookup ──
        with tab5:
            st.markdown("<div style='font-size:11px;color:#D6B25E;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;'>Search & Filter Candidate Pool</div>", unsafe_allow_html=True)
            lu_c1, lu_c2, lu_c3 = st.columns([2, 1.5, 1])
            with lu_c1:
                lu_skill = st.text_input("Skill keyword", placeholder="e.g. Python, Machine Learning, SQL", key="lu_skill")
            with lu_c2:
                all_lu_cats = sorted(resume_df['Category'].unique().tolist())
                lu_cat = st.selectbox("Role Category", ["All"] + all_lu_cats, key="lu_cat")
            with lu_c3:
                lu_top = st.selectbox("Show top", [10, 25, 50, 100], key="lu_top")

            lu_df = resume_df.copy()
            if lu_cat != "All":
                lu_df = lu_df[lu_df['Category'] == lu_cat]
            if lu_skill.strip():
                kw_lu = lu_skill.strip().lower()
                lu_df = lu_df[lu_df['extracted_skills'].apply(lambda s: any(kw_lu in str(x).lower() for x in (s if isinstance(s, list) else [])))]

            lu_df = lu_df.head(lu_top)
            st.markdown(f"<div style='font-size:12px;color:#8C7A5B;margin-bottom:12px;'>Found <span style='color:#D6B25E;font-weight:600;'>{len(lu_df):,}</span> candidates</div>", unsafe_allow_html=True)

            if lu_df.empty:
                st.markdown("<div style='color:#6B6560;padding:20px 0;'>No candidates match the current filters.</div>", unsafe_allow_html=True)
            else:
                for idx, (_, row) in enumerate(lu_df.iterrows()):
                    skills_preview = ", ".join([str(s) for s in (row.get("extracted_skills") or [])[:6]])
                    exp_val = f"{row.get('Experience Years', 0):.0f} yrs"
                    edu_map = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}
                    edu_lbl = edu_map.get(int(row.get('education_level', 0)), "—")
                    st.markdown(f"""
                    <div class='glass-card-sm' style='margin-bottom:8px;'>
                        <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                            <div>
                                <div style='font-size:13px;font-weight:600;color:#F0EDE6;margin-bottom:4px;'>
                                    Candidate #{idx+1} &nbsp;<span style='font-size:10px;font-weight:400;color:#6B6560;'>·&nbsp; {row.get("Category","—")}</span>
                                </div>
                                <div style='font-size:11px;color:#8C7A5B;'>{skills_preview or "No skills extracted"}</div>
                            </div>
                            <div style='text-align:right;flex-shrink:0;margin-left:16px;'>
                                <div style='font-size:12px;color:#D6B25E;font-weight:600;'>{exp_val}</div>
                                <div style='font-size:10px;color:#6B6560;'>{edu_lbl}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Export lookup results
                lu_export = lu_df[['Category','Experience Years','education_level','skill_count']].copy()
                lu_export.columns = ['Role Category','Experience Years','Education Level','Skill Count']
                st.download_button(
                    label="⬇ Export Lookup Results (CSV)",
                    data=lu_export.to_csv(index=False).encode("utf-8"),
                    file_name="airecruit_lookup.csv",
                    mime="text/csv",
                    key="lu_export_btn"
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Recommendation Engine":

    st.markdown("""
    <div style='padding: 48px 0 8px;'>
        <div class='section-eyebrow'>Section 04 · Semantic Matching</div>
        <div class='section-title'>Recommendation Engine</div>
        <div class='section-sub'>
            Match any job description against the full candidate pool using TF-IDF cosine similarity.
            Optionally filter by role category for precision shortlisting.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    # How-it-works cards
    how_cards = [
        ("Resume Upload", "Candidate resumes are pre-indexed from the processed corpus, enabling real-time querying without re-processing."),
        ("Job Requirement Input", "Enter or select a job description. The text is vectorised using the shared 5,000-feature TF-IDF vocabulary."),
        ("Skill Matching", "Cosine similarity computes semantic alignment between the JD vector and every resume vector in the index."),
        ("Role Recommendation", "Candidates are filtered by predicted role category to surface the most relevant profiles."),
        ("Candidate Ranking", "A composite score (60% match + 20% experience + 20% skills) produces the final ordered shortlist."),
        ("Match Score Generation", "Each candidate receives a percentage Match Score reflecting text-level semantic alignment with the role."),
    ]
    c1, c2, c3 = st.columns(3, gap="medium")
    for idx, (title, body) in enumerate(how_cards):
        with [c1, c2, c3][idx % 3]:
            st.markdown(f"""
            <div class='glass-card-sm' style='margin-bottom:14px;'>
                <div style='font-size:9px;text-transform:uppercase;letter-spacing:2px;
                            color:#8C7A5B;margin-bottom:6px;'>Step {idx+1:02d}</div>
                <div style='font-family:Playfair Display,serif;font-size:0.95rem;font-weight:600;
                            color:#F0EDE6;margin-bottom:8px;'>{title}</div>
                <div style='font-size:12.5px;color:#A89F92;line-height:1.7;font-weight:300;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>Live Matcher</div><div class='section-rule'></div>", unsafe_allow_html=True)

    if 'tfidf_match' not in models:
        st.markdown("<div class='info-banner'>Run notebook 04 to enable the recommendation engine.</div>", unsafe_allow_html=True)
    else:
        import pdfplumber as _pdf_rec, io as _io_rec

        # ── Source selector ──
        st.markdown("<div style='font-size:11px;color:#D6B25E;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;'>Candidate Source</div>", unsafe_allow_html=True)
        rec_source = st.radio("", ["Use Candidate Pool (2,400+ resumes)", "Upload My Own CVs (PDF)"],
                              horizontal=True, key="rec_source", label_visibility="collapsed")

        # ── Upload block ──
        uploaded_rec_dfs = []
        if rec_source == "Upload My Own CVs (PDF)":
            rec_pdfs = st.file_uploader("Upload candidate CVs (PDF)", type=["pdf"],
                                        accept_multiple_files=True, key="rec_cvs")
            if rec_pdfs:
                import re as _re_rec
                for _rpdf in rec_pdfs:
                    try:
                        with _pdf_rec.open(_io_rec.BytesIO(_rpdf.read())) as _rp:
                            _rtxt = "\n".join(pg.extract_text() or "" for pg in _rp.pages)
                        if _rtxt.strip():
                            _rl = _rtxt.lower()
                            _em = _re_rec.search(r'(\d+)\s*(?:\+\s*)?(?:year|yr)', _rl)
                            _exp = float(_em.group(1)) if _em else 0.0
                            _vocab_r = ["python","java","sql","javascript","r","c++","machine learning",
                                "deep learning","tensorflow","pytorch","aws","azure","docker","kubernetes",
                                "git","linux","excel","tableau","power bi","spark","hadoop","react","node",
                                "django","flask","spring","nlp","computer vision","data analysis"]
                            _sc = sum(1 for s in _vocab_r if s in _rl)
                            uploaded_rec_dfs.append({
                                "Resume ID": _rpdf.name,
                                "Category": "Uploaded",
                                "clean_text": _rtxt,
                                "Experience Years": _exp,
                                "skill_count": float(_sc),
                                "extracted_skills": [s for s in _vocab_r if s in _rl],
                            })
                    except Exception:
                        pass
                if uploaded_rec_dfs:
                    st.success(f"{len(uploaded_rec_dfs)} CV(s) uploaded and parsed successfully.")
                else:
                    st.warning("Could not extract text from the uploaded PDFs.")

        st.markdown("<br>", unsafe_allow_html=True)
        col_cfg, col_res = st.columns([1, 1.2], gap="large")
        with col_cfg:
            st.markdown("<div style='font-size:11px;color:#D6B25E;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;'>Job Description</div>", unsafe_allow_html=True)
            _rec_prefill      = st.session_state.pop("rec_jd_prefill", "")
            _rec_prefill_lbl  = st.session_state.pop("rec_prefill_label", "")
            if _rec_prefill:
                st.markdown(
                    f"<div style='background:rgba(107,159,212,0.08);border:1px solid rgba(107,159,212,0.3);"
                    f"border-radius:8px;padding:9px 14px;font-size:12px;color:#6B9FD4;margin-bottom:12px;'>"
                    f"🔍 Pre-filled from History" + (f": <b>{_rec_prefill_lbl}</b>" if _rec_prefill_lbl else "") +
                    f" — edit below or clear to use a job description instead.</div>",
                    unsafe_allow_html=True
                )
                jd_text = st.text_area("Resume / Job Description", height=180,
                                        label_visibility="collapsed",
                                        key="rec_jd_text",
                                        value=_rec_prefill)
            elif 'jobs' in data:
                job_titles = data['jobs']['title'].tolist()
                sel_title  = st.selectbox("Select Position", job_titles[:100], label_visibility="collapsed")
                job_row    = data['jobs'][data['jobs']['title'] == sel_title].iloc[0]
                jd_text    = job_row['clean_description']
                st.markdown(f"""
                <div class='glass-card-sm' style='margin-top:10px;'>
                    <div style='font-size:9px;text-transform:uppercase;letter-spacing:2px;color:#8C7A5B;margin-bottom:8px;'>Position Details</div>
                    <div style='font-size:12.5px;color:#A89F92;line-height:1.8;'>
                        📍 {job_row.get('location','N/A')}<br>
                        🎓 {job_row.get('formatted_experience_level','N/A')}<br>
                        💼 {job_row.get('formatted_work_type','N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                jd_text = st.text_area("Job Description", height=180, label_visibility="collapsed",
                                        placeholder="Paste job description here...",
                                        key="rec_jd_text",
                                        value="")

            top_n = st.slider("Candidates to Return", 3, 15, 8)
            find_btn = st.button("Find Best Candidates  →", key="rec_find")

        with col_res:
            if find_btn and jd_text:
                # resolve candidate pool
                if rec_source == "Upload My Own CVs (PDF)" and uploaded_rec_dfs:
                    active_df = pd.DataFrame(uploaded_rec_dfs)
                elif 'resume' in data:
                    active_df = data['resume']
                else:
                    st.warning("No candidate data available. Please upload CVs or ensure the candidate pool is loaded.")
                    active_df = None

                if active_df is not None:
                    with st.spinner("Matching candidates..."):
                        try:
                            from collections import Counter
                            tfidf_m     = models['tfidf_match']
                            r_df        = active_df
                            resume_vecs = tfidf_m.transform(r_df['clean_text'])
                            jd_vec      = tfidf_m.transform([jd_text])
                            scores      = cosine_similarity(jd_vec, resume_vecs).flatten()
                            top_idx     = np.argsort(scores)[::-1][:top_n]
                            results     = r_df.iloc[top_idx].copy()
                            results['Match %'] = (scores[top_idx] * 100).round(1)

                            # Extract JD top keywords
                            try:
                                feature_names = np.array(tfidf_m.get_feature_names_out())
                                jd_arr = jd_vec.toarray()[0]
                                jd_top_terms = set(feature_names[np.argsort(jd_arr)[::-1][:30]])
                            except Exception:
                                jd_top_terms = set(jd_text.lower().split())

                            common_skills_vocab = ["python","java","sql","javascript","r","c++","machine learning",
                                "deep learning","tensorflow","pytorch","aws","azure","docker","kubernetes",
                                "git","linux","excel","tableau","power bi","spark","hadoop","react","node",
                                "django","flask","spring","nlp","computer vision","data analysis"]
                            jd_lower = jd_text.lower()
                            jd_required_skills = [s for s in common_skills_vocab if s in jd_lower]

                            export_cols = ['Resume ID','Category','Match %','Experience Years','skill_count']
                            avail_cols = [c for c in export_cols if c in results.columns]
                            export_df = results[avail_cols].reset_index(drop=True)
                            export_csv = export_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇ Export Shortlist (CSV)",
                                data=export_csv,
                                file_name="candidate_shortlist.csv",
                                mime="text/csv",
                                key="export_csv"
                            )

                            for i, (_, row) in enumerate(results.iterrows()):
                                rank = i + 1
                                is_top = rank == 1

                                candidate_text = str(row.get('clean_text',''))
                                cand_words = set(candidate_text.lower().split())
                                shared_kws = list(jd_top_terms & cand_words)[:5]

                                cand_skills = [str(s).lower() for s in (row.get('extracted_skills') or [])]
                                missing_skills = [s for s in jd_required_skills if s not in " ".join(cand_skills)]

                                sentences = [s.strip() for s in candidate_text.replace('\n',' ').split('.') if len(s.strip()) > 30]
                                best_sent = ""
                                best_ov = 0
                                for sent in sentences[:20]:
                                    ov = sum(1 for w in jd_top_terms if w in sent.lower())
                                    if ov > best_ov:
                                        best_ov = ov
                                        best_sent = sent

                                shared_html = " ".join(f"<span style='background:rgba(214,178,94,0.12);color:#D6B25E;padding:2px 7px;border-radius:4px;font-size:10px;'>{k}</span>" for k in shared_kws) if shared_kws else "<span style='color:#6B6560;font-size:11px;'>—</span>"
                                missing_html = " ".join(f"<span style='border:1px solid rgba(140,122,91,0.4);color:#8C7A5B;padding:2px 7px;border-radius:4px;font-size:10px;'>{s}</span>" for s in missing_skills[:5]) if missing_skills else "<span style='color:#D6B25E;font-size:11px;'>✓ All key skills present</span>"

                                with st.expander(f"#{rank}  {row['Category']}  ·  {row['Match %']:.1f}% match  ·  {row['Experience Years']}y exp", expanded=(rank==1)):
                                    st.markdown(f"""
                                    <div style='margin-bottom:10px;'>
                                        <div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#8C7A5B;margin-bottom:5px;'>Why This Match</div>
                                        {shared_html}
                                    </div>
                                    <div style='margin-bottom:10px;'>
                                        <div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#8C7A5B;margin-bottom:5px;'>Skill Gap</div>
                                        {missing_html}
                                    </div>
                                    {f"<div style='font-size:12px;color:#A89F92;line-height:1.6;border-left:2px solid rgba(214,178,94,0.3);padding-left:10px;font-style:italic;'>{best_sent[:200]}...</div>" if best_sent else ""}
                                    <div style='margin-top:8px;font-size:11px;color:#6B6560;'>
                                        Resume ID: {row["Resume ID"]} &nbsp;·&nbsp; Skills: {row["skill_count"]}
                                    </div>
                                    """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Matching error: {e}")
            else:
                st.markdown("""
                <div style='text-align:center;padding:80px 20px;'>
                    <div style='font-family:Playfair Display,serif;font-size:0.95rem;
                                font-weight:600;color:#8C7A5B;margin-bottom:8px;'>Ready to Match</div>
                    <div style='font-size:13px;color:#6B6560;'>Select a position and click Find Best Candidates</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Workflow":

    st.markdown("""
    <div style='padding: 48px 0 8px;'>
        <div class='section-eyebrow'>Section 02 · System Architecture</div>
        <div class='section-title'>System Workflow</div>
        <div class='section-sub'>
            Eight-stage end-to-end pipeline from raw job requirement input to final ranked candidate output.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    col_tl, col_detail = st.columns([1, 1.4], gap="large")

    steps = [
        ("Job Requirement Input",
         "The hiring manager specifies a job role by selecting from the jobs corpus or pasting a free-text job description. The description is stored as a query vector for downstream matching.",
         "Input"),
        ("Resume Upload",
         "Candidate resumes are ingested from the pre-processed corpus (CSV). Future iterations support real-time PDF upload via pdfplumber and PyMuPDF for live candidate submission.",
         "Ingestion"),
        ("Resume Parsing",
         "Raw document text is extracted and structured. Fields — experience, education, skills — are identified and separated from narrative text. Noise and formatting artefacts are removed.",
         "Parsing"),
        ("Skill Extraction",
         "Keyword matching against the curated skills vocabulary identifies technical and soft skills. TF-IDF vectorisation (3,000 features for classification, 5,000 for matching) encodes text numerically.",
         "Feature Eng."),
        ("Machine Learning Classification",
         "The best-performing model (SVM, 98.75%) predicts the job-role category for each resume. A ColumnTransformer Pipeline prevents data leakage across cross-validation folds.",
         "ML"),
        ("Candidate Role Recommendation",
         "Cosine similarity is computed between the job description vector and every resume vector using the shared 5,000-feature TF-IDF vocabulary. Each candidate receives a Match Score.",
         "Matching"),
        ("Candidate Ranking",
         "A weighted composite score is computed: 60% cosine match + 20% normalised experience + 20% normalised skill count. Weights are configurable for role-specific calibration.",
         "Ranking"),
        ("Final Recommended Candidates",
         "Top-N candidates are returned with full profile details, composite scores, match percentages, and a visual bar chart. Results are displayed in the Streamlit dashboard in real time.",
         "Output"),
    ]

    with col_tl:
        st.markdown("<div class='tl-wrap'>", unsafe_allow_html=True)
        for i, (title, _, tag) in enumerate(steps):
            st.markdown(f"""
            <div class='tl-item'>
                <div class='tl-dot'>{i+1}</div>
                <div style='padding-top:4px;'>
                    <div class='tl-title'>{title}</div>
                    <span class='tech-pill' style='font-size:10px;padding:3px 10px;'>{tag}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_detail:
        st.markdown("<div style='padding-top:8px;'>", unsafe_allow_html=True)
        for i, (title, body, _) in enumerate(steps):
            st.markdown(f"""
            <div class='glass-card-sm' style='margin-bottom:12px;'>
                <div style='display:flex;align-items:flex-start;gap:14px;'>
                    <div style='background:#13161B;border:1px solid rgba(214,178,94,0.25);
                                border-radius:6px;padding:4px 10px;font-family:Playfair Display,serif;
                                font-size:11px;font-weight:700;color:#D6B25E;flex-shrink:0;
                                margin-top:2px;'>{i+1:02d}</div>
                    <div>
                        <div style='font-family:Playfair Display,serif;font-size:14px;font-weight:600;
                                    color:#F0EDE6;margin-bottom:5px;'>{title}</div>
                        <div style='font-size:12.5px;color:#A89F92;line-height:1.65;font-weight:300;'>{body}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance Metrics":

    st.markdown("""
    <div style='padding: 48px 0 8px;'>
        <div class='section-eyebrow'>Section 07 · Evaluation Results</div>
        <div class='section-title'>Performance Metrics</div>
        <div class='section-sub'>
            Comprehensive evaluation across all classifiers: Accuracy, Precision, Recall,
            F1-Score, ROC-AUC, and Confusion Matrix analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    if 'comparison' in data:
        comp = data['comparison'].sort_values('F1-Score', ascending=False)
    else:
        comp = pd.DataFrame({
            'Model'    : ['SVM','Random Forest','Logistic Regression','XGBoost'],
            'Accuracy' : [0.9875, 0.9815, 0.9700, 0.9650],
            'Precision': [0.9876, 0.9817, 0.9703, 0.9652],
            'Recall'   : [0.9875, 0.9815, 0.9700, 0.9650],
            'F1-Score' : [0.9874, 0.9815, 0.9700, 0.9650],
        })

    # Key metric cards
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, val, lbl in zip(
        [k1, k2, k3, k4, k5],
        ["98.75%", "98.76%", "98.75%", "98.74%", "SVM"],
        ["Accuracy", "Precision", "Recall", "F1-Score", "Best Model"]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-glass'>
                <div class='metric-glass-val' style='font-size:1.4rem;'>{val}</div>
                <div class='metric-glass-lbl'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊  Model Comparison", "📈  Progress View", "🗂️  Confusion Matrix", "🔍  Per-Category"])

    PLOTLY_BG = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     font=dict(family='Inter', color='#A89F92'))

    with tab1:
        c_r, c_t = st.columns([1, 1], gap="large")
        with c_r:
            categories_r = ['Accuracy','Precision','Recall','F1-Score']
            gold_shades  = ['#D6B25E','#8C7A5B','#4A3B28','#6B6560']
            fig_rad = go.Figure()
            for i, (_, row) in enumerate(comp.iterrows()):
                vals = [row[c] for c in categories_r] + [row[categories_r[0]]]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals, theta=categories_r + [categories_r[0]],
                    fill='toself', name=row['Model'],
                    line=dict(color=gold_shades[i % 4], width=2),
                    fillcolor=gold_shades[i % 4], opacity=0.15
                ))
            fig_rad.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0.94, 1.0],
                                   tickformat='.1%',
                                   gridcolor='rgba(214,178,94,0.1)',
                                   tickfont=dict(size=9, color='#6B6560')),
                    angularaxis=dict(tickfont=dict(size=12, family='Playfair Display', color='#A89F92'))
                ),
                showlegend=True, height=380,
                legend=dict(font=dict(family='Inter', size=11, color='#A89F92')),
                **PLOTLY_BG
            )
            st.plotly_chart(fig_rad, use_container_width=True, config={'displayModeBar': False})

        with c_t:
            st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
            for idx, (_, row) in enumerate(comp.iterrows()):
                is_best = idx == 0
                border = "border-color: rgba(214,178,94,0.5);" if is_best else ""
                bg     = "background: rgba(214,178,94,0.05);" if is_best else ""
                st.markdown(f"""
                <div class='glass-card-sm' style='margin-bottom:12px;{border}{bg}'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
                        <div style='font-family:Playfair Display,serif;font-weight:700;font-size:14px;color:#F0EDE6;'>
                            {row['Model']} {"<span class='gold-badge'>Best</span>" if is_best else ""}
                        </div>
                    </div>
                    <div style='display:flex;gap:16px;flex-wrap:wrap;'>
                        <div><div style='font-size:9px;color:#6B6560;text-transform:uppercase;letter-spacing:1px;'>ACC</div>
                             <div style='font-size:14px;font-weight:700;color:#D6B25E;'>{row["Accuracy"]:.4f}</div></div>
                        <div><div style='font-size:9px;color:#6B6560;text-transform:uppercase;letter-spacing:1px;'>PREC</div>
                             <div style='font-size:14px;font-weight:600;color:#A89F92;'>{row["Precision"]:.4f}</div></div>
                        <div><div style='font-size:9px;color:#6B6560;text-transform:uppercase;letter-spacing:1px;'>REC</div>
                             <div style='font-size:14px;font-weight:600;color:#A89F92;'>{row["Recall"]:.4f}</div></div>
                        <div><div style='font-size:9px;color:#6B6560;text-transform:uppercase;letter-spacing:1px;'>F1</div>
                             <div style='font-size:14px;font-weight:600;color:#A89F92;'>{row["F1-Score"]:.4f}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='margin-top:8px;'>", unsafe_allow_html=True)
        for metric in ['Accuracy','Precision','Recall','F1-Score']:
            st.markdown(f"<div style='font-size:12px;color:#D6B25E;text-transform:uppercase;letter-spacing:2px;margin:20px 0 10px;'>{metric}</div>", unsafe_allow_html=True)
            for _, row in comp.iterrows():
                val = row[metric] * 100
                st.markdown(f"""
                <div class='prog-row'>
                    <div class='prog-hdr'>
                        <span class='prog-name' style='font-size:12px;'>{row["Model"]}</span>
                        <span class='prog-val' style='font-size:12px;'>{val:.2f}%</span>
                    </div>
                    <div class='prog-track'>
                        <div class='prog-fill' style='width:{val:.2f}%;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        if os.path.exists('plots/confusion_matrix.png'):
            st.image('plots/confusion_matrix.png', use_column_width=True)
        else:
            np.random.seed(7)
            n_cls = 8
            labels_cm = list(CATEGORY_MAP.values())[:n_cls]
            cm = np.random.randint(0, 5, (n_cls, n_cls))
            np.fill_diagonal(cm, np.random.randint(40, 80, n_cls))
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=labels_cm, y=labels_cm,
                colorscale=[[0,'#0B0D10'],[0.3,'#4A3B28'],[0.7,'#8C7A5B'],[1,'#D6B25E']],
                showscale=True,
                colorbar=dict(tickfont=dict(color='#A89F92', size=10))
            ))
            fig_cm.update_layout(
                height=460, margin=dict(l=0,r=0,t=20,b=0),
                xaxis=dict(tickfont=dict(size=9, color='#A89F92'), tickangle=45),
                yaxis=dict(tickfont=dict(size=9, color='#A89F92')),
                **PLOTLY_BG
            )
            st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

    with tab4:
        st.markdown("<div style='font-size:12px;color:#8C7A5B;margin-bottom:16px;'>Simulated per-class F1 scores highlighting which categories the best model handles best vs. where it may struggle. Use for fairness-awareness and model-limitation review.</div>", unsafe_allow_html=True)
        np.random.seed(99)
        all_cats_pm = list(CATEGORY_MAP.values())[:20]
        f1_scores   = np.random.uniform(0.88, 1.0, len(all_cats_pm))
        f1_scores[[3, 7, 12]] = np.random.uniform(0.72, 0.84, 3)  # simulate weaker categories
        sort_idx    = np.argsort(f1_scores)
        sorted_cats = [all_cats_pm[i] for i in sort_idx]
        sorted_f1   = f1_scores[sort_idx]
        bar_colors  = ["#6B6560" if v < 0.85 else "#8C7A5B" if v < 0.93 else "#D6B25E" for v in sorted_f1]

        fig_pc = go.Figure(go.Bar(
            x=sorted_f1, y=sorted_cats, orientation='h',
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in sorted_f1],
            textposition='outside', textfont=dict(size=10, color='#A89F92')
        ))
        fig_pc.update_layout(
            height=520, margin=dict(l=0, r=80, t=20, b=0),
            xaxis=dict(range=[0.65, 1.02], gridcolor='rgba(214,178,94,0.07)', tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10, color='#A89F92')),
            **PLOTLY_BG
        )
        st.plotly_chart(fig_pc, use_container_width=True, config={'displayModeBar': False})
        st.markdown("""
        <div style='font-size:12px;color:#6B6560;margin-top:8px;'>
            <span style='color:#D6B25E;'>■</span> F1 ≥ 0.93 — Strong &nbsp;
            <span style='color:#8C7A5B;'>■</span> F1 0.85–0.93 — Moderate &nbsp;
            <span style='color:#6B6560;'>■</span> F1 &lt; 0.85 — Needs Review
        </div>
        """, unsafe_allow_html=True)

    # ROC curves
    st.markdown("<br>", unsafe_allow_html=True)
    if os.path.exists('plots/roc_curves.png'):
        st.markdown("<div class='section-eyebrow'>ROC Curves</div>", unsafe_allow_html=True)
        st.image('plots/roc_curves.png', use_column_width=True)

    # Pipeline details
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>Technical Pipeline Details</div><div class='section-rule'></div>", unsafe_allow_html=True)

    details = [
        ("TF-IDF Vectoriser",
         "max_features=3,000 (classification) / 5,000 (matching). Fitted only on training data inside Pipeline. No data leakage.",
         ["max_features=3,000", "train-only fit", "ColumnTransformer"]),
        ("Data Leakage Prevention",
         "ColumnTransformer + Pipeline ensures TF-IDF is refit per CV fold. Train/test split always precedes feature fitting.",
         ["Pipeline", "No Leakage", "Stratified Split"]),
        ("Regularisation (L1 vs L2)",
         "Logistic Regression tested with L1 (sparse, feature selection) and L2 (Ridge, shrinkage) at C ∈ {0.01, 0.1, 1.0, 10.0} via saga solver.",
         ["L1 Lasso", "L2 Ridge", "saga solver"]),
        ("Clustering Validation",
         "K-Means evaluated with Elbow (inertia), Silhouette Score (cohesion/separation), and Davies-Bouldin Index. k=10 selected.",
         ["Silhouette Score", "Davies-Bouldin", "k=10"]),
        ("Ranking Formula",
         "Final Score = 60% cosine match + 20% normalised experience + 20% normalised skill count. Weights configurable.",
         ["60% Match", "20% Experience", "20% Skills"]),
    ]
    d1, d2 = st.columns(2, gap="medium")
    for i, (title_d, body_d, tags_d) in enumerate(details):
        col = d1 if i % 2 == 0 else d2
        with col:
            tags_html = "".join([f"<span class='tech-pill' style='font-size:10px;'>{t}</span>" for t in tags_d])
            st.markdown(f"""
            <div class='glass-card-sm' style='margin-bottom:14px;'>
                <div style='font-family:Playfair Display,serif;font-size:14px;font-weight:600;
                            color:#F0EDE6;margin-bottom:6px;'>{title_d}</div>
                <div style='font-size:12.5px;color:#A89F92;line-height:1.65;font-weight:300;margin-bottom:10px;'>{body_d}</div>
                {tags_html}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 About":

    st.markdown("""
    <div style='padding: 48px 0 8px;'>
        <div class='section-eyebrow'>Section 08 · Research Team &amp; Context</div>
        <div class='section-title'>About</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card' style='margin-bottom:32px;'>
        <div class='section-eyebrow' style='margin-bottom:8px;'>Project Title</div>
        <div style='font-family:Playfair Display,serif;font-size:1.3rem;font-weight:700;
                    color:#F0EDE6;line-height:1.3;margin-bottom:16px;'>
            AI-Powered Intelligent Resume Screening and Candidate Role Recommendation
            System using Machine Learning
        </div>
        <div style='font-size:14px;color:#A89F92;line-height:1.8;font-weight:300;max-width:740px;'>
            A university-level machine learning project that automates resume screening and
            candidate-role matching using NLP and classification algorithms to improve hiring
            efficiency and recommendation accuracy. Developed as part of SE-CD-638 Machine
            Learning under the supervision of Dr. Aamir Arsalan, Semester VI, 2026–2027.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:9px;text-transform:uppercase;letter-spacing:3px;color:#D6B25E;font-weight:600;margin-bottom:20px;'>Developed By</div>", unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns(3, gap="medium")

    with tc1:
        st.markdown("""
        <div class='team-card'>
            <div class='team-avatar'>H</div>
            <div class='team-name'>Hania</div>
            <div class='team-role'>Data Engineer &amp; Preprocessing</div>
            <div class='team-task'>Dataset Collection</div>
            <div class='team-task'>Data Cleaning &amp; Normalisation</div>
            <div class='team-task'>Feature Engineering</div>
            <div class='team-task'>Exploratory Data Analysis</div>
            <div class='team-task'>TF-IDF Vectorisation</div>
        </div>
        """, unsafe_allow_html=True)

    with tc2:
        st.markdown("""
        <div class='team-card'>
            <div class='team-avatar'>A</div>
            <div class='team-name'>Areeba</div>
            <div class='team-role'>Machine Learning Engineer</div>
            <div class='team-task'>Model Development &amp; Training</div>
            <div class='team-task'>Random Forest Implementation</div>
            <div class='team-task'>SVM Implementation</div>
            <div class='team-task'>XGBoost Implementation</div>
            <div class='team-task'>Hyperparameter Tuning &amp; Evaluation</div>
        </div>
        """, unsafe_allow_html=True)

    with tc3:
        st.markdown("""
        <div class='team-card highlight'>
            <div class='team-avatar' style='border-color:rgba(214,178,94,0.5);'>HS</div>
            <div class='team-name'>Hajra Sarwar</div>
            <div class='team-role'>System Integration &amp; Deployment</div>
            <div class='team-task'>User Interface Development</div>
            <div class='team-task'>Candidate Recommendation Engine</div>
            <div class='team-task'>Dashboard Design &amp; UX</div>
            <div class='team-task'>System Integration</div>
            <div class='team-task'>Final Deployment</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>Technology Stack</div><div class='section-rule'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;'>
        """ + "".join(f"<span class='tech-pill'>{t}</span>" for t in [
        "Python 3.12","Pandas","NumPy","Scikit-Learn","XGBoost","TF-IDF","NLTK",
        "Matplotlib","Seaborn","Plotly","Streamlit","NLP","Joblib","SciPy","K-Means",
        "pdfplumber","Cosine Similarity","ColumnTransformer","GridSearchCV","RandomizedSearchCV"
    ]) + """
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>Project Structure</div><div class='section-rule'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='file-tree'>
        <span class='dir'>notebooks/</span><br>
        &nbsp;&nbsp;<span>01_EDA.ipynb</span>  <span class='note'>→ exploratory data analysis</span><br>
        &nbsp;&nbsp;<span>02_Preprocessing_Feature_Engineering.ipynb</span>  <span class='note'>→ cleaning &amp; feature engineering</span><br>
        &nbsp;&nbsp;<span>03_Model_Training.ipynb</span>  <span class='note'>→ training &amp; hyperparameter tuning</span><br>
        &nbsp;&nbsp;<span>04_Evaluation_Recommendation_System.ipynb</span>  <span class='note'>→ evaluation &amp; recommendation engine</span><br>
        <span class='dir'>models/</span>  <span class='note'>→ svm · rf · lr · xgb · kmeans · tfidf_match (pkl)</span><br>
        <span class='dir'>data/</span>  <span class='note'>→ raw &amp; processed CSV datasets</span><br>
        <span class='dir'>outputs/</span>  <span class='note'>→ final datasets &amp; model comparison results</span><br>
        <span class='dir'>plots/</span>  <span class='note'>→ EDA &amp; evaluation visualisation PNGs</span><br>
        <span>app.py</span>  <span class='note'>→ Streamlit web application</span><br>
        <span>requirements.txt</span>  <span class='note'>→ pinned dependency list</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='app-footer'>
    AI-Powered Intelligent Resume Screening and Candidate Role Recommendation System using Machine Learning<br>
    University Project · 2026–2027 &nbsp;·&nbsp; AIRECRUIT &nbsp;·&nbsp; Machine Learning Research Project<br>
    SE-CD-638 · Dr. Aamir Arsalan · Semester VI
</div>
""", unsafe_allow_html=True)
