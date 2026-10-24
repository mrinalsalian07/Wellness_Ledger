"""
app.py
------
Wellness Ledger — financial wellness + recurring monthly check-ins
+ savings goals.
"""

import pandas as pd
import streamlit as st
import joblib
import shap
import plotly.graph_objects as go
import html
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

from train_model import engineer_features, compute_wellness_score, FEATURE_COLUMNS
import auth
import user_data


# -----------------------------------------------------------------
# Modern visual theme
# -----------------------------------------------------------------
ACTIVE_THEME = "lavender"

THEMES = {
    "sage": {
        "ink": "#18313B",
        "paper": "#F5F1E8",
        "card": "#FFFDF8",
        "line": "#DDD8CC",
        "accent": "#B58A36",
        "accent_soft": "#EEF0E7",
        "good": "#4E765B",
        "risk": "#B45B55",
        "muted": "#6B6F69",
    },
    "lavender": {
        "ink": "#333044",
        "paper": "#FAF8FF",
        "card": "#FFFFFF",
        "line": "#E4DDF2",
        "accent": "#7558B7",
        "accent_soft": "#E9E0FF",
        "good": "#4E7A64",
        "risk": "#B85C64",
        "muted": "#756F86",
    },
    "plum": {
        "ink": "#2D2535",
        "paper": "#F7F3F8",
        "card": "#FFFFFF",
        "line": "#E4DCE8",
        "accent": "#8C5A83",
        "accent_soft": "#EFE3EE",
        "good": "#4E7A64",
        "risk": "#B85C64",
        "muted": "#756D78",
    },
    "ocean": {
        "ink": "#18313D",
        "paper": "#F2F7F8",
        "card": "#FFFFFF",
        "line": "#D7E3E7",
        "accent": "#2F8291",
        "accent_soft": "#DDEEF0",
        "good": "#39745A",
        "risk": "#B85C50",
        "muted": "#65757C",
    },
    "terracotta": {
        "ink": "#302B28",
        "paper": "#F8F2EB",
        "card": "#FFFDFC",
        "line": "#E5D8CD",
        "accent": "#B86B4B",
        "accent_soft": "#F2E2D8",
        "good": "#587052",
        "risk": "#B9574D",
        "muted": "#756D67",
    },
}

C = THEMES[ACTIVE_THEME]
GOOD = C["good"]
BRASS = C["accent"]
RISK = C["risk"]

MODERN_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
:root {{
  --bg:{C['paper']}; --card:{C['card']}; --ink:{C['ink']}; --muted:{C['muted']};
  --line:{C['line']}; --accent:{C['accent']}; --soft:{C['accent_soft']};
}}

html, body, [class*="css"] {{
  font-family:'DM Sans',sans-serif !important;
  color:{C['ink']} !important;
  font-size:16px !important;
}}
.stApp {{ background:{C['paper']} !important; }}
#MainMenu, footer {{ visibility:hidden !important; }}
header[data-testid="stHeader"] {{ background:transparent !important; height:0 !important; }}
.block-container {{ max-width:1000px !important; padding:1.15rem 1.35rem 4rem !important; }}

/* ---------- Brand/header ---------- */
.ledger-topbar {{
  display:flex; align-items:center; justify-content:space-between;
  border-bottom:1px solid {C['line']}; padding:.15rem 0 .75rem;
  margin-bottom:.55rem;
}}
.ledger-topbar .brand {{
  font-family:'DM Serif Display',serif !important; font-size:1.7rem !important;
  font-weight:400 !important; color:{C['ink']} !important; letter-spacing:-.025em;
}}

/* ---------- Main navigation ---------- */
.ledger-nav-space {{ height:.1rem; }}
.ledger-nav-space + div[data-testid="stHorizontalBlock"] {{ gap:.55rem !important; margin-bottom:.8rem !important; }}
.ledger-nav-space + div[data-testid="stHorizontalBlock"] .stButton > button {{
  min-height:2.45rem !important; padding:.4rem .8rem !important;
  border:1px solid {C['line']} !important; border-radius:10px !important;
  background:{C['card']} !important; color:{C['ink']} !important;
  box-shadow:0 1px 4px rgba(24,49,59,.035) !important;
  font-size:.79rem !important; font-weight:700 !important;
}}
.ledger-nav-space + div[data-testid="stHorizontalBlock"] .stButton > button:hover {{
  border-color:{C['accent']} !important; color:{C['accent']} !important;
}}

/* ---------- Typography ---------- */
.ledger-section {{
  color:{C['ink']} !important; font-family:'DM Serif Display',serif !important;
  font-size:1.48rem !important; margin:1.35rem 0 .65rem !important;
  padding:0 !important; border:0 !important;
}}
.page-heading {{ margin:.95rem 0 1.1rem !important; }}
.page-heading h1 {{
  color:{C['ink']} !important; font-family:'DM Serif Display',serif !important;
  font-size:2.15rem !important; font-weight:400 !important;
  letter-spacing:-.025em; margin:0 !important;
}}
.page-heading p {{ color:{C['muted']} !important; font-size:.86rem !important; margin:.25rem 0 0 !important; }}
.dashboard-kicker {{
  color:{C['muted']} !important; text-transform:uppercase; letter-spacing:.09em;
  font-size:.65rem !important; font-weight:700;
}}

/* ---------- Cards ---------- */
.ledger-dashboard-score, .ledger-dashboard-empty, .ledger-dashboard-focus,
.ledger-goal-card, .history-row, .empty-card, .factor-card, .score-trend-card {{
  background:{C['card']} !important; border:1px solid {C['line']} !important;
  border-radius:14px !important;
  box-shadow:0 5px 18px rgba(24,49,59,.045) !important;
}}
.ledger-dashboard-score {{ padding:1.15rem 1.2rem !important; margin:.3rem 0 .75rem !important; }}
.ledger-dashboard-focus {{ border-left:3px solid {C['accent']} !important; padding:.85rem .95rem !important; background:{C['accent_soft']} !important; }}
.ledger-dashboard-empty {{ padding:1.25rem !important; }}
.score-layout {{ display:flex; align-items:center; justify-content:space-between; gap:1.25rem; }}
.score-big {{ font-family:'DM Serif Display',serif; font-size:3.55rem; line-height:.95; color:{C['ink']}; }}
.score-band {{ color:{C['accent']}; font-weight:700; font-size:.86rem; }}
.score-caption {{ color:{C['muted']}; font-size:.75rem; margin-top:.3rem; }}
.ledger-score-number {{
  display:inline-block; font-family:'DM Serif Display',serif !important;
  color:{C['ink']} !important; font-size:3.5rem !important; line-height:1 !important;
  margin:.2rem .45rem .75rem 0 !important;
}}
.ledger-score-band {{
  display:inline-block; color:{C['accent']} !important; font-size:1rem !important;
  font-weight:700 !important; vertical-align:baseline;
}}
.ledger-factor-row {{
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  background:{C['card']} !important; border:1px solid {C['line']} !important;
  border-radius:10px !important; padding:.7rem .85rem !important; margin-top:.5rem !important;
}}
.ledger-factor-label {{ color:{C['ink']} !important; font-weight:700 !important; }}
.ledger-factor-value {{ color:{C['ink']} !important; font-weight:600 !important; }}
.ledger-factor-points {{ color:{C['muted']} !important; font-size:.78rem !important; margin-left:.35rem !important; }}
.ledger-factor-advice {{
  color:{C['muted']} !important; font-size:.82rem !important; line-height:1.5 !important;
  padding:.35rem .85rem .65rem !important;
}}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.4rem; }}
.dashboard-empty-title {{ font-family:'DM Serif Display',serif; color:{C['ink']}; font-size:1.25rem; margin-top:.3rem; }}

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {{
  background:{C['card']} !important; border:1px solid {C['line']} !important;
  border-radius:12px !important; padding:.85rem .9rem !important;
  box-shadow:none !important; min-height:72px !important;
}}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] * {{ color:{C['muted']} !important; opacity:1 !important; font-size:.7rem !important; font-weight:600 !important; }}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{
  color:{C['ink']} !important; opacity:1 !important; font-size:1.5rem !important; line-height:1.15 !important;
}}
div[data-testid="stMetricDelta"] * {{ color:{C['muted']} !important; }}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {{ color:{C['muted']} !important; opacity:1 !important; }}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{ color:{C['ink']} !important; opacity:1 !important; }}

/* ---------- Inputs ---------- */
div[data-testid="stNumberInput"], div[data-testid="stSelectbox"], div[data-testid="stTextInput"] {{ background:transparent !important; }}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label {{
  color:{C['ink']} !important; opacity:1 !important; font-size:.74rem !important;
  font-weight:600 !important; margin-bottom:.2rem !important;
}}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {{
  background:{C['card']} !important;
  color:{C['ink']} !important;
  -webkit-text-fill-color:{C['ink']} !important;
  caret-color:{C['ink']} !important;
  border:1px solid {C['line']} !important;
  border-radius:9px !important;
  min-height:2.25rem !important; height:2.25rem !important;
  padding:.4rem .6rem !important; font-size:.84rem !important;
  box-shadow:none !important; outline:none !important;
  opacity:1 !important;
}}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {{
  border-color:{C['accent']} !important;
  box-shadow:0 0 0 2px {C['accent_soft']} !important;
}}
div[data-testid="stNumberInput"] button {{
  background:{C['card']} !important; color:{C['ink']} !important;
  border:0 !important; opacity:.8 !important;
}}
div[data-testid="stNumberInput"] button:hover {{ color:{C['accent']} !important; }}
div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
  background:{C['card']} !important; border:1px solid {C['line']} !important;
  border-radius:9px !important; min-height:2.25rem !important; box-shadow:none !important;
}}
div[data-testid="stSelectbox"] div[data-baseweb="select"] * {{ color:{C['ink']} !important; font-size:.84rem !important; }}

/* ---------- Check-in sections ---------- */
.checkin-group-title {{
  display:flex; align-items:baseline; gap:.55rem;
  background:{C['card']} !important; border:1px solid {C['line']} !important;
  border-radius:11px !important; padding:.7rem .9rem !important;
  margin:1.05rem 0 .35rem !important; color:{C['ink']} !important;
  font-family:'DM Serif Display',serif !important; font-size:1.08rem !important;
}}
.checkin-group-title small {{
  font-family:'DM Sans',sans-serif !important; color:{C['muted']} !important;
  font-size:.68rem !important; font-weight:400 !important;
}}

/* ---------- Explanations / recommendations ---------- */
.factor-card {{ padding:.9rem !important; min-height:100px; }}
.factor-title {{ font-weight:700; font-size:.8rem; color:{C['ink']}; }}
.factor-status {{ color:{C['accent']}; font-size:.68rem; font-weight:700; margin-top:.18rem; }}
.factor-copy {{ color:{C['muted']}; font-size:.72rem; line-height:1.45; margin-top:.4rem; }}
.ledger-live-note {{ color:{C['muted']} !important; font-size:.76rem !important; line-height:1.5 !important; margin:.45rem 0 !important; }}
.ledger-live-note * {{ color:{C['muted']} !important; }}

/* ---------- Goals & history ---------- */
.ledger-goal-card {{ padding:.9rem 1rem !important; margin:.4rem 0 !important; }}
.ledger-progress-track {{ height:6px !important; background:{C['accent_soft']} !important; border-radius:99px !important; margin:.4rem 0 !important; }}
.ledger-progress-fill {{ background:{C['accent']} !important; border-radius:99px !important; }}
.goal-name {{ font-size:.82rem !important; font-weight:700 !important; }}
.goal-meta,.goal-category,.goal-card-bottom {{ color:{C['muted']} !important; font-size:.69rem !important; }}
.history-row {{ padding:.85rem .95rem !important; margin:.4rem 0 !important; }}
.history-month {{ color:{C['ink']} !important; font-weight:700 !important; font-size:.82rem !important; }}
.history-numbers {{ color:{C['muted']} !important; font-size:.74rem !important; }}

/* ---------- Buttons ---------- */
.stButton > button, .stFormSubmitButton > button {{
  background:{C['ink']} !important; color:#fff !important; border:0 !important;
  border-radius:9px !important; min-height:2.35rem !important;
  font-size:.79rem !important; font-weight:700 !important; padding:.4rem .8rem !important;
  box-shadow:0 2px 7px rgba(24,49,59,.08) !important;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
  background:{C['accent']} !important; color:#fff !important;
}}

/* ---------- Alerts / messages ---------- */
div[data-testid="stAlert"] {{ border-radius:10px !important; font-size:.78rem !important; }}

/* ---------- Charts ---------- */
.score-trend-card {{ padding:.85rem !important; color:{C['ink']} !important; }}
.score-trend-card .dashboard-kicker {{ color:{C['muted']} !important; }}
.score-trend-card .score-caption {{ color:{C['muted']} !important; }}

/* ---------- Readability across Streamlit controls ---------- */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, [data-testid="stCaptionContainer"] * {{
  color:{C['ink']} !important;
  opacity:1 !important;
  line-height:1.55 !important;
}}
.stCaption, [data-testid="stCaptionContainer"] * {{ color:{C['muted']} !important; }}
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stCheckbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stSlider"] label {{
  color:{C['ink']} !important; opacity:1 !important; font-weight:600 !important;
}}
div[data-testid="stSelectbox"] [data-baseweb="select"] span,
div[data-testid="stMultiSelect"] [data-baseweb="select"] span {{
  color:{C['ink']} !important; -webkit-text-fill-color:{C['ink']} !important;
}}
div[data-testid="stCheckbox"] p, div[data-testid="stRadio"] p {{ color:{C['ink']} !important; opacity:1 !important; }}
div[data-testid="stExpander"] {{ border-color:{C['line']} !important; }}
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * {{ color:{C['ink']} !important; }}
div[data-testid="stDataFrame"] {{ border:1px solid {C['line']} !important; border-radius:10px !important; }}

/* ---------- Streamlit spacing ---------- */
div[data-testid="stVerticalBlock"] {{ gap:.42rem !important; }}
div[data-testid="stHorizontalBlock"] {{ gap:.65rem !important; }}
[data-testid="stElementToolbar"] {{ display:none !important; }}

/* ---------- Auth ---------- */
.ledger-cover {{
  background:{C['ink']} !important; border-radius:16px !important;
  padding:2rem 1.6rem !important; box-shadow:0 12px 32px rgba(24,49,59,.12);
  margin-bottom:.7rem !important;
}}
.ledger-cover .cover-mark {{
  color:#F5F1E8 !important; font-size:.7rem !important; font-weight:700 !important;
  letter-spacing:.16em !important; margin-bottom:.55rem !important;
}}
.ledger-cover .title {{ font-family:'DM Serif Display',serif !important; font-size:2.1rem !important; color:#FFFDF8 !important; line-height:1.05 !important; }}
.ledger-cover .rule {{ background:{C['accent']} !important; height:2px !important; width:42px !important; margin:.75rem 0 !important; }}
.ledger-cover .sub {{ color:rgba(255,253,248,.86) !important; font-size:.82rem !important; line-height:1.4 !important; }}

/* ---------- Login / signup controls ---------- */
div[data-testid="stRadio"] {{
  margin:.15rem 0 .75rem !important;
}}
div[data-testid="stRadio"] > label {{ display:none !important; }}
div[data-testid="stRadio"] div[role="radiogroup"] {{
  gap:.45rem !important; align-items:center !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label {{
  color:{C['ink']} !important; background:transparent !important;
  font-size:.78rem !important; font-weight:600 !important;
  padding:.35rem .7rem !important; border:1px solid transparent !important;
  border-radius:999px !important; cursor:pointer !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label p {{
  color:{C['ink']} !important; opacity:1 !important; margin:0 !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
  background:{C['accent_soft']} !important; border-color:{C['line']} !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {{
  color:{C['ink']} !important; font-weight:700 !important;
}}
div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {{
  display:none !important;
}}

div[data-testid="stTextInput"] label {{
  color:{C['ink']} !important; opacity:1 !important; font-size:.74rem !important;
  font-weight:600 !important;
}}
div[data-testid="stTextInput"] > div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="input"] {{
  background:{C['card']} !important; border:1px solid {C['line']} !important;
  border-radius:10px !important; box-shadow:none !important;
}}
div[data-testid="stTextInput"] > div[data-baseweb="input"]:focus-within,
div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {{
  border-color:{C['accent']} !important; box-shadow:0 0 0 2px {C['accent_soft']} !important;
}}
div[data-testid="stTextInput"] input {{
  color:{C['ink']} !important;
  -webkit-text-fill-color:{C['ink']} !important;
  caret-color:{C['ink']} !important;
  background:{C['card']} !important;
  border:0 !important;
  box-shadow:none !important;
  opacity:1 !important;
}}

div[data-testid="stTextInput"] input::placeholder {{
  color:#AEB8BA !important;
  -webkit-text-fill-color:#AEB8BA !important;
  opacity:1 !important;
}}

div[data-testid="stTextInput"] input:focus {{
  color:{C['ink']} !important;
  -webkit-text-fill-color:{C['ink']} !important;
  caret-color:{C['ink']} !important;
}}
div[data-testid="stTextInput"] button {{
  color:{C['muted']} !important; background:transparent !important; border:0 !important;
}}

div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="base-input"],
div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
div[data-testid="stTextInput"] div[data-baseweb="base-input"] > div {{
  background-color: #FFFDF8 !important;
}}
div[data-testid="stTextInput"] input,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextInput"] input:-webkit-autofill,
div[data-testid="stTextInput"] input:-webkit-autofill:hover,
div[data-testid="stTextInput"] input:-webkit-autofill:focus {{
  background-color: #FFFDF8 !important;
  color: #18313B !important;
  -webkit-text-fill-color: #18313B !important;
  caret-color: #18313B !important;
  opacity: 1 !important;
  box-shadow: 0 0 0 1000px #FFFDF8 inset !important;
}}

div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="base-input"] {{
  background:{C['card']} !important;
}}
div[data-testid="stTextInput"] input,
div[data-testid="stTextInput"] input:focus {{
  background:{C['card']} !important;
  color:{C['ink']} !important;
  -webkit-text-fill-color:{C['ink']} !important;
  caret-color:{C['ink']} !important;
  opacity:1 !important;
}}

.block-container {{ max-width:1120px !important; }}
.ledger-dashboard-score, .ledger-dashboard-empty, .ledger-dashboard-focus,
.ledger-goal-card, .history-row, .empty-card, .factor-card, .score-trend-card {{
  box-shadow:0 2px 10px rgba(51,48,68,.035) !important;
  border-radius:12px !important;
}}
.stButton > button {{ transition:background-color .15s ease, border-color .15s ease, color .15s ease; }}
.stButton > button:focus-visible {{ outline:2px solid {C['accent']} !important; outline-offset:2px !important; }}

@media(max-width:800px) {{
  .block-container {{ max-width:100% !important; padding:.85rem .8rem 3rem !important; }}
  .score-layout {{ display:block; }}
  .score-big {{ font-size:3rem; }}
}}
</style>
"""

st.set_page_config(
    page_title="Wellness Ledger",
    layout="wide",
)

st.markdown(MODERN_CSS, unsafe_allow_html=True)

APPEARANCE = st.session_state.get("ui_appearance", "Light")
if APPEARANCE == "Dark":
    APPEARANCE_CSS = """
    <style>
    :root { color-scheme: dark; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] { background:#171522 !important; color:#F2EEF9 !important; }
    .block-container, .stMarkdown, .stMarkdown p, .stMarkdown li,
    h1,h2,h3,h4,h5,h6,label,legend,[data-testid="stWidgetLabel"] * { color:#F2EEF9 !important; }
    .ledger-topbar { border-bottom-color:#3B354B !important; }
    .ledger-topbar .brand,.ledger-section,.page-heading h1,.score-big,
    .ledger-score-number,.ledger-factor-label,.ledger-factor-value,
    .dashboard-empty-title { color:#F2EEF9 !important; }
    .page-heading p,.dashboard-kicker,.score-caption,.ledger-factor-points,
    .ledger-factor-advice,.stCaption,[data-testid="stCaptionContainer"] * { color:#B8B0C9 !important; }
    .ledger-dashboard-score,.ledger-dashboard-empty,.ledger-dashboard-focus,
    .ledger-goal-card,.history-row,.empty-card,.factor-card,.score-trend-card,
    .ledger-factor-row,.checkin-group-title,div[data-testid="stMetric"] {
      background:#211E2D !important; border-color:#3B354B !important; box-shadow:none !important;
    }
    .ledger-dashboard-focus { background:#302743 !important; border-left-color:#B9A0F0 !important; }
    .stButton > button { background:#211E2D !important; color:#F2EEF9 !important; border-color:#3B354B !important; }
    .stButton > button[kind="primary"],.stButton > button[data-testid="baseButton-primary"] {
      background:#7558B7 !important; color:#FFFFFF !important; border-color:#7558B7 !important;
    }
    div[data-testid="stNumberInput"] input,div[data-testid="stTextInput"] input,
    div[data-testid="stDateInput"] input,div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stMultiSelect"] div[data-baseweb="select"],
    div[data-baseweb="input"],div[data-baseweb="base-input"] {
      background:#211E2D !important; color:#F2EEF9 !important; -webkit-text-fill-color:#F2EEF9 !important;
      caret-color:#F2EEF9 !important; border-color:#3B354B !important; box-shadow:none !important;
    }
    div[data-testid="stTextInput"] input:-webkit-autofill {
      -webkit-text-fill-color:#F2EEF9 !important; box-shadow:0 0 0 1000px #211E2D inset !important;
    }
    [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {
      background:#211E2D !important; color:#F2EEF9 !important;
    }
    [role="option"] { color:#F2EEF9 !important; }
    .score-trend-card * { color:#B8B0C9 !important; }
    .score-trend-card .trend-title { color:#B9A0F0 !important; }
    .ledger-dashboard-score *, .ledger-dashboard-empty *, .ledger-dashboard-focus *,
    .ledger-goal-card *, .history-row *, .empty-card *, .factor-card *,
    .ledger-factor-row *, .ledger-factor-advice *, .checkin-group-title * {
      color:#EAE5F4 !important; -webkit-text-fill-color:#EAE5F4 !important;
      opacity:1 !important;
    }
    .ledger-dashboard-score .dashboard-kicker,
    .ledger-dashboard-empty .dashboard-kicker,
    .ledger-dashboard-focus .dashboard-kicker,
    .ledger-goal-card .dashboard-kicker,
    .factor-card .dashboard-kicker,
    .ledger-factor-points { color:#B8B0C9 !important; -webkit-text-fill-color:#B8B0C9 !important; }
    .ledger-dashboard-score .score-band,
    .ledger-dashboard-score .ledger-score-band,
    .ledger-dashboard-score .score-band * { color:#B9A0F0 !important; -webkit-text-fill-color:#B9A0F0 !important; }
    .stButton > button[aria-label="☼"], .stButton > button[aria-label="☀"], .stButton > button[aria-label="☾"],
    button[title*="Switch to"] { min-width:0 !important; width:100% !important;
      height:2.5rem !important; padding:0 !important; white-space:nowrap !important; border-radius:10px !important;
      background:#302743 !important; color:#F2EEF9 !important; border:1px solid #514365 !important; }
    </style>
    """
else:
    APPEARANCE_CSS = """
    <style>
    :root { color-scheme: light; }
    .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
    [data-testid="stMainBlockContainer"] { background:#FAF8FF !important; color:#333044 !important; }
    .block-container,.stMarkdown,.stMarkdown p,.stMarkdown li,
    h1,h2,h3,h4,h5,h6,label,legend,[data-testid="stWidgetLabel"] * { color:#333044 !important; }
    .ledger-topbar { border-bottom-color:#E4DDF2 !important; }
    .ledger-topbar .brand,.ledger-section,.page-heading h1,.score-big,
    .ledger-score-number,.ledger-factor-label,.ledger-factor-value,
    .dashboard-empty-title { color:#333044 !important; }
    .page-heading p,.dashboard-kicker,.score-caption,.ledger-factor-points,
    .ledger-factor-advice,.stCaption,[data-testid="stCaptionContainer"] * { color:#756F86 !important; }
    .ledger-dashboard-score,.ledger-dashboard-empty,.ledger-dashboard-focus,
    .ledger-goal-card,.history-row,.empty-card,.factor-card,.score-trend-card,
    .ledger-factor-row,.checkin-group-title,div[data-testid="stMetric"] {
      background:#FFFFFF !important; border-color:#E4DDF2 !important; box-shadow:none !important;
    }
    .ledger-dashboard-focus { background:#F2ECFF !important; border-left-color:#7558B7 !important; }
    .stButton > button { background:#FFFFFF !important; color:#333044 !important; border-color:#E4DDF2 !important; }
    .stButton > button[kind="primary"],.stButton > button[data-testid="baseButton-primary"] {
      background:#7558B7 !important; color:#FFFFFF !important; border-color:#7558B7 !important;
    }
    div[data-testid="stNumberInput"] input,div[data-testid="stTextInput"] input,
    div[data-testid="stDateInput"] input,div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stMultiSelect"] div[data-baseweb="select"],
    div[data-baseweb="input"],div[data-baseweb="base-input"] {
      background:#FFFFFF !important; color:#333044 !important; -webkit-text-fill-color:#333044 !important;
      caret-color:#333044 !important; border-color:#D8CDEB !important; box-shadow:none !important;
    }
    div[data-testid="stTextInput"] input:-webkit-autofill {
      -webkit-text-fill-color:#333044 !important; box-shadow:0 0 0 1000px #FFFFFF inset !important;
    }
    [data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] { background:#FFFFFF !important; color:#333044 !important; }
    [role="option"] { color:#333044 !important; }
    .score-trend-card * { color:#756F86 !important; }
    .score-trend-card .trend-title { color:#7558B7 !important; }
    .stButton > button[aria-label="☼"], .stButton > button[aria-label="☀"], .stButton > button[aria-label="☾"],
    button[title*="Switch to"] { min-width:0 !important; width:100% !important;
      height:2.5rem !important; padding:0 !important; white-space:nowrap !important; border-radius:10px !important;
      background:#F2ECFF !important; color:#7558B7 !important; border:1px solid #E4DDF2 !important; }
    </style>
    """
st.markdown(APPEARANCE_CSS, unsafe_allow_html=True)

if APPEARANCE == "Dark":
    COVER_FIX_CSS = """
    <style>
    .ledger-cover { background:#211E2D !important; border-color:#3B354B !important; }
    .ledger-cover .cover-mark,.ledger-cover .title,.ledger-cover .sub {
        color:#F2EEF9 !important; -webkit-text-fill-color:#F2EEF9 !important; opacity:1 !important;
    }
    </style>
    """
else:
    COVER_FIX_CSS = """
    <style>
    .ledger-cover { background:#EEE7FA !important; border-color:#E4DDF2 !important; }
    .ledger-cover .cover-mark,.ledger-cover .title,.ledger-cover .sub {
        color:#333044 !important; -webkit-text-fill-color:#333044 !important; opacity:1 !important;
    }
    </style>
    """
st.markdown(COVER_FIX_CSS, unsafe_allow_html=True)

AUTH_CONTRAST_CSS = """
<style>
div[data-testid="stTextInput"] [data-baseweb="input"],
div[data-testid="stTextInput"] [data-baseweb="base-input"],
div[data-testid="stTextInput"] [data-baseweb="input"] > div,
div[data-testid="stTextInput"] [data-baseweb="base-input"] > div,
div[data-testid="stPassword"] [data-baseweb="input"],
div[data-testid="stPassword"] [data-baseweb="base-input"],
div[data-testid="stPassword"] [data-baseweb="input"] > div,
div[data-testid="stPassword"] [data-baseweb="base-input"] > div {
  background: var(--wl-input-bg) !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stPassword"] input {
  background: var(--wl-input-bg) !important;
  color: var(--wl-input-text) !important;
  -webkit-text-fill-color: var(--wl-input-text) !important;
  caret-color: var(--wl-input-text) !important;
}
.ledger-cover, .ledger-cover * { color: var(--wl-text) !important; }
.ledger-cover { background: var(--wl-card) !important; }
</style>
"""
if APPEARANCE == "Dark":
    AUTH_CONTRAST_CSS = AUTH_CONTRAST_CSS.replace("var(--wl-input-bg)", "#211E2D").replace("var(--wl-input-text)", "#F2EEF9").replace("var(--wl-text)", "#F2EEF9").replace("var(--wl-card)", "#211E2D")
else:
    AUTH_CONTRAST_CSS = AUTH_CONTRAST_CSS.replace("var(--wl-input-bg)", "#FFFFFF").replace("var(--wl-input-text)", "#333044").replace("var(--wl-text)", "#333044").replace("var(--wl-card)", "#FFFFFF")
st.markdown(AUTH_CONTRAST_CSS, unsafe_allow_html=True)

if APPEARANCE == "Dark":
    st.markdown(
        """<style>
        .ledger-cover { background:#211E2D !important; border-color:#3B354B !important; }
        .ledger-cover .cover-mark,.ledger-cover .title,.ledger-cover .sub {
            color:#F2EEF9 !important; -webkit-text-fill-color:#F2EEF9 !important; opacity:1 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """<style>
        .ledger-cover { background:#EEE7FA !important; border-color:#E4DDF2 !important; }
        .ledger-cover .cover-mark,.ledger-cover .title,.ledger-cover .sub {
            color:#333044 !important; -webkit-text-fill-color:#333044 !important; opacity:1 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <style>
    div[data-testid="stSelectbox"]:has(select[aria-label="Appearance"]),
    div[data-testid="stSelectbox"]:has(input[aria-label="Appearance"]) {
        max-width: 150px !important;
    }
    .ledger-cover {
        border: 1px solid var(--wl-line, #E4DDF2) !important;
    }
    .ledger-cover .cover-mark,
    .ledger-cover .title,
    .ledger-cover .sub {
        color: var(--wl-text, #333044) !important;
        -webkit-text-fill-color: var(--wl-text, #333044) !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if APPEARANCE == "Dark":
    st.markdown("""<style>
    .ledger-cover { background:#211E2D !important; border:1px solid #3B354B !important; }
    .ledger-cover, .ledger-cover * { color:#F2EEF9 !important; -webkit-text-fill-color:#F2EEF9 !important; opacity:1 !important; }
    .ledger-cover .sub { color:#B8B0C9 !important; -webkit-text-fill-color:#B8B0C9 !important; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
    .ledger-cover { background:#EEE7FA !important; border:1px solid #E4DDF2 !important; }
    .ledger-cover, .ledger-cover * { color:#333044 !important; -webkit-text-fill-color:#333044 !important; opacity:1 !important; }
    .ledger-cover .sub { color:#756F86 !important; -webkit-text-fill-color:#756F86 !important; }
    </style>""", unsafe_allow_html=True)

if APPEARANCE == "Dark":
    MENU_FIX_CSS = """
    <style>
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background:#211E2D !important; color:#F2EEF9 !important;
        border:1px solid #514365 !important;
    }
    [role="option"], [role="option"] *, [data-baseweb="menu"] li,
    [data-baseweb="menu"] li * {
        background:#211E2D !important; color:#F2EEF9 !important;
        -webkit-text-fill-color:#F2EEF9 !important; opacity:1 !important;
    }
    [role="option"][aria-selected="true"], [role="option"][aria-selected="true"] *,
    [role="option"]:hover, [role="option"]:hover * {
        background:#3A2E50 !important; color:#FFFFFF !important;
        -webkit-text-fill-color:#FFFFFF !important;
    }
    .stButton > button[aria-label="☼"], .stButton > button[aria-label="☀"], .stButton > button[aria-label="☾"] {
        width:2.35rem !important; min-width:2.35rem !important; max-width:2.35rem !important;
        height:2.35rem !important; padding:0 !important; border-radius:50% !important;
        font-size:1.35rem !important; line-height:1 !important;
        background:#302743 !important; color:#E6D9FF !important;
        border:1px solid #514365 !important; box-shadow:none !important;
    }
    </style>
    """
else:
    MENU_FIX_CSS = """
    <style>
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background:#FFFFFF !important; color:#333044 !important;
        border:1px solid #E4DDF2 !important;
    }
    [role="option"], [role="option"] *, [data-baseweb="menu"] li,
    [data-baseweb="menu"] li * {
        background:#FFFFFF !important; color:#333044 !important;
        -webkit-text-fill-color:#333044 !important; opacity:1 !important;
    }
    [role="option"][aria-selected="true"], [role="option"][aria-selected="true"] *,
    [role="option"]:hover, [role="option"]:hover * {
        background:#E9E0FF !important; color:#333044 !important;
        -webkit-text-fill-color:#333044 !important;
    }
    .stButton > button[aria-label="☼"], .stButton > button[aria-label="☀"], .stButton > button[aria-label="☾"] {
        width:2.35rem !important; min-width:2.35rem !important; max-width:2.35rem !important;
        height:2.35rem !important; padding:0 !important; border-radius:50% !important;
        font-size:1.35rem !important; line-height:1 !important;
        background:#F2ECFF !important; color:#7558B7 !important;
        border:1px solid #E4DDF2 !important; box-shadow:none !important;
    }
    </style>
    """
st.markdown(MENU_FIX_CSS, unsafe_allow_html=True)

THEME_TOGGLE_FIX_CSS = """
<style>
div.st-key-toggle_appearance_icon button {
    box-sizing: border-box !important;
    width: 4rem !important;
    min-width: 4rem !important;
    max-width: 4rem !important;
    height: 2.35rem !important;
    min-height: 2.35rem !important;
    max-height: 2.35rem !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 10px !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    opacity: 1 !important;
}
div.st-key-toggle_appearance_icon button p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
    font-size: 1.25rem !important;
    opacity: 1 !important;
}
div.st-key-toggle_appearance_icon button span {
    opacity: 1 !important;
}
div[data-testid="stHorizontalBlock"]:has(div.st-key-toggle_appearance_icon) > div:nth-child(3) button {
    white-space: nowrap !important;
    min-width: 4.5rem !important;
    width: 100% !important;
    height: 2.35rem !important;
    min-height: 2.35rem !important;
    padding: 0 .5rem !important;
}
</style>
"""
if APPEARANCE == "Dark":
    THEME_TOGGLE_FIX_CSS = THEME_TOGGLE_FIX_CSS.replace(
        "border-radius: 10px !important;",
        "border-radius: 10px !important; background: #302743 !important; border: 1px solid #514365 !important; color: #F2EEF9 !important; -webkit-text-fill-color: #F2EEF9 !important;"
    )
else:
    THEME_TOGGLE_FIX_CSS = THEME_TOGGLE_FIX_CSS.replace(
        "border-radius: 10px !important;",
        "border-radius: 10px !important; background: #F2ECFF !important; border: 1px solid #E4DDF2 !important; color: #7558B7 !important; -webkit-text-fill-color: #7558B7 !important;"
    )
st.markdown(THEME_TOGGLE_FIX_CSS, unsafe_allow_html=True)


@st.cache_resource
def initialize_database():
    user_data.init_db()
    return True

initialize_database()


# -----------------------------------------------------------------
# Model
# -----------------------------------------------------------------
@st.cache_resource
def load_model():
    project_dir = Path(__file__).resolve().parent
    model = joblib.load(project_dir / "wellness_model.joblib")
    feature_columns = joblib.load(project_dir / "feature_columns.joblib")
    explainer = shap.TreeExplainer(model)
    return model, feature_columns, explainer


try:
    model, feature_columns, explainer = load_model()
except Exception as exc:  # missing or outdated model files
    st.error(f"The scoring model is not ready: {exc}")
    st.info("Open a terminal in the project folder and run:  `python train_model.py`")
    st.stop()

# Newer train_model.py versions call the savings feature "savings_cushion";
# older ones call it "savings_rate". Support both so the app never breaks.
SAVINGS_KEY = "savings_cushion" if "savings_cushion" in feature_columns else "savings_rate"

# Spending above income can never be called "Moderate": the score is capped.
DEFICIT_SCORE_CAP = 49.0


BENCHMARKS = {
    "savings_rate": {
        "ideal": 0.5,
        "label": "Savings cushion (vs annual income)",
        "fmt": "x",
        "good_direction": "high",
    },
    "debt_to_income": {
        "ideal": 1.0,
        "label": "Debt-to-income (annual)",
        "fmt": "x",
        "good_direction": "low",
    },
    "debt_repayment_burden": {
        "ideal": 0.30,
        "label": "Debt repayment burden",
        "fmt": "pct",
        "good_direction": "low",
    },
    "expense_to_income": {
        "ideal": 0.70,
        "label": "Expense-to-income",
        "fmt": "pct",
        "good_direction": "low",
    },
    "emergency_fund_months": {
        "ideal": 6.0,
        "label": "Emergency fund",
        "fmt": "months",
        "good_direction": "high",
    },
    "discretionary_ratio": {
        "ideal": 0.30,
        "label": "Discretionary spend share",
        "fmt": "pct",
        "good_direction": "low",
    },
}


FEATURE_ADVICE = {
    "savings_rate": {
        "low": "Your savings balance is below the app's illustrative half-year-income cushion benchmark. Build a reserve gradually.",
        "high": "Your savings balance meets the app's illustrative half-year-income cushion benchmark.",
    },
    "debt_to_income": {
        "low": "Your total debt is more than {ideal:.1f}x your annual income. Pay down the highest-interest debt first before taking on more.",
        "high": "Your debt load relative to income is under control.",
    },
    "debt_repayment_burden": {
        "low": "Your EMIs/debt payments take up more than {ideal:.0%} of your monthly income, which limits flexibility.",
        "high": "Your debt repayments take up a manageable share of your income.",
    },
    "expense_to_income": {
        "low": "Your total expenses are above {ideal:.0%} of your income, leaving little room to save or absorb a shock.",
        "high": "You're keeping total expenses at a healthy share of income.",
    },
    "emergency_fund_months": {
        "low": "Your emergency fund covers less than {ideal:.0f} months of essential expenses. Build this before increasing discretionary spending.",
        "high": "Your emergency fund comfortably covers unexpected expenses.",
    },
    "discretionary_ratio": {
        "low": "A large share of your spending is discretionary rather than essential — usually the fastest lever to free up cash.",
        "high": "Your discretionary spending is well controlled relative to essentials.",
    },
    "income_stability": {
        "low": "You currently rely on a single income source, which adds risk if it's disrupted.",
        "high": "More than one income source adds a cushion against income shocks.",
    },
}


# -----------------------------------------------------------------
# Recommendation layer
# -----------------------------------------------------------------
RECOMMENDATION_RULES = {
    "savings_rate": (
        "Build a savings cushion gradually. Start with an amount you can sustain consistently, "
        "and consider setting aside a small amount each month."
    ),
    "debt_to_income": (
        "Focus on reducing outstanding debt, especially high-interest balances, "
        "before taking on additional borrowing."
    ),
    "debt_repayment_burden": (
        "Review your monthly debt payments and look for ways to reduce repayment "
        "pressure while keeping payments on schedule."
    ),
    "expense_to_income": (
        "Review your largest recurring expenses and identify one or two non-essential "
        "costs you can reduce this month."
    ),
    "emergency_fund_months": (
        "Build your emergency reserve gradually. A practical first milestone is "
        "one month of essential expenses, then work toward a larger cushion."
    ),
    "discretionary_ratio": (
        "Review discretionary spending and redirect part of it toward savings or "
        "debt reduction."
    ),
    "income_stability": (
        "Consider building a second reliable income source or strengthening your "
        "cash buffer to reduce dependence on a single source."
    ),
}


def _rename_savings_key(table):
    return {(SAVINGS_KEY if k == "savings_rate" else k): v for k, v in table.items()}


BENCHMARKS = _rename_savings_key(BENCHMARKS)
FEATURE_ADVICE = _rename_savings_key(FEATURE_ADVICE)
RECOMMENDATION_RULES = _rename_savings_key(RECOMMENDATION_RULES)


def generate_recommendations(feat, top_factors, max_items=3):
    """Turn important negative SHAP contributors into practical guidance."""
    recommendations = []

    for key, shap_value in top_factors:
        if shap_value >= 0 or key not in RECOMMENDATION_RULES:
            continue

        value = float(feat.iloc[0][key])

        if key == "income_stability":
            value_text = "1 income source" if value < 1 else "2+ income sources"
            label = "Income stability"
        else:
            value_text = format_value(key, value)
            label = BENCHMARKS.get(key, {}).get("label", key)

        recommendations.append({
            "feature": key,
            "label": label,
            "shap": float(shap_value),
            "value": value_text,
            "text": RECOMMENDATION_RULES[key],
        })

        if len(recommendations) >= max_items:
            break

    if not recommendations:
        recommendations.append({
            "feature": "general",
            "label": "Keep building",
            "shap": 0.0,
            "value": "",
            "text": (
                "Your strongest model contributors are currently positive. "
                "Keep your savings, spending and debt habits consistent and "
                "continue checking in each month."
            ),
        })

    return recommendations


def score_band(score):
    if score >= 75:
        return "Strong", GOOD
    if score >= 50:
        return "Moderate", BRASS
    return "At risk", RISK


def format_value(key, value):
    fmt = BENCHMARKS.get(key, {}).get("fmt")
    if fmt == "pct":
        return f"{value:.0%}"
    if fmt == "x":
        return f"{value:.1f}x"
    if fmt == "months":
        return f"{value:.1f} mo"
    return f"{value:.2f}"


def meets_benchmark(key, value):
    """True if the number is on the healthy side of its target."""
    if key == "income_stability":
        return value >= 1
    bench = BENCHMARKS[key]
    if bench["good_direction"] == "high":
        return value >= bench["ideal"]
    return value <= bench["ideal"]


def factor_label(key):
    return "Income stability" if key == "income_stability" else BENCHMARKS[key]["label"]


def inr(value):
    """Indian digit grouping: 1234567 -> ₹12,34,567."""
    n = int(round(float(value)))
    sign = "-" if n < 0 else ""
    digits = str(abs(n))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return f"{sign}₹{digits}"


def flash(message, kind="success"):
    """Show a message after the next rerun (st.success followed by st.rerun vanishes)."""
    st.session_state["_flash"] = (kind, message)


def show_flash():
    item = st.session_state.pop("_flash", None)
    if item:
        getattr(st, item[0])(item[1])


def chart_colors():
    """Chart text/grid colours that stay readable in both Light and Dark mode."""
    if APPEARANCE == "Dark":
        return {"text": "#F2EEF9", "muted": "#B9B2CC", "grid": "#3B354B"}
    return {"text": C["ink"], "muted": C["muted"], "grid": C["line"]}


def style_fig(fig, height=260):
    cc = chart_colors()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=cc["text"], size=12),
        legend=dict(font=dict(color=cc["text"])),
    )
    fig.update_xaxes(gridcolor=cc["grid"], zerolinecolor=cc["grid"], tickfont=dict(color=cc["muted"]))
    fig.update_yaxes(gridcolor=cc["grid"], zerolinecolor=cc["grid"], tickfont=dict(color=cc["muted"]))
    return fig


def focus_label(feat, top_factors):
    """The main thing to work on: the first top factor that misses its target."""
    for key, _ in top_factors:
        if not meets_benchmark(key, float(feat.iloc[0][key])):
            return factor_label(key)
    return "All top factors on track"


def refresh_scores(rows):
    """
    Keep saved scores in step with the loaded model (for example after
    retraining). Updates the rows and the database when they differ.
    """
    for row in rows:
        feat, score, top = calculate_assessment(row)
        focus = focus_label(feat, top)
        if abs(float(row["score"]) - score) > 0.05 or row.get("top_factor") != focus:
            user_data.save_checkin(row["username"], row["month"], row, score, focus)
            row["score"], row["top_factor"] = score, focus
    return rows


def checkin_problems(data):
    """(errors, warnings) for a check-in. Errors block saving."""
    errors, warnings = [], []
    income = float(data["income"])
    repay = float(data["monthly_debt_repayment"])
    debt = float(data["total_debt"])
    if income <= 0:
        errors.append("Monthly income must be more than ₹0. The score is built from ratios of your income.")
    elif repay > income * 5:
        errors.append("Monthly debt repayments look unusually high compared with income. Please check the amount.")
    if float(data["essential_expenses"]) <= 0:
        warnings.append("Essential expenses are ₹0. Rent, food and bills are rarely zero, so please double-check.")
    if repay > 0 and debt <= 0:
        warnings.append("You entered loan repayments but no remaining debt. Check the debt amount.")
    if debt > 0 and repay > debt:
        warnings.append("Your monthly repayment is bigger than your total debt. Check both numbers.")
    return errors, warnings


# ---- Login protection (works with the existing auth.py) -------------
MAX_LOGIN_FAILURES = 5
LOCKOUT_SECONDS = 300
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")


@st.cache_resource
def _login_failures():
    """identifier -> {"count": int, "until": datetime or None}. Shared by all sessions."""
    return {}


def login_blocked(identifier):
    """Returns minutes left if this identifier is locked, else 0."""
    rec = _login_failures().get(identifier.strip().lower())
    if rec and rec.get("until") and datetime.now() < rec["until"]:
        return max(1, int((rec["until"] - datetime.now()).total_seconds() // 60) + 1)
    return 0


def login_failed(identifier):
    key = identifier.strip().lower()
    rec = _login_failures().setdefault(key, {"count": 0, "until": None})
    rec["count"] += 1
    if rec["count"] >= MAX_LOGIN_FAILURES:
        rec["count"] = 0
        rec["until"] = datetime.fromtimestamp(datetime.now().timestamp() + LOCKOUT_SECONDS)


def login_succeeded(identifier):
    _login_failures().pop(identifier.strip().lower(), None)


def delete_account(username):
    """Remove the login, all check-ins and all goals for this user."""
    user_data.delete_user_data(username)
    if hasattr(auth, "delete_user"):
        auth.delete_user(username)
    else:  # older auth.py
        users = auth._load()
        users.pop(username.strip().lower(), None)
        auth._save(users)


def topbar():
    st.markdown(
        f"""
        <div class="ledger-topbar">
            <div class="brand">Wellness Ledger</div>
            <div style="color:{C['muted']};font-size:.82rem;">Signed in as {html.escape(str(st.session_state.get('username') or ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    spacer_col, appearance_col, logout_col = st.columns([8, 1, 1.25], gap="small")
    with appearance_col:
        current_appearance = st.session_state.get("ui_appearance", "Light")
        theme_label = "☾" if current_appearance == "Light" else "☀"
        if st.button(
            theme_label,
            key="toggle_appearance_icon",
            help=f"Switch to {'dark' if current_appearance == 'Light' else 'light'} mode",
            use_container_width=True,
        ):
            st.session_state["ui_appearance"] = "Dark" if current_appearance == "Light" else "Light"
            st.rerun()
    with logout_col:
        if st.button("Log out", use_container_width=True):
            st.session_state.username = None
            st.session_state.view = "dashboard"
            st.session_state.result = None
            st.rerun()


def nav_bar():
    st.markdown('<div class="ledger-nav-space"></div>', unsafe_allow_html=True)

    labels = [
        ("Dashboard", "dashboard"),
        ("Monthly Check-in", "checkin"),
        ("Goals", "goals"),
        ("History", "history"),
        ("Guide", "guide"),
    ]

    cols = st.columns(5)
    for col, (label, view) in zip(cols, labels):
        with col:
            if st.button(
                label,
                key=f"nav_{view}",
                use_container_width=True,
                type="primary" if st.session_state.view == view else "secondary",
            ):
                st.session_state.view = view
                st.session_state.result = None
                st.session_state.pop("checkin_result_month", None)
                st.session_state.pop("checkin_saved_notice", None)
                st.session_state.pop("checkin_notes", None)
                st.rerun()


def calculate_assessment(data):
    income_value = float(data["income"])
    if income_value <= 0:
        raise ValueError("Monthly income must be greater than ₹0 before calculating a score.")
    raw = pd.DataFrame([{
        "income": income_value,
        "essential_expenses": data["essential_expenses"],
        "discretionary_expenses": data["discretionary_expenses"],
        "savings_balance": data["savings_balance"],
        "total_debt": data["total_debt"],
        "monthly_debt_repayment": data["monthly_debt_repayment"],
        "emergency_fund": data["emergency_fund"],
        "income_sources": data["income_sources"],
        "age": data["age"],
        "dependents": data["dependents"],
    }])

    feat = engineer_features(raw)
    X = feat[feature_columns]
    score = float(model.predict(X)[0])
    if float(feat.iloc[0]["expense_to_income"]) > 1.0:
        score = min(score, DEFICIT_SCORE_CAP)
    score = max(0.0, min(100.0, score))

    shap_values = explainer.shap_values(X)[0]
    contributions = dict(zip(feature_columns, shap_values))
    sorted_factors = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_factors = [
        f for f in sorted_factors
        if f[0] in BENCHMARKS or f[0] == "income_stability"
    ][:3]

    return feat, score, top_factors


def render_score_result(result):
    r = result
    feat = r["feat"]
    score = r["score"]
    band_label, _ = score_band(score)
    top_factors = r["top_factors"]

    if float(feat.iloc[0]["expense_to_income"]) > 1.0:
        st.error(
            f"Spending is above income this month, so the score is capped at {DEFICIT_SCORE_CAP:.0f} "
            "(\"At risk\") until spending is below income."
        )

    st.markdown(
        '<div class="ledger-section">Your score</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<span class="ledger-score-number">{score:.0f}</span>'
        f'<span class="ledger-score-band">/100 &middot; {band_label}</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ledger-section">What\'s driving it</div>',
        unsafe_allow_html=True,
    )

    for key, shap_val in top_factors:
        user_value = float(feat.iloc[0][key])
        on_track = meets_benchmark(key, user_value)
        dot_color = GOOD if on_track else RISK

        if key == "income_stability":
            display_value = "2+ sources" if user_value >= 1 else "1 source"
        else:
            display_value = format_value(key, user_value)
        label = factor_label(key)
        advice_text = FEATURE_ADVICE[key]["high" if on_track else "low"].format(
            ideal=BENCHMARKS.get(key, {}).get("ideal", 0)
        )

        st.markdown(
            f"""
            <div class="ledger-factor-row">
                <div>
                    <span class="dot" style="background-color:{dot_color};"></span>
                    <span class="ledger-factor-label">{label}</span>
                </div>
                <div>
                    <span class="ledger-factor-value">{display_value}</span>
                    <span class="ledger-factor-points">{shap_val:+.1f} pts</span>
                </div>
            </div>
            <div class="ledger-factor-advice">{advice_text}</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="ledger-section">How close you are to each target</div>',
        unsafe_allow_html=True,
    )

    labels, progress, colours = [], [], []
    for k, bench in BENCHMARKS.items():
        v = float(feat.iloc[0][k])
        if bench["good_direction"] == "high":
            ratio = v / bench["ideal"] if bench["ideal"] else 1.5
        else:
            ratio = bench["ideal"] / v if v > 1e-9 else 1.5
        ratio = max(0.0, min(ratio, 1.5))
        labels.append(bench["label"])
        progress.append(ratio)
        colours.append(GOOD if ratio >= 1.0 else RISK)

    fig = go.Figure(go.Bar(
        y=labels, x=progress, orientation="h", marker_color=colours,
        hovertemplate="%{y}: %{x:.0%} of target<extra></extra>",
    ))
    fig.add_vline(x=1.0, line_dash="dash", line_color=chart_colors()["text"],
                  annotation_text="Target", annotation_position="top")
    style_fig(fig, 320)
    fig.update_xaxes(range=[0, 1.5], tickformat=".0%", showgrid=True, zeroline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Green bars reach the target (100%). Red bars fall short of it.")


def checkin_form(existing=None, target_month=None):
    existing = existing or {}
    target_month = target_month or user_data.current_month()
    target_month_label = pd.to_datetime(target_month + '-01').strftime('%B %Y')

    st.markdown(
        f'<div class="ledger-section">Monthly check-in · '
        f'{target_month_label}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="checkin-group-title"><span>Money in</span><small>Income and incoming cash</small></div>', unsafe_allow_html=True)
    income_col, essential_col = st.columns(2)

    with income_col:
        income = st.number_input(
            "Monthly take-home income (₹)",
            min_value=0,
            value=int(existing.get("income", 40000)),
            step=1000,
            key=f"checkin_income_{target_month}",
            help="Enter the money you usually receive in one month after deductions. Include salary and other regular income.",
        )

    with essential_col:
        essential_expenses = st.number_input(
            "Essential monthly expenses (₹)",
            min_value=0,
            value=int(existing.get("essential_expenses", 18000)),
            step=500,
            key=f"checkin_essential_{target_month}",
            help="Costs you need to pay, such as rent, groceries, electricity, travel to work or college, and medicines.",
        )

    st.markdown('<div class="checkin-group-title"><span>Money out</span><small>What you spend each month</small></div>', unsafe_allow_html=True)
    discretionary_col, savings_col = st.columns(2)

    with discretionary_col:
        discretionary_expenses = st.number_input(
            "Non-essential monthly spending (₹)",
            min_value=0,
            value=int(existing.get("discretionary_expenses", 6000)),
            step=500,
            key=f"checkin_discretionary_{target_month}",
            help="Money spent on things you could reduce if needed, such as eating out, entertainment, shopping or subscriptions.",
        )

    with savings_col:
        savings_balance = st.number_input(
            "Total savings and investments (₹)",
            min_value=0,
            value=int(existing.get("savings_balance", 100000)),
            step=5000,
            key=f"checkin_savings_{target_month}",
            help="Enter the total amount you have already saved or invested. This is your current balance, not how much you save each month.",
        )

    st.markdown('<div class="checkin-group-title"><span>Financial safety</span><small>Savings, debt and your emergency cushion</small></div>', unsafe_allow_html=True)
    debt_col, repayment_col, emergency_col = st.columns(3)

    with debt_col:
        total_debt = st.number_input(
            "Total debt still owed (₹)",
            min_value=0,
            value=int(existing.get("total_debt", 50000)),
            step=5000,
            key=f"checkin_debt_{target_month}",
            help="Add up the remaining balances on loans, credit cards and other debts. Enter 0 if you owe nothing.",
        )

    with repayment_col:
        monthly_debt_repayment = st.number_input(
            "Total monthly loan repayments (₹)",
            min_value=0,
            value=int(existing.get("monthly_debt_repayment", 4000)),
            step=500,
            key=f"checkin_repay_{target_month}",
            help="The total amount you must pay towards loans and debt each month, including EMIs.",
        )

    with emergency_col:
        emergency_fund = st.number_input(
            "Money set aside for emergencies (₹)",
            min_value=0,
            value=int(existing.get("emergency_fund", 30000)),
            step=5000,
            key=f"checkin_emergency_{target_month}",
            help="Money kept available for unexpected costs or loss of income. Do not count money you cannot access quickly.",
        )

    st.markdown('<div class="checkin-group-title"><span>Profile</span><small>Optional details saved with your check-in; not used to calculate the score</small></div>', unsafe_allow_html=True)
    age_col, sources_col, dependents_col = st.columns(3)

    with age_col:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=75,
            value=int(existing.get("age", 24)),
            step=1,
            key=f"checkin_age_{target_month}",
        )

    with sources_col:
        source_default = int(existing.get("income_sources", 1))
        source_default = min(max(source_default, 1), 3)
        income_sources = st.selectbox(
            "Income sources",
            options=[1, 2, 3],
            index=source_default - 1,
            key=f"checkin_sources_{target_month}",
        )

    with dependents_col:
        dep_default = int(existing.get("dependents", 0))
        dep_default = min(max(dep_default, 0), 4)
        dependents = st.selectbox(
            "Dependents",
            options=[0, 1, 2, 3, 4],
            index=dep_default,
            key=f"checkin_dependents_{target_month}",
        )

    form_data = {
        "income": income, "essential_expenses": essential_expenses,
        "discretionary_expenses": discretionary_expenses, "savings_balance": savings_balance,
        "total_debt": total_debt, "monthly_debt_repayment": monthly_debt_repayment,
        "emergency_fund": emergency_fund, "income_sources": income_sources,
        "age": age, "dependents": dependents,
    }
    form_errors, form_warnings = checkin_problems(form_data)

    if income > 0:
        expenses = essential_expenses + discretionary_expenses
        left = income - expenses
        expense_ratio = expenses / income

        p1, p2 = st.columns(2)
        with p1:
            if left < 0:
                st.error(f"Cash deficit this month: {inr(abs(left))}")
            else:
                st.metric("Cash left after spending", f"{inr(left)}")
        with p2:
            st.metric("Expense share", f"{expense_ratio:.0%}")

    for msg in form_errors:
        st.error(msg)
    for msg in form_warnings:
        st.warning(msg)

    st.markdown(
        '<div class="ledger-button-space"></div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Calculate & save my wellness score →",
        use_container_width=True,
        type="primary",
        key="save_checkin",
        disabled=bool(form_errors),
    ):
        data = {
            "income": income,
            "essential_expenses": essential_expenses,
            "discretionary_expenses": discretionary_expenses,
            "savings_balance": savings_balance,
            "total_debt": total_debt,
            "monthly_debt_repayment": monthly_debt_repayment,
            "emergency_fund": emergency_fund,
            "income_sources": income_sources,
            "age": age,
            "dependents": dependents,
        }

        if income <= 0:
            st.error("Monthly income must be greater than ₹0 to calculate a score.")
            st.stop()

        feat, score, top_factors = calculate_assessment(data)
        rule_score = float(compute_wellness_score(feat, add_noise=False)[0])
        total_spend = essential_expenses + discretionary_expenses
        notes = []
        if total_spend > income:
            notes.append(
                f"Your spending ({inr(total_spend)}) is higher than your income ({inr(income)}), so your "
                f"score is capped at {DEFICIT_SCORE_CAP:.0f} until spending is below income."
            )
        if abs(score - rule_score) >= 12:
            notes.append(
                f"The machine-learning estimate ({score:.1f}) differs from the rule-based score "
                f"({rule_score:.1f}) by {abs(score - rule_score):.1f} points. Treat the estimate cautiously."
            )
        top_factor_label = focus_label(feat, top_factors)

        try:
            user_data.save_checkin(
                st.session_state.username,
                target_month,
                data,
                score,
                top_factor_label,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        st.session_state.checkin_notes = notes

        st.session_state.result = {
            "feat": feat,
            "score": score,
            "top_factors": top_factors,
        }
        st.session_state.view = "checkin"
        st.session_state.checkin_result_month = target_month
        st.session_state.checkin_saved_notice = (
            f"{target_month_label} check-in saved — "
            f"your wellness score is {score:.0f}/100."
        )
        st.rerun()


def render_longitudinal_shap(history):
    """Compare SHAP contributions recalculated from saved monthly check-ins."""
    if len(history) < 2:
        return

    ordered = sorted(history, key=lambda row: row["month"])
    months = [row["month"] for row in ordered]
    st.markdown(
        '<div class="ledger-section">How money factors changed over time</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        earlier_month = st.selectbox(
            "Earlier month", months, index=max(0, len(months) - 2),
            key="longitudinal_shap_earlier",
        )
    with right:
        later_month = st.selectbox(
            "Compare with", months, index=len(months) - 1,
            key="longitudinal_shap_later",
        )

    if earlier_month == later_month:
        st.info("Choose two different months to compare their explanations.")
        return

    by_month = {row["month"]: row for row in ordered}
    feature_labels = {
        SAVINGS_KEY: "Savings cushion",
        "debt_to_income": "Debt-to-income",
        "debt_repayment_burden": "Debt repayment burden",
        "expense_to_income": "Expense-to-income",
        "emergency_fund_months": "Emergency fund months",
        "discretionary_ratio": "Discretionary spending",
        "income_stability": "Income stability",
        "dependents": "Dependents",
        "age": "Age",
    }

    def shap_for_checkin(row):
        raw = pd.DataFrame([{
            "income": max(float(row["income"]), 1.0),
            "essential_expenses": float(row["essential_expenses"]),
            "discretionary_expenses": float(row["discretionary_expenses"]),
            "savings_balance": float(row["savings_balance"]),
            "total_debt": float(row["total_debt"]),
            "monthly_debt_repayment": float(row["monthly_debt_repayment"]),
            "emergency_fund": float(row["emergency_fund"]),
            "income_sources": int(row["income_sources"]),
            "age": int(row["age"]),
            "dependents": int(row["dependents"]),
        }])
        features = engineer_features(raw)
        X = features[feature_columns]
        prediction = float(model.predict(X)[0])
        if float(features.iloc[0]["expense_to_income"]) > 1.0:
            prediction = min(prediction, DEFICIT_SCORE_CAP)
        values = explainer.shap_values(X)[0]
        return prediction, dict(zip(feature_columns, values))

    try:
        earlier_score, earlier_values = shap_for_checkin(by_month[earlier_month])
        later_score, later_values = shap_for_checkin(by_month[later_month])
    except (KeyError, TypeError, ValueError) as exc:
        st.warning(f"Couldn't calculate the historical explanations for these months: {exc}")
        return

    score_delta = later_score - earlier_score
    m1, m2, m3 = st.columns(3)
    m1.metric("Earlier model score", f"{earlier_score:.1f}/100")
    m2.metric("Later model score", f"{later_score:.1f}/100", delta=f"{score_delta:+.1f} points")
    all_keys = list(dict.fromkeys([*earlier_values.keys(), *later_values.keys()]))
    changes = []
    for key in all_keys:
        old = float(earlier_values.get(key, 0.0))
        new = float(later_values.get(key, 0.0))
        changes.append({
            "feature": feature_labels.get(key, key.replace("_", " ").title()),
            "earlier": old,
            "later": new,
            "change": new - old,
        })
    changes.sort(key=lambda item: abs(item["change"]), reverse=True)
    if changes:
        largest = changes[0]
        m3.metric("Largest SHAP change", largest["feature"], delta=f"{largest['change']:+.2f} points")
    else:
        m3.metric("Largest SHAP change", "No change")
    plot_data = changes[:6]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=earlier_month,
        y=[item["feature"] for item in plot_data],
        x=[item["earlier"] for item in plot_data],
        orientation="h",
    ))
    fig.add_trace(go.Bar(
        name=later_month,
        y=[item["feature"] for item in plot_data],
        x=[item["later"] for item in plot_data],
        orientation="h",
    ))
    style_fig(fig, max(300, 48 * len(plot_data) + 95))
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title="Contribution to estimated score (points)", zeroline=True)
    fig.update_yaxes(title="", showgrid=False, autorange="reversed")
    fig.data[0].marker.color = chart_colors()["muted"]
    fig.data[1].marker.color = BRASS
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("**The six factors whose contributions changed most**")
    delta_df = pd.DataFrame([{
        "Financial factor": item["feature"],
        f"{earlier_month}": round(item["earlier"], 2),
        f"{later_month}": round(item["later"], 2),
        "Change": round(item["change"], 2),
    } for item in plot_data])
    st.dataframe(delta_df, use_container_width=True, hide_index=True)


def render_dashboard():
    month = user_data.current_month()
    checkin = user_data.get_checkin(st.session_state.username, month)
    goals = user_data.get_goals(st.session_state.username, include_completed=True)

    st.markdown(
        '<div class="page-heading"><h1>Welcome back</h1><p>A simple snapshot of the money details you entered and what they may mean.</p></div>',
        unsafe_allow_html=True,
    )

    if not checkin:
        st.markdown(
            f'<div class="ledger-dashboard-empty"><div class="dashboard-kicker">{user_data.current_month_label()}</div>'
            '<div class="dashboard-empty-title">Your month hasn’t been recorded yet.</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Start this month’s check-in →", use_container_width=True, key="dashboard_start_checkin", type="primary"):
            st.session_state.view = "checkin"
            st.rerun()
        return

    refresh_scores([checkin])
    score = checkin["score"]
    band, _ = score_band(score)
    expense_share = ((checkin["essential_expenses"] + checkin["discretionary_expenses"]) / checkin["income"] * 100) if checkin["income"] else 0

    explanation_data = {
        "income": checkin["income"],
        "essential_expenses": checkin["essential_expenses"],
        "discretionary_expenses": checkin["discretionary_expenses"],
        "savings_balance": checkin["savings_balance"],
        "total_debt": checkin["total_debt"],
        "monthly_debt_repayment": checkin["monthly_debt_repayment"],
        "emergency_fund": checkin["emergency_fund"],
        "income_sources": checkin["income_sources"],
        "age": checkin["age"],
        "dependents": checkin["dependents"],
    }
    dashboard_feat, _, dashboard_factors = calculate_assessment(explanation_data)

    left, right = st.columns([1.12, .88])
    with left:
        st.markdown(
            f"""<div class=\"ledger-dashboard-score\">
                <div class=\"dashboard-kicker\">{user_data.current_month_label()} check-in</div>
                <div class=\"score-layout\">
                  <div><span class=\"score-big\">{score:.0f}</span><span class=\"score-band\"> / 100 · {band}</span>
                  <div class=\"score-caption\">Your current financial wellness score</div></div>
                  <div style=\"text-align:right;color:{GOOD};font-size:.78rem;font-weight:600;\">{html.escape(str(checkin.get('top_factor') or 'Financial snapshot'))}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    with right:
        history = refresh_scores(user_data.get_checkins(st.session_state.username, limit=6))
        if len(history) >= 2:
            hdf = pd.DataFrame(history).sort_values("month")
            fig = go.Figure(go.Scatter(x=hdf["month"], y=hdf["score"], mode="lines+markers",
                                       line=dict(color=BRASS, width=2), marker=dict(size=6, color=BRASS)))
            style_fig(fig, 150)
            fig.update_layout(margin=dict(l=4, r=4, t=10, b=4), font=dict(size=9))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(range=[0, 100], showgrid=True)
            st.markdown('<div class="score-trend-card"><div class="dashboard-kicker">Score trend</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="score-trend-card"><div class="dashboard-kicker">Score trend</div><div style="font-family:DM Serif Display,serif;font-size:1.05rem;margin-top:.4rem;">Your next check-in will start the trend.</div><div class="score-caption">Keep checking in monthly to see the direction.</div></div>', unsafe_allow_html=True)

    if float(dashboard_feat.iloc[0]["expense_to_income"]) > 1.0:
        st.error(
            f"Your spending is above your income this month, so your score is capped at {DEFICIT_SCORE_CAP:.0f} "
            "(\"At risk\"). Bringing spending below income is the first priority."
        )

    m1,m2,m3 = st.columns(3)
    with m1: st.metric("Monthly income", f"{inr(checkin['income'])}")
    with m2: st.metric("Monthly spending", f"{inr(checkin['essential_expenses'] + checkin['discretionary_expenses'])}")
    with m3: st.metric("Savings / investments", f"{inr(checkin['savings_balance'])}")

    st.markdown('<div class="ledger-section">What’s affecting your score?</div>', unsafe_allow_html=True)
    factors = []
    for key, shap_val in dashboard_factors:
        user_value = float(dashboard_feat.iloc[0][key])
        if key == "income_stability":
            label = "Income stability"
            status = "On track" if user_value >= 1 else "Needs attention"
            copy = (
                "You have more than one income source, which adds a cushion."
                if user_value >= 1
                else "You currently rely on one income source."
            )
        else:
            bench = BENCHMARKS[key]
            label = bench["label"]
            is_good = (
                user_value >= bench["ideal"]
                if bench["good_direction"] == "high"
                else user_value <= bench["ideal"]
            )
            status = "On track" if is_good else "Needs attention"
            copy = (
                f"{format_value(key, user_value)} — within the guidance benchmark."
                if is_good
                else f"{format_value(key, user_value)} — outside the guidance benchmark."
            )
        factors.append((label, status, copy))
    cols = st.columns(3)
    for col,(title,status,copy) in zip(cols,factors[:3]):
        with col:
            st.markdown(f'<div class="factor-card"><div class="factor-title">{title}</div><div class="factor-status">{status}</div><div class="factor-copy">{copy}</div></div>', unsafe_allow_html=True)

    recommendations = generate_recommendations(
        dashboard_feat,
        dashboard_factors,
        max_items=3,
    )

    st.markdown(
        '<div class="ledger-section">Personalized next steps</div>',
        unsafe_allow_html=True,
    )

    for rec in recommendations:
        st.markdown(
            f"""
            <div class="factor-card" style="margin-bottom:.45rem;">
                <div class="factor-title">{rec["label"]}</div>
                <div class="factor-copy">{rec["text"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_longitudinal_shap(refresh_scores(user_data.get_checkins(st.session_state.username, limit=12)))

    render_financial_twin(
        checkin,
        dashboard_feat,
        dashboard_factors,
        score,
    )

    st.markdown('<div class="ledger-section">Your goals</div>', unsafe_allow_html=True)
    active_goals = [g for g in goals if not g["completed"]]
    if active_goals:
        cols = st.columns(min(3, len(active_goals[:3])))
        for col, goal in zip(cols, active_goals[:3]):
            pct = min(100, (goal["current_amount"] / goal["target_amount"]) * 100)
            with col:
                st.markdown(f"""<div class=\"ledger-goal-card\"><div class=\"goal-name\">{html.escape(str(goal['name']))}</div><div class=\"goal-meta\">{inr(goal['current_amount'])} / {inr(goal['target_amount'])} · {pct:.0f}%</div><div class=\"ledger-progress-track\"><div class=\"ledger-progress-fill\" style=\"width:{pct:.1f}%;\"></div></div><div class=\"goal-card-bottom\">Target: {goal.get('target_date') or '—'}</div></div>""", unsafe_allow_html=True)

    if st.button("Update this month’s check-in →", use_container_width=True, key="dashboard_update_checkin"):
        st.session_state.view = "checkin"
        st.rerun()


def render_financial_twin(checkin, current_feat, current_factors, current_score):
    """Run a temporary what-if simulation without changing the saved check-in."""
    st.markdown('<div class="ledger-section">Try a what-if change</div>', unsafe_allow_html=True)

    scenario_options = {
        "Increase savings / investments": ("savings_balance", "increase"),
        "Increase emergency fund": ("emergency_fund", "increase"),
        "Reduce total debt": ("total_debt", "decrease"),
        "Reduce monthly debt repayment": ("monthly_debt_repayment", "decrease"),
        "Increase monthly income": ("income", "increase"),
    }

    scenario = st.selectbox(
        "Choose a scenario",
        list(scenario_options.keys()),
        key="twin_scenario",
    )
    field, direction = scenario_options[scenario]
    current_value = float(checkin[field])

    if direction == "increase":
        amount = st.number_input(
            "Change amount (₹)",
            min_value=0,
            value=5000,
            step=1000,
            key="twin_amount_increase",
        )
        simulated_value = current_value + amount
    else:
        max_reduction = max(0.0, current_value)
        amount = st.number_input(
            "How much would you reduce it by? (₹)",
            min_value=0.0,
            max_value=float(max_reduction),
            value=min(5000.0, max_reduction),
            step=1000.0,
            key="twin_amount_decrease",
            help="Enter the amount to test. You cannot reduce it by more than the current amount shown.",
        )
        simulated_value = max(0.0, current_value - amount)

    if field == "income":
        current_label = f"{inr(current_value)}"
        simulated_label = f"{inr(simulated_value)}"
    else:
        current_label = f"{inr(current_value)}"
        simulated_label = f"{inr(simulated_value)}"

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Current", current_label)
    with c2:
        st.metric("Simulated", simulated_label)

    simulated_data = {
        "income": checkin["income"],
        "essential_expenses": checkin["essential_expenses"],
        "discretionary_expenses": checkin["discretionary_expenses"],
        "savings_balance": checkin["savings_balance"],
        "total_debt": checkin["total_debt"],
        "monthly_debt_repayment": checkin["monthly_debt_repayment"],
        "emergency_fund": checkin["emergency_fund"],
        "income_sources": checkin["income_sources"],
        "age": checkin["age"],
        "dependents": checkin["dependents"],
    }
    simulated_data[field] = simulated_value

    if st.button("Run simulation →", use_container_width=True, key="run_financial_twin", type="primary"):
        sim_feat, sim_score, sim_factors = calculate_assessment(simulated_data)
        delta = sim_score - float(current_score)

        st.markdown(
            f'<div class="factor-card" style="margin-top:.55rem;">'
            f'<div class="factor-title">Simulated wellness score</div>'
            f'<div class="score-layout"><div><span class="score-big">{sim_score:.0f}</span>'
            f'<span class="score-band"> / 100 · {score_band(sim_score)[0]}</span></div>'
            f'<div style="text-align:right;font-weight:600;">'
            f'{"+" if delta >= 0 else ""}{delta:.1f} points vs current</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        current_map = dict(current_factors)
        simulated_map = dict(sim_factors)
        changed_keys = set(current_map) | set(simulated_map)
        changes = []
        for key in changed_keys:
            old = float(current_map.get(key, 0.0))
            new = float(simulated_map.get(key, 0.0))
            changes.append((key, new - old, old, new))
        changes.sort(key=lambda x: abs(x[1]), reverse=True)

        st.markdown('<div class="history-list-title">What changed in the model explanation?</div>', unsafe_allow_html=True)
        shown = 0
        for key, shap_delta, old, new in changes:
            if key not in BENCHMARKS and key != "income_stability":
                continue
            label = "Income stability" if key == "income_stability" else BENCHMARKS[key]["label"]
            direction_text = "more supportive" if shap_delta > 0 else "less supportive" if shap_delta < 0 else "unchanged"
            st.markdown(
                f'<div style="font-size:.82rem;margin:.35rem 0;"><b>{label}</b> became {direction_text} of the prediction '
                f'({shap_delta:+.1f} SHAP change).</div>',
                unsafe_allow_html=True,
            )
            shown += 1
            if shown >= 3:
                break


def render_goals():
    st.markdown(
        '<div class="ledger-section">Goals</div>',
        unsafe_allow_html=True,
    )

    goals = user_data.get_goals(
        st.session_state.username,
        include_completed=True,
    )

    with st.form("new_goal_form"):
        name = st.text_input(
            "Goal name",
            placeholder="Laptop, emergency fund, trip...",
            max_chars=50,
        )

        goal_col, category_col = st.columns(2)

        with goal_col:
            target_amount = st.number_input(
                "Target amount (₹)",
                min_value=1,
                value=10000,
                step=1000,
            )

        with category_col:
            category = st.selectbox(
                "Category",
                ["Emergency fund", "Education", "Travel", "Purchase", "Other"],
            )

        current_col, date_col = st.columns(2)

        with current_col:
            current_amount = st.number_input(
                "Already saved (₹)",
                min_value=0,
                value=0,
                step=500,
            )

        with date_col:
            target_date = st.date_input(
                "Target date",
                value=None,
            )

        create = st.form_submit_button(
            "Add goal",
            use_container_width=True,
        )

    if create:
        if not name.strip():
            st.error("Give the goal a name first.")
        elif target_amount <= 0:
            st.error("Target amount must be greater than ₹0.")
        elif len(name.strip()) > 50:
            st.error("Goal names must be 50 characters or fewer.")
        else:
            try:
                user_data.create_goal(
                    st.session_state.username,
                    name,
                    category,
                    target_amount,
                    current_amount,
                    target_date.isoformat() if target_date else None,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                flash("Goal added.")
                st.rerun()

    if not goals:
        return

    st.markdown(
        '<div class="ledger-section">Your goals</div>',
        unsafe_allow_html=True,
    )

    for goal in goals:
        pct = min(
            100,
            (goal["current_amount"] / goal["target_amount"]) * 100,
        )

        status = "Complete" if goal["completed"] else f"{pct:.0f}%"

        pace_text = ""
        if goal.get("target_date") and not goal["completed"]:
            try:
                days_left = (date.fromisoformat(goal["target_date"]) - date.today()).days
            except ValueError:
                days_left = None
            if days_left is not None:
                if days_left <= 0:
                    pace_text = "The target date has passed."
                else:
                    still_needed = goal["target_amount"] - goal["current_amount"]
                    per_month = still_needed / max(1, math.ceil(days_left / 30))
                    pace_text = f"Save about {inr(per_month)} per month to reach it by {goal['target_date']}."

        goal_html = (
            f'<div class="ledger-goal-card">'
            f'<div class="goal-card-top">'
            f'<div>'
            f'<div class="goal-name">{html.escape(str(goal["name"]))}</div>'
            f'<div class="goal-category">{html.escape(str(goal["category"]))}</div>'
            f'</div>'
            f'<div class="goal-status">{status}</div>'
            f'</div>'
            f'<div class="ledger-progress-track">'
            f'<div class="ledger-progress-fill" style="width:{pct:.1f}%;"></div>'
            f'</div>'
            f'<div class="goal-card-bottom">'
            f'<span>{inr(goal["current_amount"])} saved</span>'
            f'<span>{inr(goal["target_amount"])} target</span>'
            f'</div>'
            f'<div style="font-size:.72rem;opacity:.75;margin-top:.25rem;">{html.escape(pace_text)}</div>'
            f'</div>'
        )

        st.markdown(goal_html, unsafe_allow_html=True)

        if not goal["completed"]:
            progress_col, action_col = st.columns([4, 1])

            with progress_col:
                amount_to_add = st.number_input(
                    "Add amount (₹)",
                    min_value=0,
                    value=0,
                    step=500,
                    key=f"goal_add_{goal['id']}",
                    label_visibility="collapsed",
                )

            with action_col:
                if st.button(
                    "Add",
                    key=f"goal_save_{goal['id']}",
                    use_container_width=True,
                ):
                    updated_amount = goal["current_amount"] + amount_to_add

                    user_data.update_goal_progress(
                        st.session_state.username,
                        goal["id"],
                        updated_amount,
                    )
                    flash("Progress updated.")
                    st.rerun()

        with st.expander("Correct the saved amount"):
            fixed_amount = st.number_input(
                "Total saved so far (₹)",
                min_value=0.0,
                value=float(goal["current_amount"]),
                step=500.0,
                key=f"goal_fix_{goal['id']}",
                help="Use this to fix a mistake. It replaces the saved amount (and reopens the goal if needed).",
            )
            if st.button("Save correction", key=f"goal_fix_btn_{goal['id']}", use_container_width=True):
                user_data.update_goal_progress(st.session_state.username, goal["id"], fixed_amount)
                flash("Saved amount corrected.")
                st.rerun()

        if st.button(
            "Delete",
            key=f"goal_delete_{goal['id']}",
        ):
            user_data.delete_goal(
                st.session_state.username,
                goal["id"],
            )
            st.rerun()


def render_history():
    st.markdown(
        '<div class="page-heading"><h1>Your history</h1><p>See how your financial wellness has changed month by month.</p></div>',
        unsafe_allow_html=True,
    )

    checkins = refresh_scores(user_data.get_checkins(st.session_state.username, limit=12))

    if not checkins:
        st.markdown(
            '<div class="empty-card"><strong>No history yet.</strong><br>Complete your first monthly check-in and your progress will appear here.</div>',
            unsafe_allow_html=True,
        )
        render_data_controls()
        return

    history_df = pd.DataFrame(checkins).sort_values("month")

    st.markdown('<div class="history-chart-title">Wellness score</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["month"],
            y=history_df["score"],
            mode="lines+markers",
            line=dict(color=BRASS, width=3),
            marker=dict(size=8, color=BRASS),
            hovertemplate="%{x}<br>Score: %{y:.0f}<extra></extra>",
        )
    )
    style_fig(fig, 280)
    fig.update_yaxes(range=[0, 100], showgrid=True, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.add_hrect(y0=75, y1=100, fillcolor=GOOD, opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=50, fillcolor=RISK, opacity=0.07, line_width=0)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Green zone: 75 and above (Strong). Red zone: below 50 (At risk).")

    st.markdown('<div class="history-list-title">Monthly snapshots</div>', unsafe_allow_html=True)

    for row in checkins:
        band, _ = score_band(row["score"])
        cash_left = row["income"] - row["essential_expenses"] - row["discretionary_expenses"]
        st.markdown(
            f"""
            <div class="history-row">
                <div>
                    <div class="history-month">{datetime.strptime(row["month"] + "-01", "%Y-%m-%d").strftime("%B %Y")}</div>
                    <div class="history-focus">Focus: {html.escape(str(row.get("top_factor") or "—"))}</div>
                </div>
                <div class="history-numbers">
                    <span><b>{row["score"]:.0f}</b> / 100 · {band}</span>
                    <small>{'Cash deficit' if cash_left < 0 else 'Cash left'} {inr(abs(cash_left) if cash_left < 0 else cash_left)}</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_data_controls()


def render_data_controls():
    """Download or delete your own data."""
    rows = user_data.get_checkins(st.session_state.username, limit=120)
    with st.expander("Your data"):
        if rows:
            export = pd.DataFrame(rows).drop(columns=["id", "username"], errors="ignore")
            st.download_button(
                "Download my check-ins (CSV)",
                export.to_csv(index=False).encode("utf-8"),
                file_name="wellness_ledger_checkins.csv",
                mime="text/csv",
            )
        st.markdown("**Delete my account**")
        st.caption("Removes your login, all check-ins and all goals. This cannot be undone.")
        sure = st.checkbox("I understand this cannot be undone", key="delete_sure")
        if st.button("Delete my account and data", key="delete_account", disabled=not sure, use_container_width=True):
            delete_account(st.session_state.username)
            st.session_state.username = None
            st.session_state.view = "dashboard"
            st.session_state.result = None
            flash("Your account and data were deleted.")
            st.rerun()


def render_guide():
    st.markdown(
        '<div class="page-heading"><h1>How to Use & What You’ll Get</h1>'
        '<p>Everything you need to get started, and what Wellness Ledger helps you understand.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### How to use Wellness Ledger")
    st.markdown(
        "1. Open **Monthly Check-in** and enter your monthly income, essential and discretionary expenses, "
        "savings balance, total debt, monthly debt repayments, emergency fund, income sources, age and dependents.\n"
        "2. Choose the month from the calendar and select **Calculate & save my wellness score**. Your check-in is "
        "saved under that month; saving again updates that month’s figures.\n"
        "3. Visit **Dashboard** for your latest snapshot, **History** to compare saved months, and **Goals** to set "
        "and monitor savings targets.\n"
        "4. Use the scenario comparison to explore how a possible financial change may affect your estimate."
    )

    st.markdown("### What you’ll get")
    st.markdown(
        "- **Wellness score out of 100** with a simple status band.\n"
        "- **Category-wise indicators** for savings, debt, spending and emergency savings.\n"
        "- **Income and expense summary**, including estimated cash left and expense share.\n"
        "- **Key factors and personalised suggestions** to highlight areas worth reviewing.\n"
        "- **Scenario comparisons** to test possible changes without overwriting your saved check-in.\n"
        "- **Monthly history and savings goals** to help track progress over time."
    )

    st.markdown("### About your score")
    st.markdown(
        "- **How it is made.** A machine-learning model estimates a 0-100 score from the numbers you enter. "
        "It was trained on **synthetic (computer-generated) examples** scored by common rules of thumb "
        "(for example 6 months of savings and loan payments under 30% of income). It has not been tested on real people.\n"
        "- **What “On track” means.** Your number is on the healthy side of a rule-of-thumb target. "
        "It is a guide, not a judgement about you.\n"
        "- **Spending above income.** If you spend more than you earn in a month, the score is capped at "
        f"{DEFICIT_SCORE_CAP:.0f} (“At risk”).\n"
        "- **Why a factor is highlighted.** The app uses SHAP to show which factors moved the estimate most. "
        "This describes the model, not cause and effect in your life.\n"
        "- **Not financial advice.** Talk to a qualified adviser before big money decisions.\n"
        "- **Your data.** Your entries stay in a local database on this device. You can download or delete them "
        "at the bottom of the History page."
    )


# ===================================================================
# AUTH GATE
# ===================================================================
if "username" not in st.session_state:
    st.session_state.username = None

if "view" not in st.session_state:
    st.session_state.view = "dashboard"

if "result" not in st.session_state:
    st.session_state.result = None

if "recovery_step" not in st.session_state:
    st.session_state.recovery_step = None

if "recovery_email" not in st.session_state:
    st.session_state.recovery_email = ""

if "recovery_token" not in st.session_state:
    st.session_state.recovery_token = None


if st.session_state.username is None:
    st.markdown(
        """
        <div class="ledger-cover">
            <div class="cover-mark"></div>
            <div class="title">Wellness Ledger</div>
            <div class="rule"></div>
            <div class="sub">
                A clearer picture of your money, month by month.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "auth_choice" not in st.session_state:
        st.session_state.auth_choice = "Log in"

    if st.session_state.pop("password_reset_success", False):
        st.success("Password reset successfully. You can now log in.")
    show_flash()

    auth_login_col, auth_signup_col = st.columns(2)
    with auth_login_col:
        if st.button(
            "Log in",
            use_container_width=True,
            key="auth_login_tab",
            type="primary" if st.session_state.auth_choice == "Log in" else "secondary",
        ):
            st.session_state.auth_choice = "Log in"
            st.rerun()

    with auth_signup_col:
        if st.button(
            "Create account",
            use_container_width=True,
            key="auth_signup_tab",
            type="primary" if st.session_state.auth_choice == "Create account" else "secondary",
        ):
            st.session_state.auth_choice = "Create account"
            st.rerun()

    auth_choice = st.session_state.auth_choice

    if auth_choice == "Log in" and st.session_state.get("recovery_step") is None:
        identifier = st.text_input(
            "Username or Gmail address",
            key="login_id",
        )
        password = st.text_input(
            "Password",
            type="password",
            key="login_pw",
        )

        if st.button(
            "Log in",
            use_container_width=True,
            key="login_btn",
        ):
            wait = login_blocked(identifier)
            if wait:
                st.error(f"Too many wrong attempts. Try again in about {wait} minute(s).")
            else:
                ok, username = auth.login(identifier, password)
                if ok:
                    login_succeeded(identifier)
                    st.session_state.username = username
                    st.session_state.view = "dashboard"
                    st.rerun()
                else:
                    login_failed(identifier)
                    st.error("Incorrect username/Gmail or password.")

        if st.button(
            "Forgot username or password?",
            use_container_width=True,
            key="forgot_account_btn",
        ):
            st.session_state.recovery_step = "email"
            st.session_state.recovery_email = ""
            st.session_state.recovery_token = None
            st.session_state.pop("recovered_username_result", None)
            st.session_state.pop("recovered_username_error", None)
            st.rerun()

    elif auth_choice == "Create account" and st.session_state.get("recovery_step") is None:
        nu = st.text_input("Choose a username", key="signup_user")
        ne = st.text_input(
            "Gmail address",
            key="signup_email",
            placeholder="you@gmail.com",
        )
        np_ = st.text_input(
            "Choose a password",
            type="password",
            key="signup_pw",
        )

        if st.button(
            "Create account",
            use_container_width=True,
            key="signup_btn",
        ):
            if not USERNAME_RE.match(nu.strip().lower()):
                st.error("Username must be 3-30 characters: letters, numbers, dot, dash or underscore.")
            else:
                ok, msg = auth.create_user(nu, ne, np_)
                if ok:
                    st.session_state.username = nu.strip().lower()
                    st.session_state.view = "dashboard"
                    st.rerun()
                else:
                    st.error(msg)

    recovery_step = st.session_state.get("recovery_step")

    if recovery_step == "email":
        st.subheader("Recover your account")
        st.write("Enter the Gmail address registered with your account.")

        email = st.text_input(
            "Registered Gmail address",
            key="recovery_email_input",
            placeholder="you@gmail.com",
        )

        if st.button("Send OTP", use_container_width=True, key="send_recovery_otp"):
            if not auth.is_valid_gmail(email):
                st.error("Please enter a valid Gmail address.")
            else:
                ok, message = auth.request_recovery_otp(email)
                if ok:
                    st.session_state.recovery_email = email.strip().lower()
                    st.session_state.recovery_step = "otp"
                    flash("If that Gmail is registered, an OTP has been sent.", "info")
                    st.rerun()
                else:
                    st.error(message)

        if st.button("Back to login", use_container_width=True, key="recovery_back_login"):
            st.session_state.recovery_step = None
            st.session_state.recovery_email = ""
            st.session_state.recovery_token = None
            st.rerun()

    elif recovery_step == "otp":
        st.subheader("Verify your email")
        st.write(f"Enter the 6-digit OTP sent to **{st.session_state.recovery_email}**.")

        otp = st.text_input(
            "OTP",
            max_chars=6,
            key="recovery_otp",
            placeholder="123456",
        )

        if st.button("Verify OTP", use_container_width=True, key="verify_recovery_otp"):
            ok, token = auth.verify_recovery_otp(
                st.session_state.recovery_email,
                otp,
            )
            if ok:
                st.session_state.recovery_token = token
                st.session_state.recovery_step = "choice"
                st.rerun()
            else:
                st.error("Invalid or expired OTP.")

        if st.button("Resend OTP", use_container_width=True, key="resend_recovery_otp"):
            ok, message = auth.request_recovery_otp(st.session_state.recovery_email)
            if ok:
                flash("A new OTP has been sent if the account exists.", "info")
                st.rerun()
            else:
                st.error(message)

        if st.button("Back", use_container_width=True, key="recovery_back_email"):
            st.session_state.recovery_step = "email"
            st.session_state.recovery_token = None
            st.rerun()

    elif recovery_step == "choice":
        st.subheader("Account recovery")
        choice = st.radio(
            "What would you like to recover?",
            ["Recover my username", "Reset my password"],
            key="recovery_choice",
        )

        if st.button("Continue", use_container_width=True, key="recovery_continue"):
            st.session_state.recovery_step = (
                "username" if choice == "Recover my username" else "password"
            )
            st.rerun()

    elif recovery_step == "username":
        st.subheader("Your username")

        if "recovered_username_result" not in st.session_state:
            ok, recovered = auth.recover_username(
                st.session_state.recovery_email,
                st.session_state.recovery_token,
            )
            st.session_state.recovered_username_result = recovered if ok else None
            st.session_state.recovered_username_error = None if ok else "The recovery session has expired or was already used."
            st.session_state.recovery_token = None

        if st.session_state.recovered_username_result:
            st.success(f"Your username is: **{st.session_state.recovered_username_result}**")
        else:
            st.error(st.session_state.recovered_username_error or "Unable to recover the username.")

        if st.button("Back to login", use_container_width=True, key="username_recovery_done"):
            st.session_state.recovery_step = None
            st.session_state.pop("recovered_username_result", None)
            st.session_state.pop("recovered_username_error", None)
            st.rerun()

    elif recovery_step == "password":
        st.subheader("Reset your password")

        new_password = st.text_input(
            "New password",
            type="password",
            key="new_recovery_password",
        )
        confirm_password = st.text_input(
            "Confirm new password",
            type="password",
            key="confirm_recovery_password",
        )

        if st.button("Reset password", use_container_width=True, key="reset_recovery_password"):
            if not new_password or not confirm_password:
                st.error("Please enter and confirm your new password.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = auth.reset_password(
                    st.session_state.recovery_email,
                    st.session_state.recovery_token,
                    new_password,
                )
                if ok:
                    st.session_state.recovery_step = None
                    st.session_state.recovery_token = None
                    st.session_state.recovery_email = ""
                    st.session_state.password_reset_success = True
                    st.rerun()
                else:
                    st.error(message)

    st.stop()


# ===================================================================
# LOGGED-IN APP
# ===================================================================
topbar()
nav_bar()
show_flash()

if st.session_state.view == "dashboard":
    render_dashboard()

elif st.session_state.view == "checkin":
    saved_checkins = user_data.get_checkins(st.session_state.username, limit=120)
    selected_date = st.date_input(
        "Choose a date in the month you want to enter or update",
        value=pd.to_datetime(user_data.current_month() + "-01").date(),
        min_value=date(pd.Timestamp.today().year - 2, pd.Timestamp.today().month, 1),
        max_value=pd.Timestamp.today().date(),
        format="DD/MM/YYYY",
        key="checkin_target_date",
        help="Use the calendar to choose any date in the month. The app records the month and year only.",
    )
    selected_month = selected_date.strftime("%Y-%m")
    st.caption(
        "Choose any day in the month using the calendar. Your check-in is saved under that month, "
        "not under the exact day. Saving again replaces that month's figures."
    )
    existing_checkin = user_data.get_checkin(
        st.session_state.username,
        selected_month,
    )
    checkin_form(existing_checkin, selected_month)

    if st.session_state.get("result") and st.session_state.get("checkin_result_month") == selected_month:
        notice = st.session_state.pop("checkin_saved_notice", None)
        if notice:
            st.success(notice)
        for note in st.session_state.pop("checkin_notes", []):
            st.warning(note)
        render_score_result(st.session_state.result)

elif st.session_state.view == "goals":
    render_goals()

elif st.session_state.view == "history":
    render_history()

elif st.session_state.view == "guide":
    render_guide()

else:
    st.session_state.view = "dashboard"
    st.rerun()


if APPEARANCE == "Dark":
    st.markdown("""
    <style>
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stMultiSelect"] div[data-baseweb="select"],
    div[data-testid="stDateInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
      background:#211E2D !important;
      color:#F2EEF9 !important;
      -webkit-text-fill-color:#F2EEF9 !important;
      border-color:#514365 !important;
      caret-color:#F2EEF9 !important;
      opacity:1 !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] * {
      color:#F2EEF9 !important;
      -webkit-text-fill-color:#F2EEF9 !important;
      opacity:1 !important;
    }
    .ledger-dashboard-score *, .ledger-dashboard-empty *, .ledger-dashboard-focus *,
    .ledger-goal-card *, .history-row *, .empty-card *, .factor-card *,
    .score-trend-card *, .ledger-factor-row *, .ledger-factor-advice *,
    .ledger-factor-points, .goal-meta, .goal-category, .goal-card-bottom,
    .history-numbers, .stCaption,
    [data-testid="stCaptionContainer"] * {
      opacity:1 !important;
    }
    .ledger-dashboard-score .score-big, .ledger-dashboard-score .score-number,
    .ledger-dashboard-score .ledger-score-number, .ledger-dashboard-score .score-value,
    .ledger-dashboard-empty .score-big, .ledger-dashboard-focus strong,
    .ledger-dashboard-score strong, .ledger-dashboard-score b,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
      color:#F2EEF9 !important;
      -webkit-text-fill-color:#F2EEF9 !important;
    }
    .ledger-dashboard-score .dashboard-kicker, .ledger-dashboard-empty .dashboard-kicker,
    .ledger-dashboard-focus .dashboard-kicker, .factor-card .factor-copy,
    .factor-card .factor-title, .factor-card .factor-status,
    .score-trend-card .score-caption,
    .goal-meta, .goal-category, .goal-card-bottom, .history-numbers,
    [data-testid="stCaptionContainer"] * {
      color:#C7BEDA !important;
      -webkit-text-fill-color:#C7BEDA !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
      background:#211E2D !important; color:#F2EEF9 !important;
    }
    [role="option"], [role="option"] * {
      background:#211E2D !important; color:#F2EEF9 !important;
      -webkit-text-fill-color:#F2EEF9 !important; opacity:1 !important;
    }
    [role="option"][aria-selected="true"], [role="option"]:hover,
    [role="option"][aria-selected="true"] *, [role="option"]:hover * {
      background:#3A2E50 !important; color:#FFFFFF !important;
      -webkit-text-fill-color:#FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stMultiSelect"] div[data-baseweb="select"],
    div[data-testid="stDateInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
      background:#FFFFFF !important;
      color:#333044 !important;
      -webkit-text-fill-color:#333044 !important;
      border-color:#D8CDEB !important;
      caret-color:#333044 !important;
      opacity:1 !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] * {
      color:#333044 !important;
      -webkit-text-fill-color:#333044 !important;
      opacity:1 !important;
    }
    .ledger-dashboard-score *, .ledger-dashboard-empty *, .ledger-dashboard-focus *,
    .ledger-goal-card *, .history-row *, .empty-card *, .factor-card *,
    .score-trend-card *, .ledger-factor-row *, .ledger-factor-advice *,
    .ledger-factor-points, .goal-meta, .goal-category, .goal-card-bottom,
    .history-numbers, .stCaption,
    [data-testid="stCaptionContainer"] * {
      opacity:1 !important;
    }
    .ledger-dashboard-score .score-big, .ledger-dashboard-score .score-number,
    .ledger-dashboard-score .ledger-score-number, .ledger-dashboard-score .score-value,
    .ledger-dashboard-empty .score-big, .ledger-dashboard-focus strong,
    .ledger-dashboard-score strong, .ledger-dashboard-score b,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
      color:#333044 !important;
      -webkit-text-fill-color:#333044 !important;
    }
    .ledger-dashboard-score .dashboard-kicker, .ledger-dashboard-empty .dashboard-kicker,
    .ledger-dashboard-focus .dashboard-kicker, .factor-card .factor-copy,
    .factor-card .factor-title, .factor-card .factor-status,
    .score-trend-card .score-caption,
    .goal-meta, .goal-category, .goal-card-bottom, .history-numbers,
    [data-testid="stCaptionContainer"] * {
      color:#625A72 !important;
      -webkit-text-fill-color:#625A72 !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
      background:#FFFFFF !important; color:#333044 !important;
    }
    [role="option"], [role="option"] * {
      background:#FFFFFF !important; color:#333044 !important;
      -webkit-text-fill-color:#333044 !important; opacity:1 !important;
    }
    [role="option"][aria-selected="true"], [role="option"]:hover,
    [role="option"][aria-selected="true"] *, [role="option"]:hover * {
      background:#E9E0FF !important; color:#333044 !important;
      -webkit-text-fill-color:#333044 !important;
    }
    </style>
    """, unsafe_allow_html=True)

if APPEARANCE == "Dark":
    WHAT_IF_CONTROL_CSS = """
    <style>
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) [data-baseweb="select"],
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) [data-baseweb="select"] > div {
        background-color:#211E2D !important; border:1px solid #6A587F !important;
        border-radius:10px !important; color:#F7F2FF !important;
        -webkit-text-fill-color:#F7F2FF !important; box-shadow:none !important;
    }
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) [data-baseweb="select"] *,
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) input {
        color:#F7F2FF !important; -webkit-text-fill-color:#F7F2FF !important;
        caret-color:#F7F2FF !important; opacity:1 !important;
    }
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) svg { fill:#D8C5FF !important; color:#D8C5FF !important; }
    div[data-testid="stNumberInput"]:has(input[aria-label="Change amount (₹)"]) input,
    div[data-testid="stNumberInput"]:has(input[aria-label="How much would you reduce it by? (₹)"]) input {
        background:#211E2D !important; color:#F7F2FF !important; -webkit-text-fill-color:#F7F2FF !important;
        border:1px solid #6A587F !important; border-radius:10px !important; caret-color:#F7F2FF !important; opacity:1 !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] { background:#211E2D !important; border:1px solid #6A587F !important; }
    [role="option"], [role="option"] * { background:#211E2D !important; color:#F7F2FF !important; -webkit-text-fill-color:#F7F2FF !important; opacity:1 !important; }
    [role="option"][aria-selected="true"], [role="option"]:hover,
    [role="option"][aria-selected="true"] *, [role="option"]:hover * { background:#493660 !important; color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }
    </style>
    """
else:
    WHAT_IF_CONTROL_CSS = """
    <style>
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) [data-baseweb="select"],
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) [data-baseweb="select"] > div {
        background-color:#FFFFFF !important; border:1px solid #BBA9DD !important;
        border-radius:10px !important; color:#333044 !important;
        -webkit-text-fill-color:#333044 !important; box-shadow:none !important;
    }
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) [data-baseweb="select"] *,
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) input {
        color:#333044 !important; -webkit-text-fill-color:#333044 !important;
        caret-color:#333044 !important; opacity:1 !important;
    }
    div[data-testid="stSelectbox"]:has(input[aria-label="Choose a scenario"]) svg { fill:#7558B7 !important; color:#7558B7 !important; }
    div[data-testid="stNumberInput"]:has(input[aria-label="Change amount (₹)"]) input,
    div[data-testid="stNumberInput"]:has(input[aria-label="How much would you reduce it by? (₹)"]) input {
        background:#FFFFFF !important; color:#333044 !important; -webkit-text-fill-color:#333044 !important;
        border:1px solid #BBA9DD !important; border-radius:10px !important; caret-color:#333044 !important; opacity:1 !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] { background:#FFFFFF !important; border:1px solid #E4DDF2 !important; }
    [role="option"], [role="option"] * { background:#FFFFFF !important; color:#333044 !important; -webkit-text-fill-color:#333044 !important; opacity:1 !important; }
    [role="option"][aria-selected="true"], [role="option"]:hover,
    [role="option"][aria-selected="true"] *, [role="option"]:hover * { background:#E9E0FF !important; color:#333044 !important; -webkit-text-fill-color:#333044 !important; }
    </style>
    """
st.markdown(WHAT_IF_CONTROL_CSS, unsafe_allow_html=True)

if APPEARANCE == "Dark":
    st.markdown("""
    <style>
    div[data-testid="stSelectbox"] [data-baseweb="select"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
        background-color:#211E2D !important;
        color:#F7F2FF !important;
        -webkit-text-fill-color:#F7F2FF !important;
        border-color:#6A587F !important;
        opacity:1 !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] *,
    div[data-testid="stSelectbox"] [data-baseweb="select"] input {
        color:#F7F2FF !important;
        -webkit-text-fill-color:#F7F2FF !important;
        opacity:1 !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
        fill:#D8C5FF !important; color:#D8C5FF !important;
    }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
        background-color:#211E2D !important; color:#F7F2FF !important;
    }
    [role="option"], [role="option"] * {
        background-color:#211E2D !important; color:#F7F2FF !important;
        -webkit-text-fill-color:#F7F2FF !important; opacity:1 !important;
    }
    [role="option"][aria-selected="true"], [role="option"]:hover,
    [role="option"][aria-selected="true"] *, [role="option"]:hover * {
        background-color:#493660 !important; color:#FFFFFF !important;
        -webkit-text-fill-color:#FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    f'<div style="color:{C["muted"]};font-size:.72rem;text-align:center;margin-top:2rem;'
    f'padding-top:.9rem;border-top:1px solid {C["line"]};">Educational tool, not financial advice. '
    'Scores come from a model trained on synthetic data. Your entries are stored on this device.</div>',
    unsafe_allow_html=True,
)