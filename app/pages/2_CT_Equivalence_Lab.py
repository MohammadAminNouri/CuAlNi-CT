from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.research_workspaces_v3 import render_research_extension

st.set_page_config(
    page_title="CT Equivalence Laboratory",
    page_icon="🔬",
    layout="wide",
)

render_research_extension()
