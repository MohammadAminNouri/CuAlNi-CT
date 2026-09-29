from __future__ import annotations

"""Stable Streamlit entry point for the Phase-1/V6 workstation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.streamlit_workstation_v6 import main

main()