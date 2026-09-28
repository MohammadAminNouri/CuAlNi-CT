from __future__ import annotations

"""Stable Streamlit entry point."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.streamlit_workstation_v4 import main

main()
