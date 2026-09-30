from __future__ import annotations

# CT Equivalence Lab page entry point.
# Streamlit may execute a page module with app/pages as the script location,
# so ensure the repository root is importable before resolving the app package.
# This is import bootstrapping only; no scientific backend is changed.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.research_workspaces_v12 import render_research_extension

render_research_extension()
