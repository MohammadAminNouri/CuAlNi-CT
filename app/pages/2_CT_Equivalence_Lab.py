from __future__ import annotations

"""CT Equivalence Lab page entry point.

Streamlit can execute a page module with ``app/pages`` as the script location.
Ensure the repository root is importable before resolving the ``app`` package.
This changes only page import bootstrapping; no scientific backend is touched.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.research_workspaces_v12 import render_research_extension

render_research_extension()
