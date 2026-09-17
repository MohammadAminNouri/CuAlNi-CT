"""Metric-only example using the Cu-Al-Ni values tabulated by James & Hane (2000).

This example deliberately does NOT insert a DO3->6M correspondence matrix.
The point is to show the dimensional and normalized metric layers without inventing C.
"""
import json
from pathlib import Path

from cualni_cryst.lattice import Lattice

HERE = Path(__file__).resolve().parents[1]
data = json.loads((HERE / "data/literature/james_hane_2000_cualni_6m_example.json").read_text())

A = Lattice.cubic(data["parent"]["a_A_angstrom"], "DO3 parent")
M = Lattice.monoclinic_unique_b(
    data["martensite_6M"]["a_angstrom"],
    data["martensite_6M"]["b_angstrom"],
    data["martensite_6M"]["c_angstrom"],
    data["martensite_6M"]["beta_deg"],
    "6M martensite",
)

MA = A.metric(); MM = M.metric()
print("M_A [A^2]:\n", MA)
print("M_M [A^2]:\n", MM)
print("Simple parent-scale normalized M_M = M_M/a_A^2:\n", MM / A.a**2)
print("No CT calculation is performed because a verified correspondence C was not supplied.")
