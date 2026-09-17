from __future__ import annotations

"""Human-readable reports designed for discussion with supervisors/professors."""

from pathlib import Path

import numpy as np
import sympy as sp

from .ct import analyze_austenite_martensite
from .cualni_models import (
    do3_to_2h_branch,
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from .group_theory import correspondence_groupoid
from .james_hane import cube_edge_6m_variants
from .reference_6m import (
    C_REF_A_FROM_M,
    C_REF_M_FROM_A,
    DO3_6M_BASIS_CONVENTION,
    parent_axis_permutation_relating_reference_to_alt,
)
from .stretch import (
    generate_stretch_variants,
    optimal_match_matrix_family,
    principal_stretches,
    stretch_from_metrics,
)
from .symmetry import classify_symmetry, cubic_proper_rotations


def _m(M):
    return sp.pretty(sp.Matrix(M))


def branch_report(branch) -> str:
    g = correspondence_groupoid(
        list(branch.parent_point_group),
        list(branch.product_point_group),
        branch.correspondence,
    )
    lines = [
        f"# {branch.name}: exact discrete CT report",
        "",
        "Direct-space convention: `u_M = C_M_from_A @ u_A`.",
        "Plane-covector convention: `p_M = C_M_from_A^{-T} @ p_A`.",
        "",
        "## Selected reference correspondence variant",
        "```text",
        _m(branch.correspondence.C_M_from_A),
        "```",
        "",
        f"Parent group order: {len(branch.parent_point_group)}",
        f"Product group order: {len(branch.product_point_group)}",
        f"|H_C| = {len(g.subgroup)}",
        f"N_C = {g.n_variants}",
        f"N_operators = {g.n_operators}",
        f"Double-coset sizes = {[len(x) for x in g.operators]}",
        f"Burnside cross-check = {g.burnside_count}",
        "",
        "## H_C",
    ]
    for h in g.subgroup:
        info = classify_symmetry(h)
        lines += [
            "```text",
            _m(h),
            "```",
            f"kind={info.kind}, det={info.determinant}, order={info.order}",
            "",
        ]
    lines += ["## Operator summaries"]
    for s in g.summaries:
        lines.append(
            f"- O{s.index}: size={s.size}, inverse=O{s.inverse_index}, "
            f"ambivalent={s.ambivalent}, contents={dict(s.symmetry_kinds)}"
        )
    lines += ["", "## Variant -> operator adjacency", "```text"]
    lines.extend(" ".join(f"{x:2d}" for x in row) for row in g.adjacency)
    lines += ["```", ""]
    return "\n".join(lines)


def verification_report() -> str:
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()
    U = stretch_from_metrics(A.metric(), M.metric(), branch.correspondence)
    lam, _ = principal_stretches(U)
    derived = generate_stretch_variants(
        U,
        [np.array(q, float) for q in cubic_proper_rotations()],
    )
    ref = cube_edge_6m_variants(A.a, M.a, M.b, M.c, M.beta_deg)
    match = optimal_match_matrix_family(derived, ref, tol=1e-12)
    ct = analyze_austenite_martensite(
        A.metric(),
        M.metric(),
        branch.correspondence,
    )
    P = parent_axis_permutation_relating_reference_to_alt()
    basis = DO3_6M_BASIS_CONVENTION

    match_lines = [
        f"- derived U{i} -> James-Hane U{j}: residual={r:.3e}"
        for (i, j), r in zip(match.mapping, match.residuals, strict=True)
    ]

    return "\n".join(
        [
            "# Independent verification report: DO3 -> 6M",
            "",
            "The James-Hane Cu-Al-Ni numbers are used only as a reproducibility benchmark.",
            "They are not default specimen data.",
            "",
            "## Basis and correspondence truth lock",
            f"- parent basis order: {basis.parent_axis_order}",
            f"- daughter basis order: {basis.daughter_axis_order}",
            f"- non-right angle: {basis.non_right_angle_between}",
            f"- source symbol: {basis.source_angle_symbol}",
            f"- internal symbol: {basis.internal_angle_symbol}",
            "",
            "`C_A_from_M` (daughter basis vectors written in parent coordinates):",
            "```text",
            _m(C_REF_A_FROM_M),
            "```",
            "`C_M_from_A = C_A_from_M^{-1}`:",
            "```text",
            _m(C_REF_M_FROM_A),
            "```",
            "Earlier exploratory matrix is related by the proper cubic parent-axis permutation:",
            "```text",
            _m(P),
            "```",
            "",
            "## Stretch from correspondence + metrics",
            "```text",
            np.array2string(U, precision=12),
            "```",
            f"Principal stretches: {np.array2string(lam, precision=12)}",
            f"|lambda2-1| = {abs(lam[1] - 1):.12g}",
            f"Number of symmetry-generated stretch variants = {len(derived)}",
            f"Optimal one-to-one James-Hane Eq.(10) family match = {match.success}",
            f"Maximum Frobenius mismatch = {match.maximum_residual:.3e}",
            f"RMS Frobenius mismatch = {match.rms_residual:.3e}",
            "",
            "### Optimal variant-family assignment",
            *match_lines,
            "",
            "## CMC",
            f"normalized CMC eigenvalues = {np.array2string(ct.analysis.eigenvalues, precision=12)}",
            f"exact CT A/M compatibility at rounded literature parameters = {ct.analysis.exact_compatible}",
            f"nearest normalized CMC eigenvalue magnitude = {ct.analysis.nearest_zero_residual:.12g}",
            "",
            (
                "Interpretation: the source reports a nearly compatible Cu-Al-Ni example. "
                "Exact equality must not be claimed from rounded Table 4 lattice parameters."
            ),
        ]
    )


def write_all_reports(directory: str | Path) -> list[Path]:
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, text in [
        ("DO3_6M_GROUPoid.md", branch_report(do3_to_6m_branch())),
        ("DO3_2H_GROUPoid.md", branch_report(do3_to_2h_branch())),
        ("DO3_6M_VERIFICATION.md", verification_report()),
    ]:
        path = base / name
        path.write_text(text, encoding="utf-8")
        outputs.append(path)
    return outputs
