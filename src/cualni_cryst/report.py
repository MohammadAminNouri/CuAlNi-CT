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
from .twin_compare import build_twin_atlas


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


def twin_atlas_report() -> str:
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()
    atlas = build_twin_atlas(branch, A.metric(), M.metric())

    lines = [
        "# DO3 -> 6M operator/twin atlas: CT vs Ball-James/Mallard",
        "",
        "Benchmark only: James-Hane Cu-14 wt% Al-4 wt% Ni rounded lattice parameters.",
        "The numerical shears below are not universal Cu-Al-Ni constants.",
        "",
        "## Operator summary",
        "",
        "| Operator | Size | Classification | Proper rotation orders | Exact relations | Shear(s) |",
        "|---:|---:|---|---|---:|---|",
    ]

    for op in atlas.operators:
        shears = ", ".join(f"{r.shear:.12g}" for r in op.relations) or "-"
        lines.append(
            f"| O{op.operator_index} | {op.size} | {op.classification} | "
            f"{op.proper_rotation_orders or '-'} | {len(op.relations)} | {shears} |"
        )

    lines += [
        "",
        f"Total nontrivial exact reference-to-target relations: {atlas.n_exact_relations}",
        f"Maximum CT-vs-Mallard geometry angle: {atlas.max_geometry_angle_deg:.3e} deg",
        f"Maximum relative shear mismatch: {atlas.max_relative_shear_residual:.3e}",
        f"Maximum Mallard rank-one residual: {atlas.max_rank_one_residual:.3e}",
        "",
        "## Exact relation details",
        "",
    ]

    for op in atlas.operators:
        if not op.relations:
            continue
        lines += [f"### O{op.operator_index} — {op.classification}", ""]
        for j, rel in enumerate(op.relations, start=1):
            lines += [
                f"Relation {j}: target stretch U{rel.target_stretch_index}",
                "",
                f"- parent twofold generator axes: {rel.generator_axes}",
                f"- compound: {rel.compound}",
                f"- shear: {rel.shear:.12g}",
                f"- |s_CT-I - s_CT-II|: {rel.ct_type_i_vs_ii_shear_abs:.3e}",
                f"- Type-I CT/Mallard relative shear residual: {rel.type_i_shear_rel_residual:.3e}",
                f"- Type-I K1/reference-normal angle: {rel.type_i_plane_angle_deg:.3e} deg",
                f"- Type-I deformed shear-direction angle: {rel.type_i_direction_angle_deg:.3e} deg",
                f"- Type-II CT/Mallard relative shear residual: {rel.type_ii_shear_rel_residual:.3e}",
                f"- Type-II K2/reference-normal angle: {rel.type_ii_plane_angle_deg:.3e} deg",
                f"- Type-II deformed shear-direction angle: {rel.type_ii_direction_angle_deg:.3e} deg",
                f"- Mallard Type-I rank-one residual: {rel.bj_type_i.residual:.3e}",
                f"- Mallard Type-II rank-one residual: {rel.bj_type_ii.residual:.3e}",
                "",
            ]

    lines += [
        "## Scientific interpretation",
        "",
        (
            "CT and Mallard are not being compared by feeding one theory's plane or "
            "direction into the other. They are calculated independently."
        ),
        "",
        (
            "The apparent mismatch between raw CT shear-direction coordinates and "
            "Ball-James `a` disappears only after respecting configuration: CT stores "
            "the direct direction in the parent reference coordinates, while Ball-James "
            "`a` is a deformed/current shear vector. The report therefore compares "
            "`U_j d_CT` with `a_BJ`."
        ),
        "",
        (
            "Operators classified as weak candidates are deliberately not promoted "
            "to exact twins in this report."
        ),
    ]
    return "\n".join(lines)



def compatibility_report() -> str:
    """A/M and A/M/M report for the source-rounded benchmark and exact-beta control."""

    from .compatibility_atlas import james_hane_benchmark_projection

    result = james_hane_benchmark_projection()
    obs = result.observed
    proj = result.projected

    lines = [
        "# DO3 -> 6M A/M and A/M/M compatibility atlas",
        "",
        "## Source-rounded literature benchmark",
        "",
        f"- beta = {obs.state.beta_deg:.8f} deg",
        f"- lambdas = {obs.state.lambdas}",
        f"- lambda2-1 = {obs.state.lambda2_residual:.6e}",
        f"- normalized CMC eigenvalues = {obs.state.normalized_cmc_eigenvalues}",
        f"- exact A/M compatible = {obs.state.exact_compatible}",
        f"- nearest CMC residual = {obs.state.nearest_cmc_residual:.6e}",
        f"- exact habit planes = {obs.state.n_exact_habit_planes}",
        f"- approximate diagnostic planes = {obs.state.n_approximate_habit_planes}",
        "",
        "No A/M/M epsilon is promoted to an exact result for this non-exact state.",
        "",
        "## Exact-compatible beta projection (HYPOTHETICAL_TEST)",
        "",
        f"- projected beta = {result.projected_beta_deg:.10f} deg",
        f"- beta shift from rounded benchmark = {result.beta_shift_deg:+.10f} deg",
        f"- exact beta candidates = {result.exact_beta_candidates_deg}",
        f"- lambdas = {proj.state.lambdas}",
        f"- lambda2-1 = {proj.state.lambda2_residual:.6e}",
        f"- normalized CMC eigenvalues = {proj.state.normalized_cmc_eigenvalues}",
        f"- exact habit planes = {proj.state.n_exact_habit_planes}",
        "",
        "### CMC vs Ball-James single-variant habit cross-check",
        "",
        "| CMC habit | BJ branch | plane angle (deg) | rank-one residual |",
        "|---:|---:|---:|---:|",
    ]

    for row in proj.am_habit_crosscheck:
        lines.append(
            f"| {row.cmc_habit_index} | {row.ball_james_branch:+d} | "
            f"{row.plane_angle_deg:.3e} | {row.ball_james_rank_one_residual:.3e} |"
        )

    lines += [
        "",
        "### Exact M/M systems tested against A/M compatibility",
        "",
        (
            "| O | relation | twin | compound | shear | best habit | epsilon_CT | "
            "angle(d_A,a) deg | CC1 | CC2 | CC3 margin | cofactor | PTMC f |"
        ),
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for row in proj.twin_systems:
        roots = ", ".join(f"{x:.8f}" for x in row.ptmc_volume_fractions) or "-"
        lines.append(
            f"| O{row.operator_index} | {row.relation_index} | {row.twin_kind} | "
            f"{row.compound} | {row.twin_shear:.8f} | {row.best_habit_index} | "
            f"{row.supercompatibility_residual:.6e} | "
            f"{row.shear_direction_angle_deg:.6f} | "
            f"{row.cc1_residual:.3e} | {row.cc2_residual:.3e} | "
            f"{row.cc3_margin:.3e} | {row.cofactor_satisfied} | {roots} |"
        )

    lines += [
        "",
        "Interpretation:",
        "",
        "- The observed/source-rounded state and hypothetical exact-compatible control are never mixed.",
        "- epsilon_CT is a dimensionless incompatibility amplitude, not an energy.",
        "- CC1/CC2/CC3 and PTMC roots are calculated independently from Mallard/Ball-James data.",
        "- A future interactive frontend can consume the same backend dataclasses as JSON.",
    ]
    return "\n".join(lines)

def write_all_reports(directory: str | Path) -> list[Path]:
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, text in [
        ("DO3_6M_GROUPoid.md", branch_report(do3_to_6m_branch())),
        ("DO3_2H_GROUPoid.md", branch_report(do3_to_2h_branch())),
        ("DO3_6M_VERIFICATION.md", verification_report()),
        ("DO3_6M_TWIN_ATLAS.md", twin_atlas_report()),
        ("DO3_6M_COMPATIBILITY_ATLAS.md", compatibility_report()),
    ]:
        path = base / name
        path.write_text(text, encoding="utf-8")
        outputs.append(path)
    return outputs
