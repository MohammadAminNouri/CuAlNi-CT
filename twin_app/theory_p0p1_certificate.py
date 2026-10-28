from __future__ import annotations

"""P0/P1 read-only crystallographic certificate for the EXISTING project.

P0: independent parent and product passive-basis covariance, direct/reciprocal
duality, symmetry operations and Cayron CMC/SMC/shear-shear invariants.
P1: verify the actual CT branch residuals and reuse independently recomputed
BJ cofactor/rank-one and PTMC rank-one evidence from theory_evidence_ui.

A passive GL(3,Z) basis change is not a new correspondence, phase, OR, angle,
material model or numerical prediction.  The tests deliberately do not change
the crystallographic engines or their numerical tolerances.

Primary sources: Cayron, Acta Cryst. A75 (2019) 411–437 and corrigendum p777;
Cayron, Acta Mater. 316 (2026) 122399 Eqs. 32, 37, 41, 43, 49;
Ball & James, Arch. Rational Mech. Anal. 100 (1987) 13–52;
Chen et al., JMPS 61 (2013) 2566–2587, Theorem 2.
"""

from collections.abc import Sequence
import math

import numpy as np
from scipy.linalg import eigh


PARENT_REBASE = np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])
PRODUCT_REBASE = np.array([[1., 0., 0.], [0., 1., 1.], [0., 0., 1.]])

# These are *verification tolerances*, never substituted into the backend.
DEFAULT_AUDIT_TOLERANCE = 1.0e-7


def _finite_matrix(value, *, name: str):
    out = np.asarray(value, dtype=float)
    if out.shape != (3, 3) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name}: finite 3x3 matrix required")
    return out


def _metric(value, name):
    M = _finite_matrix(value, name=name)
    if not np.allclose(M, M.T, rtol=0, atol=1e-12 * max(np.linalg.norm(M), 1.)):
        raise ValueError(f"{name}: metric is not symmetric")
    if np.min(np.linalg.eigvalsh(M)) <= 0:
        raise ValueError(f"{name}: metric is not positive definite")
    return 0.5 * (M + M.T)


def _relative(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    return float(np.linalg.norm(left - right) /
                 max(np.linalg.norm(left), np.linalg.norm(right),
                     np.finfo(float).tiny))


def _unit_covector(p, M):
    p = np.asarray(p, float).reshape(3)
    return p / math.sqrt(float(p @ np.linalg.solve(M, p)))


def _metric_norm(v, M):
    v = np.asarray(v, float).reshape(3)
    return math.sqrt(max(0., float(v @ M @ v)))


def independent_shear_shear_epsilon(M, plane_habit, shear_ips,
                                   plane_twin, direction_twin, magnitude):
    """Independently reproduce Cayron (2026) Eqs. 43/49 in parent coordinates.

    m is a reciprocal unit covector; n is a direct unit normal to the twin
    covector; a is a metric-unit twin shear direction multiplied by |s|.
    """
    M = _metric(M, "parent metric")
    mag = float(magnitude)
    if not math.isfinite(mag) or mag <= 0:
        raise ValueError("nonzero positive twin shear required")
    m = _unit_covector(plane_habit, M)
    h = _unit_covector(plane_twin, M)
    n = np.linalg.solve(M, h)
    eta = np.asarray(direction_twin, float).reshape(3)
    eta = eta / _metric_norm(eta, M)
    a = mag * eta
    d = np.asarray(shear_ips, float).reshape(3)
    mismatch = 2. * float(m @ n) * d - a
    return _metric_norm(mismatch, M) / mag


def _group_audit(operations: Sequence, metric, *, tolerance):
    """A measured/DECLARED finite group can be valid without being proven actual.

    Following Cayron's corrigendum, passing metric preservation is NECESSARY
    but never sufficient evidence that the operations are actual crystal
    symmetries of the measured atomic structure.
    """
    M = _metric(metric, "symmetry metric")
    group = [_finite_matrix(g, name="symmetry operation") for g in operations]
    if not group:
        raise ValueError("empty declared point group")
    is_integer = [
        _relative(g, np.rint(g)) <= tolerance for g in group
    ]
    is_unimodular = [
        abs(abs(float(np.linalg.det(g))) - 1.) <= tolerance for g in group
    ]
    metric_gaps = [_relative(g.T @ M @ g, M) for g in group]
    finite_group = len(group) <= 48
    identity = any(_relative(g, np.eye(3)) <= tolerance for g in group)
    # Lattice point-group matrices are exact integral in these crystal
    # bases. O(N^2) exact closure avoids an O(N^3) floating comparison.
    if finite_group and all(is_integer):
        integral = [np.rint(g).astype(np.int64) for g in group]
        keys = {tuple(g.reshape(-1)) for g in integral}
        closure = all(
            tuple((a @ b).reshape(-1)) in keys
            for a in integral for b in integral
        )
    else:
        closure = all(
            any(_relative(a @ b, c) <= tolerance for c in group)
            for a in group for b in group
        ) if finite_group else False
    passed = (all(is_integer) and all(is_unimodular)
              and max(metric_gaps) <= tolerance and identity and closure)
    return {
        "computed_group_pass": bool(passed),
        "count": len(group),
        "all_integer_lattice_automorphisms": bool(all(is_integer)),
        "all_det_plus_or_minus_one": bool(all(is_unimodular)),
        "max_metric_preservation_relative_error": float(max(metric_gaps)),
        "contains_identity": bool(identity),
        "group_closed": bool(closure),
        "actual_atomic_structure_symmetry_verified": False,
        "external_evidence_required": (
            "A metric-preserving lattice operation need not be a symmetry "
            "of the crystal structure (Cayron 2019 corrigendum, DOI "
            "10.1107/S2053273319009276). Phase structure/space-group "
            "provenance must be verified independently."
        ),
    }


def passive_basis_covariance(M_a, M_m, C_m_from_a, parent_ops,
                             product_ops, *, tolerance=DEFAULT_AUDIT_TOLERANCE):
    """Test parent-only, product-only and independent simultaneous GL(3,Z) changes.

    B'_A=B_A P_A, B'_M=B_M P_M,
    M'_A=P_A^T M_A P_A, M'_M=P_M^T M_M P_M,
    C'=P_M^-1 C P_A, u'_A=P_A^-1 u_A, h'_A=P_A^T h_A.
    These identities test representation invariance, *not* experimental OR.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid audit tolerance")
    M_a = _metric(M_a, "parent")
    M_m = _metric(M_m, "product")
    C = _finite_matrix(C_m_from_a, name="C_M_from_A")
    if abs(float(np.linalg.det(C))) <= 1.e-14:
        raise ValueError("singular correspondence")

    pa = _group_audit(parent_ops, M_a, tolerance=tolerance)
    pm = _group_audit(product_ops, M_m, tolerance=tolerance)

    P_cases = (
        ("parent_only", PARENT_REBASE, np.eye(3)),
        ("product_only", np.eye(3), PRODUCT_REBASE),
        ("both_independently", PARENT_REBASE, PRODUCT_REBASE),
    )
    parent_pulled = C.T @ M_m @ C
    cmc = parent_pulled - M_a
    # SMC is contravariant in parent crystal coordinates.
    smc = np.linalg.inv(M_a) - np.linalg.inv(parent_pulled)
    mu = eigh(parent_pulled, M_a, eigvals_only=True, check_finite=True)
    if np.min(mu) <= 0:
        raise ValueError("invalid generalized metric eigenvalues")
    u = np.array([1., 2., -1.])
    h = np.array([2., -1., 0.])  # h^T u = 0, remains invariant
    baseline_incidence = float(h @ u)
    if baseline_incidence != 0:
        raise AssertionError("incidence fixture malformed")
    baseline_length = _metric_norm(u, M_a)
    baseline_plane_norm = math.sqrt(float(h @ np.linalg.solve(M_a, h)))
    result = {}

    for label, P_a, P_m in P_cases:
        if abs(round(np.linalg.det(P_a))) != 1 or abs(round(np.linalg.det(P_m))) != 1:
            raise AssertionError("passive matrices must be unimodular")
        ia = np.linalg.inv(P_a)
        im = np.linalg.inv(P_m)
        Ma_r = P_a.T @ M_a @ P_a
        Mm_r = P_m.T @ M_m @ P_m
        C_r = im @ C @ P_a
        pulled_r = C_r.T @ Mm_r @ C_r
        cmc_r = pulled_r - Ma_r
        smc_r = np.linalg.inv(Ma_r) - np.linalg.inv(pulled_r)
        mu_r = eigh(pulled_r, Ma_r, eigvals_only=True, check_finite=True)
        u_r, h_r = ia @ u, P_a.T @ h
        parent_rebased_ops = tuple(ia @ np.asarray(g, float) @ P_a for g in parent_ops)
        product_rebased_ops = tuple(im @ np.asarray(g, float) @ P_m for g in product_ops)
        parent_group_r = _group_audit(parent_rebased_ops, Ma_r, tolerance=tolerance)
        product_group_r = _group_audit(product_rebased_ops, Mm_r, tolerance=tolerance)

        # Exact dual parent/product correspondence incidence:
        # (C^-T h)^T (C u) = h^T u; similarly for the rebased state.
        product_u_r = C_r @ u_r
        product_h_r = np.linalg.solve(C_r.T, h_r)
        checks = {
            # Scale near-zero CMC/SMC residuals to parent *metrics*,
            # not to near-zero tensors themselves. A compatible lattice
            # can have CMC ~= 0; relative-to-CMC would give a false failure.
            "CMC_covariant": float(
                np.linalg.norm(cmc_r - P_a.T @ cmc @ P_a) /
                max(np.linalg.norm(Ma_r), np.linalg.norm(pulled_r), 1.)),
            "pulled_product_metric_covariant": _relative(
                pulled_r, P_a.T @ parent_pulled @ P_a),
            "SMC_contravariant": float(
                np.linalg.norm(smc_r - ia @ smc @ ia.T) /
                max(np.linalg.norm(np.linalg.inv(Ma_r)),
                    np.linalg.norm(np.linalg.inv(pulled_r)), 1.)),
            "generalized_eigenvalues": _relative(mu_r, mu),
            "parent_direct_vector_norm": abs(
                _metric_norm(u_r, Ma_r) / baseline_length - 1.),
            "parent_reciprocal_plane_norm": abs(
                math.sqrt(float(h_r @ np.linalg.solve(Ma_r, h_r)))
                / baseline_plane_norm - 1.),
            "parent_plane_direction_incidence": abs(float(h_r @ u_r) -
                                                    baseline_incidence),
            "mapped_product_incidence": abs(float(product_h_r @ product_u_r) -
                                            baseline_incidence),
            "parent_symmetry_metric": parent_group_r[
                "max_metric_preservation_relative_error"],
            "product_symmetry_metric": product_group_r[
                "max_metric_preservation_relative_error"],
        }
        # A metric covariance certificate cannot override an invalid
        # explicitly supplied symmetry group.
        passed = (all(math.isfinite(x) and x <= tolerance
                      for x in checks.values())
                  and pa["computed_group_pass"] and pm["computed_group_pass"]
                  and parent_group_r["computed_group_pass"]
                  and product_group_r["computed_group_pass"])
        result[label] = {
            "pass": bool(passed),
            "relative_residuals": checks,
            "maximum_residual": max(checks.values()),
        }

    return {
        "pass": bool(all(item["pass"] for item in result.values())),
        "parent_declared_group": pa,
        "product_declared_group": pm,
        "parent_only": result["parent_only"],
        "product_only": result["product_only"],
        "both_independently": result["both_independently"],
        "generalized_stretches": [float(math.sqrt(v)) for v in mu],
        "actual_structure_symmetry_verified": False,
        "experimental_correspondence_verified": False,
        "audit_tolerance": tolerance,
    }


def certify_existing_theory(project, transformation_id, unified, *,
                            tolerance=DEFAULT_AUDIT_TOLERANCE,
                            max_shear_pairs=256):
    """Compute P0/P1 certificate without changing any theoretical solution."""
    tr = project.transformation(transformation_id)
    parent = project.phase(tr.parent_phase_id)
    product = project.phase(tr.product_phase_id)
    Ma, Mm = parent.lattice.metric(), product.lattice.metric()
    C = tr.correspondence.to_float64_checked()
    basis = passive_basis_covariance(
        Ma, Mm, C, parent.symmetry_matrices(), product.symmetry_matrices(),
        tolerance=tolerance,
    )
    from cualni_cryst.theory_unified import PredictionKind
    from twin_app.theory_evidence_ui import (
        ball_james_branch_evidence, ptmc_habit_evidence,
    )

    if not isinstance(max_shear_pairs, int) or max_shear_pairs <= 0:
        raise ValueError("max_shear_pairs must be a positive integer")
    pairs = [r for r in unified.rows
             if r.prediction_kind == PredictionKind.CT_SUPERCOMPATIBILITY]
    # Independently test actual CT outputs, not just tensor covariance.
    # This is distinct from both CT's own CMC and SMC implementations.
    G = C.T @ Mm @ C
    computed_cmc = G - Ma
    ctreport = getattr(unified, "ct_report", None)
    cmc_report_gap = (
        None if ctreport is None else
        float(np.linalg.norm(np.asarray(ctreport.cmc_dimensional, float)
                             - computed_cmc) /
              max(np.linalg.norm(G), np.linalg.norm(Ma), 1.))
    )
    ct_spectrum_gap = (
        None if ctreport is None else
        _relative(
            np.sort(np.asarray(ctreport.analysis.eigenvalues, float)),
            np.sort(eigh(G, Ma, eigvals_only=True) - 1.))
    )
    smc_independent = np.linalg.inv(Ma) - np.linalg.inv(G)
    habits = [r for r in unified.rows
              if r.prediction_kind == PredictionKind.CT_AM_HABIT]
    habit_shear_checks = []
    for row in habits:
        expected_d = smc_independent @ _unit_covector(
            row.habit_plane_parent_crystal, Ma)
        recorded_d = np.asarray(row.shape_vector_parent_crystal, float)
        habit_shear_checks.append({
            "ct_branch": row.row_id,
            "SMC_shear_difference": float(
                np.linalg.norm(expected_d - recorded_d) /
                max(np.linalg.norm(expected_d),
                    np.linalg.norm(recorded_d), 1.)),
        })
    cmc_smc_pass = (
        (cmc_report_gap is None or cmc_report_gap <= tolerance)
        and (ct_spectrum_gap is None or ct_spectrum_gap <= tolerance)
        and all(x["SMC_shear_difference"] <= tolerance
                for x in habit_shear_checks)
    )
    inspected = []
    for row in pairs[:max_shear_pairs]:
        expected = float(row.residuals["ct_supercompatibility_dimensionless"])
        independent = independent_shear_shear_epsilon(
            Ma, row.habit_plane_parent_crystal, row.shape_vector_parent_crystal,
            row.twin_plane_parent_crystal, row.twin_direction_parent_crystal,
            row.shear_magnitude,
        )
        rebase = PARENT_REBASE
        inv = np.linalg.inv(rebase)
        rebased = independent_shear_shear_epsilon(
            rebase.T @ Ma @ rebase,
            rebase.T @ np.asarray(row.habit_plane_parent_crystal),
            inv @ np.asarray(row.shape_vector_parent_crystal),
            rebase.T @ np.asarray(row.twin_plane_parent_crystal),
            inv @ np.asarray(row.twin_direction_parent_crystal),
            row.shear_magnitude,
        )
        inspected.append({
            "ct_branch": row.row_id,
            "original_epsilon": expected,
            "independent_epsilon": independent,
            "rebased_epsilon": rebased,
            "reported_vs_independent_error": abs(expected - independent),
            "passive_basis_error": abs(rebased - independent),
        })

    bj = ball_james_branch_evidence(unified, tolerance=tolerance)
    ptmc = ptmc_habit_evidence(unified, tolerance=tolerance)
    errors = [max(x["reported_vs_independent_error"],
                  x["passive_basis_error"]) for x in inspected]
    pass_shear = all(math.isfinite(x) and x <= tolerance for x in errors)
    pass_bj = all(x["Audit status"] == "REPRODUCED" for x in bj)
    pass_ptmc = all(x["Audit status"] == "REPRODUCED" for x in ptmc)
    # No assertions about experimental phase assignment, finite temperature,
    # or realization/selection of actual twin/habit microstructures.
    return {
        "P0_computational_pass": basis["pass"],
        "P0_basis_covariance": basis,
        "P1_computational_pass": bool(cmc_smc_pass and pass_shear
                                        and pass_bj and pass_ptmc
                                        and len(pairs) <= max_shear_pairs),
        "P1_CMC_SMC": {
            "CT_CMC_reported": cmc_report_gap is not None,
            "CMC_independent_matrix_gap": cmc_report_gap,
            "CMC_generalized_eigenspectrum_gap": ct_spectrum_gap,
            "SMC_shears_audited": len(habit_shear_checks),
            "SMC_shear_checks": habit_shear_checks,
            "verified": bool(cmc_smc_pass),
        },
        "P1_shear_shear": {
            "total_solver_pairs": len(pairs),
            "audited_pairs": len(inspected),
            "coverage": "COMPLETE" if len(pairs) <= max_shear_pairs else "PARTIAL",
            "max_independent_error": max(errors, default=None),
            "pairs": inspected,
            "verified": bool(pass_shear),
        },
        "P1_Ball_James": {
            "audited_branches": len(bj),
            "review_required": sum(x["Audit status"] != "REPRODUCED" for x in bj),
        },
        "P1_PTMC": {
            "audited_habits": len(ptmc),
            "review_required": sum(x["Audit status"] != "REPRODUCED" for x in ptmc),
        },
        "thesis_claim_scope": (
            "COMPUTATIONAL CONSISTENCY ONLY. All result-derived crystalline "
            "symmetries, lattice settings and correspondence are assumed "
            "user supplied and NOT established as a specimen's atomic "
            "structure. CT/CC/PTMC passes are not observational evidence "
            "for twin selection or EBSD habit-plane formation."
        ),
        "source_scope": (
            "Cayron 2019 + corrigendum and 2026 Eqs. 32, 37, 41, 43, 49; "
            "Ball–James 1987 rank-one compatibility; Chen et al. "
            "2013 Theorem 2 CC1–CC3; classical PTMC."
        ),
    }


def render_certificate(certificate):
    import pandas as pd
    import streamlit as st

    with st.expander("P0 / P1 — independent crystallographic verification", expanded=False):
        st.caption(
            "No new physical inputs: both-phase passive basis changes, "
            "tensor covariance, the original CT shear/shear and existing "
            "BJ/PTMC rank-one and cofactor outputs."
        )
        if certificate.get("audit_failed"):
            st.error("P0/P1 verification NOT ESTABLISHED: " + certificate["audit_failed"])
            return
        st.write({
            "P0 coordinate/matrix covariance": certificate["P0_computational_pass"],
            "P1 reproduced existing theoretical outputs": certificate["P1_computational_pass"],
            "P1 CMC/SMC reproduced": certificate["P1_CMC_SMC"]["verified"],
            "P1 CT pairs verified":
                certificate["P1_shear_shear"]["audited_pairs"],
            "P1 CT pair coverage": certificate["P1_shear_shear"]["coverage"],
        })
        for phase in ("parent_declared_group", "product_declared_group"):
            entry = certificate["P0_basis_covariance"][phase]
            if not entry["computed_group_pass"]:
                st.error(f"{phase}: declared group is not geometrically consistent.")
        basis_rows = []
        for item in ("parent_only", "product_only", "both_independently"):
            values = certificate["P0_basis_covariance"][item]
            basis_rows.append({
                "Passive change": item,
                "Numerically reproduced": values["pass"],
                "Maximum residual": values["maximum_residual"],
            })
        st.dataframe(pd.DataFrame(basis_rows), hide_index=True, width="stretch")
        if certificate["P1_shear_shear"]["coverage"] == "PARTIAL":
            st.warning("CT epsilon cross-check is sampled, not a full branch audit.")
        st.warning(
            "Cayron 2019 corrigendum: a valid lattice metric is not evidence "
            "that an operation is an ACTUAL crystal symmetry. Phase identity, "
            "structure setting and parent→martensite correspondence still "
            "require independent source/EBSD verification."
        )


def thesis_certificate_notes(certificate):
    if certificate.get("audit_failed"):
        return ("## P0/P1 independent verification\n"
                "NOT ESTABLISHED: " + certificate["audit_failed"] + "\n")
    p = certificate["P1_shear_shear"]
    return "\n".join((
        "## P0/P1 independent crystallographic certificate",
        "",
        f"P0 both-phase passive basis covariance: {certificate['P0_computational_pass']}.",
        f"P1 existing CT/BJ/PTMC reconstruction: {certificate['P1_computational_pass']}.",
        f"CT CMC/SMC independent reproduction: {certificate['P1_CMC_SMC']['verified']}.",
        f"CT shear/shear branches audited: {p['audited_pairs']}/{p['total_solver_pairs']} "
        f"({p['coverage']}); maximum computed disagreement: {p['max_independent_error']}.",
        f"BJ rank-one/cofactor audited: {certificate['P1_Ball_James']['audited_branches']}; "
        f"review: {certificate['P1_Ball_James']['review_required']}.",
        f"PTMC rank-one habits audited: {certificate['P1_PTMC']['audited_habits']}; "
        f"review: {certificate['P1_PTMC']['review_required']}.",
        "",
        certificate["thesis_claim_scope"],
        certificate["source_scope"],
        "",
    ))
