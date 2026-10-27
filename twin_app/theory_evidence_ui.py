from __future__ import annotations

"""Read-only, branch-resolved audit evidence for crystallographic theory comparisons.

This module deliberately never alters CT, Ball--James, PTMC or their tolerances.
It independently reconstructs Ball--James rank-one and cofactor expressions
from already-calculated branch matrices. A numerical pass is not a theorem,
a measurement, or evidence that a twin will form experimentally.
"""

import math
import numpy as np

from cualni_cryst.theory_unified import PredictionKind


def _cof3(matrix):
    """Adjugate-transpose cofactor without inverse (also valid at singularity)."""
    q = np.asarray(matrix, float).reshape(3, 3)
    return np.column_stack((
        np.cross(q[:, 1], q[:, 2]),
        np.cross(q[:, 2], q[:, 0]),
        np.cross(q[:, 0], q[:, 1]),
    ))


def _finite(value):
    x = float(value)
    return x if math.isfinite(x) else None


def ball_james_branch_evidence(unified, *, tolerance=1.0e-7):
    """Re-evaluate every BJ twin in the SAME parent Cartesian frame as its U.

    The branch tensor is R U_other - U_base = a outer n. Cofactor quantities
    use U_base and this identical a,n, not CT's unrelated crystal covectors.
    Fail closed on malformed branches rather than silently skipping them.
    """
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError("Audit tolerance must be finite and positive")
    bj = unified.ball_james_report
    variants = {int(v.index): np.asarray(v.U, float) for v in bj.variants}
    records = []
    for twin in bj.martensite_twin_branches:
        i, j = int(twin.base_variant_index), int(twin.other_variant_index)
        label = f"U{i}/U{j}, branch {twin.branch}"
        if i not in variants or j not in variants:
            raise ValueError(f"{label}: variant referenced by branch does not exist")
        base, other = variants[i], variants[j]
        R = np.asarray(twin.rotation, float).reshape(3, 3)
        a = np.asarray(twin.a, float).reshape(3)
        n = np.asarray(twin.n_reference, float).reshape(3)
        if not all(np.isfinite(t).all() for t in (base, other, R, a, n)):
            raise ValueError(f"{label}: non-finite rank-one construction")
        jump = R @ other - base
        tensor = np.outer(a, n)
        residual = float(np.linalg.norm(jump - tensor) /
                         max(1.0, np.linalg.norm(jump), np.linalg.norm(tensor)))
        orth = float(np.linalg.norm(R.T @ R - np.eye(3)))
        determinant = float(np.linalg.det(R))
        U2 = base @ base
        cc1 = float(np.linalg.eigvalsh(base)[1] - 1.0)
        cc2 = float(a @ base @ _cof3(U2 - np.eye(3)) @ n)
        cc3 = float(np.trace(U2) - np.linalg.det(U2)
                    - 0.25 * float(a @ a) * float(n @ n) - 2.0)
        stored = twin.cofactor
        recorded = np.array([stored.cc1_lambda2_minus_one,
                             stored.cc2_value, stored.cc3_margin], float)
        recomputed = np.array([cc1, cc2, cc3], float)
        cofactor_reproduction = float(np.max(
            np.abs(recomputed-recorded) / np.maximum(1.0, np.maximum(np.abs(recomputed), np.abs(recorded)))))
        geometric_ok = (residual <= tolerance and orth <= tolerance
                        and abs(determinant - 1.0) <= tolerance)
        cofactor_reproduced = cofactor_reproduction <= tolerance
        # This is a cross-check of the reported cofactor numbers, NOT proof
        # that the underlying twin-system assumptions hold in CT's frame.
        records.append({
            "Pair / branch": label,
            "Rank-one residual": _finite(residual),
            "Rotation orthogonality residual": _finite(orth),
            "det(R)": _finite(determinant),
            "Independent CC1": _finite(cc1),
            "Independent CC2": _finite(cc2),
            "Independent CC3 margin": _finite(cc3),
            "Cofactor numerical reproduction gap": _finite(cofactor_reproduction),
            "Stored CC1–CC3 pass": bool(stored.all_satisfied),
            "Geometric jump verified": bool(geometric_ok),
            "CC quantities reproduced": bool(cofactor_reproduced),
            "Audit status": ("REPRODUCED" if geometric_ok and cofactor_reproduced
                              else "REVIEW_REQUIRED"),
        })
    return records


def ct_supercompatibility_evidence(unified):
    """Every CT habit/twin combination, retaining the operator and branch ID."""
    records = []
    for row in unified.rows:
        if row.prediction_kind is not PredictionKind.CT_SUPERCOMPATIBILITY:
            continue
        epsilon = row.residuals.get("ct_supercompatibility_dimensionless")
        if epsilon is None or not math.isfinite(float(epsilon)):
            raise ValueError("CT shear/shear record is missing a finite residual")
        records.append({
            "CT branch": row.branch_label,
            "Twin classification": row.metadata.get("twin_classification", "unclassified"),
            "Operator index": row.metadata.get("operator_index"),
            "Twin index": row.metadata.get("twin_index"),
            "Habit index": row.metadata.get("habit_index"),
            "Habit covector in parent crystal": row.habit_plane_parent_crystal,
            "Twin plane in parent crystal": row.twin_plane_parent_crystal,
            "Twin direction in parent crystal": row.twin_direction_parent_crystal,
            "Twin shear magnitude": row.shear_magnitude,
            "Dimensionless epsilon": float(epsilon),
            "Interpretation": (
                "exact at zero within documented numerical tolerance"
                if epsilon == 0 else "nonzero geometric shear/shear mismatch"
            ),
        })
    return sorted(records, key=lambda r: r["Dimensionless epsilon"])


def assigned_interface_evidence(unified, equivalence):
    """Physical one-to-one assignments with missing fields kept explicitly null."""
    by_id = {r.row_id: r for r in unified.rows}
    output = []
    for m in equivalence.matches:
        left = by_id.get(m.left_row_id)
        right = by_id.get(m.right_row_id)
        if left is None or right is None:
            raise ValueError("Branch matcher references an unknown prediction row")
        output.append({
            "Family": m.family.value,
            "Theory A": m.left_theory.value,
            "Theory B": m.right_theory.value,
            "A row ID": m.left_row_id,
            "B row ID": m.right_row_id,
            "A provenance": left.provenance,
            "B provenance": right.provenance,
            "A physical role": left.prediction_kind.value,
            "B physical role": right.prediction_kind.value,
            "Observable residuals": m.residuals.to_dict(),
            "Required components": list(m.required_components),
            "Missing required components": list(m.missing_required_components),
            "Within tolerance by observable": dict(m.within_tolerance),
            "Agreement": (
                "COMPLETE_WITHIN_TOLERANCE"
                if m.all_required_components_within_tolerance is True
                else "DISAGREES" if m.all_required_components_within_tolerance is False
                else "INCOMPLETE_UNPROVEN"
            ),
        })
    return output


def evidence_report(unified, equivalence):
    """Machine-readable, input-specific audit downstream from all solvers."""
    bj = ball_james_branch_evidence(unified)
    ct = ct_supercompatibility_evidence(unified)
    matches = assigned_interface_evidence(unified, equivalence)
    return {
        "audit_contract": {
            "scope": "calculated whole transformation; no EBSD, stress or kinetics",
            "tolerance": 1.0e-7,
            "BJ geometry": "R U_other - U_base = a outer n, parent Cartesian",
            "CC1": "lambda2(U_base)-1",
            "CC2": "a dot U_base cof(U_base^2-I) n",
            "CC3": "tr(U_base^2)-det(U_base^2)-|a|^2|n|^2/4-2",
            "CT epsilon": "||2 (m_A dot n) d_A - a_twin||_M_A / |s|",
            "limits": [
                "CT exactness uses separately declared backend tolerances",
                "matching is observable-specific, not proof of theory equivalence",
                "cofactor identities require a genuinely admissible twin system",
                "sampled laminate volume fractions are diagnostic, not an all-f proof",
                "neither numerical geometry nor literature fit predicts actual variant selection",
                "published NiTi analytical parameter families do not apply to Cu-Al-Ni by substitution",
            ],
        },
        "BJ independent branch tensor checks": bj,
        "CT exact habit / twin shear-shear pairs": ct,
        "Assigned physical interface comparisons": matches,
        "BJ branches requiring review": sum(r["Audit status"] != "REPRODUCED" for r in bj),
        "Incomplete cross-theory assignments": sum(
            r["Agreement"] == "INCOMPLETE_UNPROVEN" for r in matches),
    }


def render_theory_evidence(unified, equivalence):
    """Evidence-first presentation, without making new physical predictions."""
    import pandas as pd
    import streamlit as st

    data = evidence_report(unified, equivalence)
    st.markdown("### Branch-resolved scientific verification")
    st.caption(
        "Recomputed from the original solver outputs, not fitted to Cayron's "
        "published values. CT A/M shear/shear pairs, BJ rank-one jumps and "
        "physical cross-theory assignments are separate tests."
    )
    if data["BJ branches requiring review"]:
        st.error(
            f'{data["BJ branches requiring review"]} Ball–James twin records did not '
            "pass independent reconstruction at the declared audit tolerance. "
            "Do not use them as thesis-validated predictions."
        )
    with st.expander("1. CT: every exact A/M habit × transformation-twin operator", expanded=False):
        rows = data["CT exact habit / twin shear-shear pairs"]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No exact CMC habit/twin pair was available. Supercompatibility was not evaluated.")
        st.caption(
            "Epsilon is a dimensionless geometric residual, NOT a percentage "
            "error, formation probability or thermal hysteresis."
        )
    with st.expander("2. Ball–James: independently reconstructed rank-one and CC equations", expanded=False):
        rows = data["BJ independent branch tensor checks"]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No BJ M/M twin branches: no cofactor conclusion is possible.")
        st.caption(
            "CC1–CC3 are conditional on the verified rank-one twin construction. "
            "A sampled 101-point f grid never proves compatibility for all f."
        )
    with st.expander("3. CT ↔ BJ ↔ PTMC: complete physical branch assignments", expanded=False):
        rows = data["Assigned physical interface comparisons"]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No assignable common physical observables. An empty assignment is not disagreement.")
        st.caption(
            "Incomplete means not proved. A/M single-variant planes and PTMC "
            "two-variant laminate habits are not interchangeable predictions."
        )
    with st.expander("4. Cayron 2026 source-validation boundaries", expanded=False):
        st.warning(
            "Acta Materialia 316 (2026) 122399, §5.3: the printed O2 Type-I "
            "family b=c=√2, a=c/sin(beta) gives d_z=1/2 and "
            "||U^(-1)e_001||²=1/2 under the paper's Eq. 41 and stated "
            "correspondence, not the required 0 and 1. "
            "The project-derived a=1/sin(beta) closes those equalities, "
            "but is NOT a confirmed author correction. Do not use the "
            "printed family as a verified design target."
        )
        st.markdown(
            "The paper's C1, C2a, C2b, C3 regions and O2/O4 NiTi "
            "examples are material- and correspondence-specific. "
            "They are literature benchmarks, not generic Cu–Al–Ni equations."
        )
    from twin_app.cayron_2026_reference import render_cayron_2026_reference
    render_cayron_2026_reference()
    return data
