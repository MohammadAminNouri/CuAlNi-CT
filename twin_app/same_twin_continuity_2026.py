from __future__ import annotations

"""Follow ONE Cayron CT transformation twin through changing lattice metrics.

The identity is the exact rational parent order-two symmetry generator and
its Type-I mirror or Type-II twofold construction, NOT an operator array
position, sorted shear, or a closest independently chosen twin.  A matching
generator is necessary but not alone sufficient: the plane/direction/shear
must continue smoothly, both phase symmetries and the correspondence remain
unchanged, and collapsed or ambiguous branches fail closed.

Reference: Cayron, Crystals 12 (2022) 130, §§3–5; Acta Materialia 316
(2026) 122399, Eqs. 19–25, 32–43, 49.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import acos, degrees, isfinite
from typing import Any

import numpy as np
import sympy as sp

from cualni_cryst.ct import (
    analyze_austenite_martensite, ct_supercompatibility_residual,
    ips_shear_from_habit_plane,
)
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.theory_unified import exact_symmetry_group_for_ct
from cualni_cryst.twinning_ct import (
    CTTwin, twins_from_operator, type_i_from_parent_reflection,
    type_ii_from_parent_twofold,
)
from twin_app.metric_design_targets import metric_stretches
from twin_app.paper_methodology_2022_2026 import shear_shear_equation_42_audit


def _operation_key(op: Any, kind: str) -> str:
    a = sp.Matrix(op)
    if a.shape != (3, 3) or kind not in ("I", "II"):
        raise ValueError("Invalid symmetry generator or construction route")
    return kind + ":" + ",".join(str(sp.simplify(v)) for v in a)


def _projective_angle(lhs: Any, rhs: Any, metric: Any, reciprocal: bool) -> float:
    G = np.linalg.inv(np.asarray(metric, dtype=float)) if reciprocal else np.asarray(metric, dtype=float)
    L = np.linalg.cholesky(G)
    def unit(v):
        q = L.T @ np.asarray(v, dtype=float).reshape(3)
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < 1e-14:
            raise ValueError("Degenerate vector cannot identify a physical twin")
        return q / n
    a, b = unit(lhs), unit(rhs)
    return degrees(acos(float(np.clip(abs(a @ b), 0., 1.))))


def _project_fingerprint(project: Any, transformation_id: str) -> dict:
    t = project.transformation(transformation_id)
    pa,pm = project.phase(t.parent_phase_id), project.phase(t.product_phase_id)
    ma,mm = pa.lattice.metric(),pm.lattice.metric()
    tol = max(project.numerical_policy.representation,
              project.numerical_policy.algebraic)
    a = exact_symmetry_group_for_ct(
        list(pa.symmetry_matrices()),ma,tolerance=tol,label="parent")
    m = exact_symmetry_group_for_ct(
        list(pm.symmetry_matrices()),mm,tolerance=tol,label="product")
    signature=lambda items:tuple(sorted(
        ",".join(str(sp.simplify(x)) for x in op) for op in items))
    return {
        "parent_phase":t.parent_phase_id,
        "product_phase":t.product_phase_id,
        "parent_metric":np.asarray(ma,float),
        "correspondence":sp.Matrix(t.correspondence.C_M_from_A),
        "parent_group":signature(a),
        "product_group":signature(m),
        "parent_symmetry":a,
        "product_symmetry":m,
        "parent_metric_matrix":ma,
        "product_metric_matrix":mm,
        "correspondence_object":t.correspondence,
    }


@dataclass(frozen=True)
class TwinAnchor:
    branch_id: str
    generator_key: str
    generator: sp.Matrix
    construction_route: str
    initial_plane_parent: tuple[float, float, float]
    initial_direction_parent: tuple[float, float, float]
    initial_shear: float
    parent_metric: np.ndarray
    parent_group_signature: tuple[str, ...]
    product_group_signature: tuple[str, ...]
    parent_phase: str
    product_phase: str
    correspondence: sp.Matrix


def enumerate_twin_anchors(project: Any, transformation_id: str) -> tuple[TwinAnchor, ...]:
    """All CT twins with immutable generator identities; zero-shear omitted."""
    fp = _project_fingerprint(project, transformation_id)
    groupoid = correspondence_groupoid(
        fp["parent_symmetry"],fp["product_symmetry"],fp["correspondence_object"])
    rows=[]
    seen=set()
    for operator in groupoid.operators:
        for twin in twins_from_operator(
            operator,fp["parent_metric_matrix"],fp["product_metric_matrix"],
            fp["correspondence_object"],
        ):
            key = _operation_key(twin.parent_symmetry, twin.kind)
            if key in seen:
                raise AssertionError(
                    "Exact CT generator appeared twice across double-coset operators"
                )
            seen.add(key)
            rows.append(TwinAnchor(
                branch_id=sha256(key.encode()).hexdigest()[:18],
                generator_key=key,
                generator=sp.Matrix(twin.parent_symmetry),
                construction_route=twin.kind,
                initial_plane_parent=tuple(float(x) for x in twin.plane_a),
                initial_direction_parent=tuple(float(x) for x in twin.direction_a),
                initial_shear=float(twin.shear),
                parent_metric=fp["parent_metric"].copy(),
                parent_group_signature=fp["parent_group"],
                product_group_signature=fp["product_group"],
                parent_phase=fp["parent_phase"],
                product_phase=fp["product_phase"],
                correspondence=fp["correspondence"],
            ))
    return tuple(sorted(rows,key=lambda a:a.generator_key))


def _same_inputs(anchor: TwinAnchor, fp: dict) -> None:
    if (anchor.parent_phase != fp["parent_phase"] or
        anchor.product_phase != fp["product_phase"] or
        anchor.parent_group_signature != fp["parent_group"] or
        anchor.product_group_signature != fp["product_group"] or
        anchor.correspondence != fp["correspondence"] or
        not np.allclose(anchor.parent_metric,fp["parent_metric"],rtol=1e-12,atol=1e-13)):
        raise ValueError(
            "Parent metric, correspondence, phase identity or point group changed: "
            "the requested CT twin cannot be tracked as the same generic relation"
        )


def reconstruct_same_twin(
    project: Any, transformation_id: str, anchor: TwinAnchor,
    *, previous_twin: CTTwin | None = None,
    max_plane_step_deg: float = 25.0,
    max_direction_step_deg: float = 25.0,
    max_shear_ratio_step: float = 3.0,
) -> tuple[CTTwin,dict]:
    """Regenerate fixed rational symmetry generator; check physical continuity.

    Comparing a candidate's local integer twin_index or operator_index
    is explicitly prohibited. Shear collapse or a discontinuous geometry
    is NOT silently remapped to a different operator or variant.
    """
    fp=_project_fingerprint(project,transformation_id)
    _same_inputs(anchor,fp)
    if anchor.construction_route=="I":
        twin=type_i_from_parent_reflection(
            anchor.generator,fp["parent_metric_matrix"],
            fp["product_metric_matrix"],fp["correspondence_object"])
    elif anchor.construction_route=="II":
        twin=type_ii_from_parent_twofold(
            anchor.generator,fp["parent_metric_matrix"],
            fp["product_metric_matrix"],fp["correspondence_object"])
    else:
        raise ValueError("Unknown frozen CT construction")
    if _operation_key(twin.parent_symmetry,twin.kind)!=anchor.generator_key:
        raise AssertionError("Reconstructed CT generator identity changed")
    if not isfinite(twin.shear) or twin.shear<=1e-10:
        raise ValueError("CT M/M shear collapsed, twin identity is degenerate")
    base_plane=(previous_twin.plane_a if previous_twin is not None
                else anchor.initial_plane_parent)
    base_dir=(previous_twin.direction_a if previous_twin is not None
              else anchor.initial_direction_parent)
    base_s=(previous_twin.shear if previous_twin is not None
            else anchor.initial_shear)
    p_angle=_projective_angle(
        base_plane,twin.plane_a,fp["parent_metric"],True)
    e_angle=_projective_angle(
        base_dir,twin.direction_a,fp["parent_metric"],False)
    shear_ratio=max(base_s/twin.shear,twin.shear/base_s)
    if p_angle>max_plane_step_deg or e_angle>max_direction_step_deg or shear_ratio>max_shear_ratio_step:
        raise ValueError(
            "Same exact generator but discontinuous physical twin: "
            f"plane={p_angle:.5g}°, direction={e_angle:.5g}°, "
            f"shear ratio={shear_ratio:.5g}; reduce lattice steps"
        )
    return twin,{
        "branch_id":anchor.branch_id,
        "generator_key":anchor.generator_key,
        "kind":twin.kind,
        "classification":twin.classification,
        "parent_symmetry_generator":[[str(v) for v in row] for row in anchor.generator.tolist()],
        "rational_element":twin.rational_element,
        "plane_parent_reciprocal":[float(v) for v in twin.plane_a],
        "direction_parent_direct":[float(v) for v in twin.direction_a],
        "shear":float(twin.shear),
        "plane_step_deg":p_angle,
        "direction_step_deg":e_angle,
        "shear_ratio_step":shear_ratio,
        "status":"SAME_EXACT_GENERATOR_AND_CONTINUOUS_PHYSICAL_GEOMETRY",
    }


def fixed_twin_am_habits(
    project: Any, transformation_id: str,
    anchor: TwinAnchor, *,
    previous_twin: CTTwin | None = None,
    reference_habit: np.ndarray | None = None,
    metric_tolerance: float = 2e-8,
    max_habit_angle_deg: float = 30.,
) -> dict:
    """Return every exact A/M habit for the SAME M/M CT generator, no branch hopping."""
    t=project.transformation(transformation_id)
    ma=project.phase(t.parent_phase_id).lattice.metric()
    mm=project.phase(t.product_phase_id).lattice.metric()
    lambda2=metric_stretches(project,transformation_id)[1]
    if abs(lambda2-1.)>metric_tolerance:
        return {"status":"AM_METRIC_NOT_COMPATIBLE","lambda2":lambda2,
                "reason":"No exact A/M IPS; cannot evaluate Eq.42 for this state",
                "habits":[]}
    twin,trace=reconstruct_same_twin(
        project,transformation_id,anchor,previous_twin=previous_twin)
    am=analyze_austenite_martensite(
        ma,mm,t.correspondence,
        tol=min(metric_tolerance,project.numerical_policy.exact_eigenvalue*10))
    if not am.exact_habit_planes:
        return {"status":"NO_EXACT_CMC_HABIT",
                "lambda2":lambda2, "twin_trace":trace,"habits":[]}
    valid=[]
    for i,habit in enumerate(am.exact_habit_planes):
        m=np.asarray(habit,float)
        ang=(None if reference_habit is None else
             _projective_angle(m,reference_habit,ma,True))
        if ang is not None and ang>max_habit_angle_deg:
            continue
        d=ips_shear_from_habit_plane(m,ma,mm,t.correspondence)
        scalar=ct_supercompatibility_residual(
            m,d,twin.plane_a,twin.direction_a,twin.shear,ma)
        details=shear_shear_equation_42_audit(
            ma,m,d,twin.plane_a,twin.direction_a,twin.shear,
            tolerance=metric_tolerance)
        if not np.isclose(scalar,details["epsilon_normalized_by_twin_shear"],
                          rtol=2e-7,atol=5e-9):
            raise AssertionError("Independent Eq.42 vector and CT scalar disagree")
        valid.append({
            "habit_index_local":i,
            "habit_parent_reciprocal":m.tolist(),
            "reference_habit_angle_deg":ang,
            "epsilon":float(scalar),
            "eq42":details,
        })
    if reference_habit is not None:
        valid.sort(key=lambda r:r["reference_habit_angle_deg"])
        if len(valid)>1 and abs(
            valid[0]["reference_habit_angle_deg"]-
            valid[1]["reference_habit_angle_deg"])<1.e-4:
            return {"status":"AMBIGUOUS_HABIT_BRANCH",
                    "reason":"Two CMC habit normals equally close; cannot track one",
                    "lambda2":lambda2,"twin_trace":trace,"habits":[]}
    return {
        "status":"EXACT_CT_AM_HABITS_AND_FIXED_MM_TWIN",
        "lambda2":float(lambda2),
        "twin_trace":trace,
        "habits":valid,
        # None of the CT data alone certificates CC1/CC2/CC3 or an
        # experimentally observed M/M boundary.
        "ball_james_cofactor_on_this_twin":"NOT_YET_PHYSICALLY_ASSIGNED",
        "experimental_feasibility":"UNASSESSED",
    }
