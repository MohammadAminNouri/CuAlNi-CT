from __future__ import annotations

"""Input-driven twin families and their exact A/M laminate habit planes.

This module is an orchestrator, not a replacement crystallography solver.
Every numerical result comes from the active ProjectState and the repository's
existing exact-group, metric, nonlinear-elasticity and PTMC engines.

No material name, alloy-specific answer or published numerical output is used
at runtime. Literature values belong only in external validation tests.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from cualni_cryst.ball_james_adapter import BallJamesAdapter, MartensiteTwinBranch
from cualni_cryst.correspondence import Correspondence
from cualni_cryst.group_theory import correspondence_groupoid, representatives
from cualni_cryst.lattice import metric_sqrt
from cualni_cryst.project_io import project_from_dict
from cualni_cryst.ptmc_adapter import PTMCAdapter, PTMCHabitSolution, PTMCTwinRelation
from cualni_cryst.stretch import stretch_from_metrics
from cualni_cryst.symmetry import matrix_key
from cualni_cryst.twinning_ct import CTTwin, twins_from_operator
from cualni_cryst.weak_operator_engine import (
    analyze_higher_order_element,
    audit_operator_route,
)
from cualni_cryst.weak_twins import BravaisNodeBasis

from .scientific_certifier import certify_ptmc_twinning

from .scientific_models import (
    ClassicalTwinSystem,
    CorrespondenceVariantRecord,
    HabitPlaneSolution,
    Matrix3Text,
    PairTwinConstruction,
    ScientificAudit,
    TwinElementRepresentation,
    TwinFamilyRecord,
    TwinFamilyReport,
    VariantPairRecord,
    Vector3,
    WeakPlaneCandidate,
)


def _vector3(value: Any) -> Vector3:
    arr = np.asarray(value, dtype=float).reshape(3)
    return tuple(float(x) for x in arr)  # type: ignore[return-value]


def _matrix_text(matrix: Any) -> Matrix3Text:
    M = sp.Matrix(matrix)
    return tuple(
        tuple(str(sp.simplify(M[i, j])) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def _matrix_float_tuple(matrix: Any) -> tuple[tuple[float, float, float], ...]:
    M = np.asarray(matrix, dtype=float).reshape(3, 3)
    return tuple(tuple(float(x) for x in row) for row in M)


def _relative_tensor_residual(a1: Any, n1: Any, a2: Any, n2: Any) -> float:
    left = np.outer(np.asarray(a1, dtype=float), np.asarray(n1, dtype=float))
    right = np.outer(np.asarray(a2, dtype=float), np.asarray(n2, dtype=float))
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0)
    return float(np.linalg.norm(left - right) / scale)


def _metric_projective_angle_deg(
    lhs: Any,
    rhs: Any,
    metric: np.ndarray,
    *,
    reciprocal: bool,
) -> float:
    u = np.asarray(lhs, dtype=float).reshape(3)
    v = np.asarray(rhs, dtype=float).reshape(3)
    M = np.asarray(metric, dtype=float).reshape(3, 3)
    G = np.linalg.solve(M, np.eye(3)) if reciprocal else M
    nu = float(np.sqrt(u @ G @ u))
    nv = float(np.sqrt(v @ G @ v))
    if min(nu, nv) <= 1.0e-15:
        raise ValueError("Cannot compare a zero crystallographic line/plane")
    cosine = abs(float(u @ G @ v)) / (nu * nv)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _exact_group_from_phase(phase: Any, *, tolerance: float) -> list[sp.Matrix]:
    """Recover exact crystallographic matrices from validated PhaseState data."""

    output: list[sp.Matrix] = []
    for index, operation in enumerate(phase.symmetry_matrices()):
        numeric = np.asarray(operation, dtype=float).reshape(3, 3)
        exact = sp.Matrix(
            [
                [
                    sp.nsimplify(
                        float(numeric[i, j]),
                        tolerance=tolerance,
                        rational=True,
                    )
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )
        reconstructed = np.asarray(exact, dtype=float)
        residual = float(
            np.linalg.norm(reconstructed - numeric)
            / max(float(np.linalg.norm(numeric)), 1.0)
        )
        if residual > tolerance:
            raise ValueError(
                f"Could not recover exact symmetry operator {index} for "
                f"{phase.phase_id!r}; residual={residual:.3e}"
            )
        determinant = sp.simplify(exact.det())
        if determinant not in {sp.Integer(-1), sp.Integer(1)}:
            raise ValueError(
                f"Symmetry operator {index} has exact determinant {determinant}; expected ±1"
            )
        output.append(exact)
    return output


def _deterministic_coset_representative(coset: Sequence[sp.Matrix]) -> tuple[sp.Matrix, bool]:
    """Prefer an orientation-preserving representative when the coset has one."""

    ordered = sorted(
        (sp.Matrix(g) for g in coset),
        key=lambda g: tuple(str(sp.simplify(x)) for x in list(g)),
    )
    proper = [g for g in ordered if sp.simplify(g.det()) == 1]
    if proper:
        return proper[0], True
    return ordered[0], False


def _nearest_stretch_variant(U: np.ndarray, ptmc_variants: Sequence[Any]) -> tuple[int | None, float | None, str]:
    if not ptmc_variants:
        return None, None, "no PTMC stretch variants"
    residuals = np.asarray(
        [np.linalg.norm(np.asarray(item.U, dtype=float) - U, ord="fro") for item in ptmc_variants],
        dtype=float,
    )
    index = int(np.argmin(residuals))
    return int(ptmc_variants[index].index), float(residuals[index]), "candidate"


def _build_correspondence_variants(
    *,
    groupoid: Any,
    base_correspondence: Correspondence,
    parent_metric: np.ndarray,
    product_metric: np.ndarray,
    ptmc_variants: Sequence[Any],
    mapping_tolerance: float,
) -> tuple[tuple[CorrespondenceVariantRecord, ...], dict[int, int | None], float]:
    records: list[CorrespondenceVariantRecord] = []
    mapping: dict[int, int | None] = {}
    maximum = 0.0

    for index, coset in enumerate(groupoid.variants):
        representative, orientation_preserving = _deterministic_coset_representative(coset)
        C_i_exact = sp.simplify(base_correspondence.C_M_from_A * representative.inv())
        C_i = Correspondence(C_i_exact, label=f"M{index + 1}")

        stretch_index: int | None = None
        residual: float | None = None
        status: str
        if not orientation_preserving:
            status = "no orientation-preserving representative in this correspondence coset"
        else:
            U_i = stretch_from_metrics(parent_metric, product_metric, C_i)
            candidate, residual, _ = _nearest_stretch_variant(U_i, ptmc_variants)
            if candidate is None or residual is None:
                status = "no stretch mapping available"
            elif residual <= mapping_tolerance:
                stretch_index = candidate
                status = "mapped"
                maximum = max(maximum, residual)
            else:
                status = (
                    "nearest stretch variant exceeds mapping tolerance: "
                    f"{residual:.3e} > {mapping_tolerance:.3e}"
                )

        mapping[index] = stretch_index
        records.append(
            CorrespondenceVariantRecord(
                variant_index=index + 1,
                representative_parent_symmetry=_matrix_text(representative),
                representative_determinant=int(sp.simplify(representative.det())),
                correspondence_M_from_A=_matrix_text(C_i_exact),
                stretch_variant_index=None if stretch_index is None else stretch_index + 1,
                stretch_mapping_residual=residual,
                mapping_status=status,
            )
        )

    return tuple(records), mapping, float(maximum)


def _same_ct_geometry(lhs: CTTwin, rhs: CTTwin, M_a: np.ndarray, M_m: np.ndarray) -> bool:
    """Deduplicate equivalent descriptions *within one CT construction route*.

    Type-I fields expose K1/eta1 while Type-II fields expose conjugate K2/eta2.
    Comparing those fields directly across routes would mix different physical
    observables. Compound classification is therefore resolved later, for one
    pair-specific rank-one branch, by independent Type-I geometry and Type-II
    generator-provenance locks.
    """

    if str(lhs.kind) != str(rhs.kind):
        return False
    shear_scale = max(abs(float(lhs.shear)), abs(float(rhs.shear)), 1.0)
    if abs(float(lhs.shear) - float(rhs.shear)) > 1.0e-8 * shear_scale:
        return False
    angles = (
        _metric_projective_angle_deg(lhs.plane_a, rhs.plane_a, M_a, reciprocal=True),
        _metric_projective_angle_deg(lhs.direction_a, rhs.direction_a, M_a, reciprocal=False),
        _metric_projective_angle_deg(lhs.plane_m, rhs.plane_m, M_m, reciprocal=True),
        _metric_projective_angle_deg(lhs.direction_m, rhs.direction_m, M_m, reciprocal=False),
    )
    return max(angles) <= 1.0e-6


def _representation(twin: CTTwin) -> TwinElementRepresentation:
    route = str(twin.kind)
    if route == "I":
        plane_symbol, direction_symbol = "K1", "η1"
    elif route == "II":
        plane_symbol, direction_symbol = "K2", "η2"
    else:  # guarded by the backend dataclass, kept defensive here
        raise ValueError(f"Unsupported classical twin route {route!r}")
    return TwinElementRepresentation(
        route=route,  # type: ignore[arg-type]
        plane_symbol=plane_symbol,
        direction_symbol=direction_symbol,
        plane_parent_crystal=_vector3(twin.plane_a),
        direction_parent_crystal=_vector3(twin.direction_a),
        plane_product_crystal=_vector3(twin.plane_m),
        direction_product_crystal=_vector3(twin.direction_m),
        shear_magnitude=float(twin.shear),
        generator_parent_symmetry=_matrix_text(twin.parent_symmetry),
        intercorrespondence_product=_matrix_float_tuple(twin.intercorrespondence),
    )


def _deduplicate_classical_systems(
    twins: Sequence[CTTwin],
    M_a: np.ndarray,
    M_m: np.ndarray,
) -> tuple[ClassicalTwinSystem, ...]:
    groups: list[list[CTTwin]] = []
    for twin in twins:
        for group in groups:
            if _same_ct_geometry(twin, group[0], M_a, M_m):
                group.append(twin)
                break
        else:
            groups.append([twin])

    systems: list[ClassicalTwinSystem] = []
    for number, group in enumerate(groups, start=1):
        routes = {item.kind for item in group}
        if routes == {"I"}:
            classification = "Type I"
        elif routes == {"II"}:
            classification = "Type II"
        else:
            # Cross-route grouping is forbidden because Type-I stores K1/eta1
            # whereas Type-II stores the conjugate K2/eta2.
            raise AssertionError(f"Unexpected mixed classical route set {routes!r}")

        representations: list[TwinElementRepresentation] = []
        seen_generators: set[tuple[str, ...]] = set()
        for item in sorted(group, key=lambda x: (x.kind, _matrix_text(x.parent_symmetry))):
            key = tuple(str(v) for row in _matrix_text(item.parent_symmetry) for v in row) + (item.kind,)
            if key in seen_generators:
                continue
            seen_generators.add(key)
            representations.append(_representation(item))

        systems.append(
            ClassicalTwinSystem(
                system_id=f"S{number}",
                classification=classification,  # type: ignore[arg-type]
                representations=tuple(representations),
                shear_magnitude=float(np.mean([item.shear for item in group])),
                provenance_count=len(group),
            )
        )
    return tuple(systems)


def _index_bj_twins(bj: Any) -> dict[tuple[int, int], list[MartensiteTwinBranch]]:
    result: dict[tuple[int, int], list[MartensiteTwinBranch]] = defaultdict(list)
    for item in bj.martensite_twin_branches:
        result[(int(item.base_variant_index), int(item.other_variant_index))].append(item)
    return result


def _match_bj_branch(
    relation: PTMCTwinRelation,
    candidates: Sequence[MartensiteTwinBranch],
    *,
    tolerance: float,
) -> tuple[MartensiteTwinBranch | None, float | None, float | None]:
    if not candidates:
        return None, None, None
    scored = [
        (
            _relative_tensor_residual(relation.a, relation.n_reference, item.a, item.n_reference),
            item,
        )
        for item in candidates
    ]
    outer_residual, selected = min(scored, key=lambda pair: pair[0])
    shear_residual = abs(
        float(relation.twin_shear_magnitude) - float(selected.shear_magnitude)
    ) / max(
        abs(float(relation.twin_shear_magnitude)),
        abs(float(selected.shear_magnitude)),
        1.0e-15,
    )
    if max(float(outer_residual), float(shear_residual)) > 100.0 * tolerance:
        return None, float(outer_residual), float(shear_residual)
    return selected, float(outer_residual), float(shear_residual)


def _system_type_i_geometry_residual(
    system: ClassicalTwinSystem,
    pair_plane_product: Any,
    pair_direction_product: Any,
    product_metric: np.ndarray,
) -> tuple[float, float] | None:
    """Compare a pair's physical K1/eta1 with an independent Type-I description.

    Type-I crystallographic output and the pair-specific rank-one relation both
    expose the physical twinning plane/shear line directly.  Type-II's K2/eta2
    are conjugate elements, so they are deliberately *not* compared to K1/eta1.
    """

    residuals: list[tuple[float, float]] = []
    for representation in system.representations:
        if representation.route != "I":
            continue
        plane = _metric_projective_angle_deg(
            representation.plane_product_crystal,
            pair_plane_product,
            product_metric,
            reciprocal=True,
        )
        direction = _metric_projective_angle_deg(
            representation.direction_product_crystal,
            pair_direction_product,
            product_metric,
            reciprocal=False,
        )
        residuals.append((plane, direction))
    if not residuals:
        return None
    return min(residuals, key=lambda item: max(item))


def _system_has_type_ii_generator_match(
    system: ClassicalTwinSystem,
    bj_branch: MartensiteTwinBranch,
    parent_group: Sequence[sp.Matrix],
) -> bool:
    """Cross-lock Type-II provenance through the exact parent twofold matrix."""

    indices = {
        int(match.parent_symmetry_index)
        for match in bj_branch.mallard_matches
        if str(match.kind).strip().upper() == "II"
    }
    if not indices:
        return False
    target_keys = {
        matrix_key(parent_group[index])
        for index in indices
        if 0 <= index < len(parent_group)
    }
    for representation in system.representations:
        if representation.route != "II":
            continue
        generator = sp.Matrix(representation.generator_parent_symmetry)
        if matrix_key(generator) in target_keys:
            return True
    return False


def _system_has_type_i_mirror_provenance_match(
    system: ClassicalTwinSystem,
    bj_branch: MartensiteTwinBranch,
    parent_group: Sequence[sp.Matrix],
) -> bool:
    """Independent centrosymmetric Type-I provenance (Mallard I ↔ mirror -Q).

    A proper parent twofold Q has eigenvalues (+1,-1,-1); its negative -Q
    is the reflection in the plane normal to Q's axis. Only if this exact
    reflection is present in the CT Type-I representations of the *same*
    operator family may it serve as independent Type-I provenance. This
    avoids comparing physical K1 in the base variant's product frame with
    a different symmetry-equivalent product variant's coordinate frame.

    This is unavailable for noncentrosymmetric parent phases where -Q is not
    necessarily a crystallographic symmetry; the original direct geometry
    cross-lock is retained for those cases.
    """
    mirror_keys: set[tuple[Any, ...]] = set()
    for match in bj_branch.mallard_matches:
        if str(match.kind).strip().upper() != "I":
            continue
        index = int(match.parent_symmetry_index)
        if index < 0 or index >= len(parent_group):
            continue
        Q = sp.Matrix(parent_group[index])
        if Q.det() != 1 or sp.trace(Q) != -1 or Q * Q != sp.eye(3):
            continue
        mirror_keys.add(matrix_key(-Q))
    if not mirror_keys:
        return False
    return any(
        rep.route == "I" and matrix_key(sp.Matrix(rep.generator_parent_symmetry)) in mirror_keys
        for rep in system.representations
    )


def _pair_classification(
    bj_branch: MartensiteTwinBranch | None,
    classical_systems: Sequence[ClassicalTwinSystem],
    *,
    family_has_classical_route: bool,
    shear_magnitude: float,
    shear_tolerance: float,
    pair_plane_product: Any,
    pair_direction_product: Any,
    product_metric: np.ndarray,
    geometry_tolerance_deg: float,
    parent_group: Sequence[sp.Matrix],
) -> tuple[
    str,
    tuple[str, ...],
    str,
    tuple[str, ...],
    float | None,
    float | None,
    float | None,
]:
    """Cross-lock one exact pair branch without inferring labels from shear alone.

    The pair-specific nonlinear-elasticity route and the discrete symmetry route
    remain independent. Type I is cross-locked by physical K1/eta1 geometry plus
    shear. Type II is cross-locked by exact parent-twofold provenance plus shear,
    because the discrete Type-II construction exposes conjugate K2/eta2 rather
    than physical K1/eta1. Compound requires *both* locks on the same rank-one
    branch; K2/eta2 are never compared directly with K1/eta1.
    """

    if not family_has_classical_route:
        return (
            "Exact rank-one relation — operator family is not classical",
            (),
            "rank-one relation exists, but the complete operator class has no exact mirror/twofold route",
            (),
            None,
            None,
            None,
        )

    proposed: str | None = None
    representations: tuple[str, ...] = ()
    if bj_branch is not None:
        kinds = tuple(
            sorted({str(match.kind).strip().upper() for match in bj_branch.mallard_matches})
        )
        if bool(bj_branch.compound_by_multiple_twofolds) or {"I", "II"}.issubset(kinds):
            proposed, representations = "Compound", ("I", "II")
        elif "I" in kinds:
            proposed, representations = "Type I", ("I",)
        elif "II" in kinds:
            proposed, representations = "Type II", ("II",)

    # Keep the two discrete routes separate.  A Type-I match is a physical
    # K1/eta1 comparison; a Type-II match is exact generator provenance.
    type_i_candidates: list[tuple[ClassicalTwinSystem, float, float, float]] = []
    type_ii_candidates: list[tuple[ClassicalTwinSystem, float]] = []

    for system in classical_systems:
        shear_residual = abs(float(system.shear_magnitude) - float(shear_magnitude)) / max(
            abs(float(system.shear_magnitude)), abs(float(shear_magnitude)), 1.0e-15
        )
        if shear_residual > shear_tolerance:
            continue

        if any(rep.route == "I" for rep in system.representations):
            geometry = _system_type_i_geometry_residual(
                system,
                pair_plane_product,
                pair_direction_product,
                product_metric,
            )
            geometric_lock = geometry is not None and max(geometry) <= geometry_tolerance_deg
            # Exact -Q mirror provenance is a second scientifically valid
            # route when the product reference frames differ by parent symmetry.
            # It still requires a matching BJ Mallard Type-I branch AND shear.
            mirror_lock = (
                bj_branch is not None
                and _system_has_type_i_mirror_provenance_match(
                    system, bj_branch, parent_group
                )
            )
            if geometric_lock or mirror_lock:
                type_i_candidates.append((
                    system,
                    float(shear_residual),
                    float(geometry[0]) if geometric_lock else None,
                    float(geometry[1]) if geometric_lock else None,
                ))

        if (
            bj_branch is not None
            and any(rep.route == "II" for rep in system.representations)
            and _system_has_type_ii_generator_match(system, bj_branch, parent_group)
        ):
            type_ii_candidates.append((system, float(shear_residual)))

    type_i_candidates.sort(key=lambda item: (
        item[1],
        item[2] if item[2] is not None else float("inf"),
        item[3] if item[3] is not None else float("inf"),
        item[0].system_id,
    ))
    type_ii_candidates.sort(key=lambda item: (item[1], item[0].system_id))

    if proposed == "Type I" and type_i_candidates:
        best = type_i_candidates[0]
        return (
            "Type I",
            ("I",),
            "cross-locked: independent rank-one and discrete routes agree by K1/eta1 geometry + shear OR exact (-Q) Type-I mirror provenance + shear",
            tuple(item[0].system_id for item in type_i_candidates),
            best[2],
            best[3],
            best[1],
        )

    if proposed == "Type II" and type_ii_candidates:
        best = type_ii_candidates[0]
        return (
            "Type II",
            ("II",),
            "cross-locked: independent rank-one and discrete symmetry routes agree by exact parent twofold provenance + shear",
            tuple(item[0].system_id for item in type_ii_candidates),
            None,
            None,
            best[1],
        )

    if proposed == "Compound" and type_i_candidates and type_ii_candidates:
        best_i = type_i_candidates[0]
        best_ii = type_ii_candidates[0]
        matched_ids = tuple(
            dict.fromkeys(
                [item[0].system_id for item in type_i_candidates]
                + [item[0].system_id for item in type_ii_candidates]
            )
        )
        return (
            "Compound",
            ("I", "II"),
            "cross-locked: same rank-one branch has Type-I K1/eta1 geometry + shear (or exact -Q mirror provenance + shear) and independent Type-II exact parent twofold provenance + shear",
            matched_ids,
            best_i[2],
            best_i[3],
            max(best_i[1], best_ii[1]),
        )

    if proposed is None and type_i_candidates:
        # A mirror-only parent family can support a Type-I construction even
        # when the independent Mallard provenance has no parent twofold index.
        # Do not infer Type II or Compound without their independent provenance.
        best = type_i_candidates[0]
        return (
            "Type I",
            ("I",),
            "cross-locked: discrete Type-I K1/eta1 geometry + shear; no conflicting nonlinear-elasticity classification provenance",
            tuple(item[0].system_id for item in type_i_candidates),
            best[2],
            best[3],
            best[1],
        )

    if bj_branch is None:
        reason = "no independent twofold provenance and no Type-I geometry match"
    elif proposed is None:
        reason = "independent nonlinear-elasticity branch has no Type-I/Type-II provenance match"
    elif proposed == "Compound":
        missing = []
        if not type_i_candidates:
            missing.append("Type-I K1/eta1 geometry or exact (-Q) mirror provenance")
        if not type_ii_candidates:
            missing.append("Type-II exact twofold provenance")
        reason = "independent route proposed Compound, but " + " and ".join(missing) + " did not cross-lock"
    else:
        reason = f"independent route proposed {proposed}, but discrete geometry/provenance did not cross-lock"
    return (
        "Exact rank-one relation — classification cross-lock unresolved",
        representations,
        reason,
        (),
        None,
        None,
        None,
    )

def _habit_solution(solution: PTMCHabitSolution, parent_metric: np.ndarray) -> HabitPlaneSolution:
    if solution.other_variant_volume_fraction is None or solution.base_variant_volume_fraction is None:
        raise AssertionError(
            "A twinning habit solution must carry both laminate volume fractions"
        )

    other_fraction = float(solution.other_variant_volume_fraction)
    base_fraction = float(solution.base_variant_volume_fraction)
    if not (np.isfinite(other_fraction) and np.isfinite(base_fraction)):
        raise AssertionError("Habit-plane laminate fractions must be finite")
    fraction_tol = 1.0e-9
    if not (-fraction_tol <= other_fraction <= 1.0 + fraction_tol):
        raise AssertionError(
            f"Other-variant laminate fraction is outside [0,1]: {other_fraction:.16g}"
        )
    if not (-fraction_tol <= base_fraction <= 1.0 + fraction_tol):
        raise AssertionError(
            f"Base-variant laminate fraction is outside [0,1]: {base_fraction:.16g}"
        )
    if abs((other_fraction + base_fraction) - 1.0) > fraction_tol:
        raise AssertionError(
            "Habit-plane laminate fractions do not sum to one: "
            f"base={base_fraction:.16g}, other={other_fraction:.16g}"
        )

    B_a = metric_sqrt(parent_metric)
    habit_normal = np.asarray(solution.habit_normal_parent_cartesian, dtype=float).reshape(3)
    habit_plane = np.asarray(solution.habit_plane_parent_crystal, dtype=float).reshape(3)
    rank_one_vector = np.asarray(solution.rank_one_vector, dtype=float).reshape(3)
    b_parent_crystal = np.linalg.solve(B_a, rank_one_vector)

    # In the symmetric metric frame, a parent reciprocal covector is p = B_A^T n
    # and a direct vector satisfies b_cart = B_A b_crystal.  Cross-check both
    # conversions before exposing the habit-plane coordinates in the UI.
    reconstructed_plane = B_a.T @ habit_normal
    plane_norm = float(np.linalg.norm(habit_plane))
    reconstructed_plane_norm = float(np.linalg.norm(reconstructed_plane))
    if min(plane_norm, reconstructed_plane_norm) <= 1.0e-15:
        raise AssertionError("Habit-plane coordinate conversion produced a zero covector")
    plane_unit = habit_plane / plane_norm
    reconstructed_plane_unit = reconstructed_plane / reconstructed_plane_norm
    plane_residual = min(
        float(np.linalg.norm(plane_unit - reconstructed_plane_unit)),
        float(np.linalg.norm(plane_unit + reconstructed_plane_unit)),
    )
    reconstructed_b = B_a @ b_parent_crystal
    b_scale = max(float(np.linalg.norm(rank_one_vector)), float(np.linalg.norm(reconstructed_b)), 1.0e-15)
    b_residual = float(np.linalg.norm(rank_one_vector - reconstructed_b) / b_scale)

    frame_tol = 1.0e-9
    if plane_residual > frame_tol:
        raise AssertionError(
            "Habit-plane parent reciprocal-coordinate conversion failed: "
            f"projective residual={plane_residual:.3e}"
        )
    if b_residual > frame_tol:
        raise AssertionError(
            "Habit-plane shape-vector parent-coordinate conversion failed: "
            f"residual={b_residual:.3e}"
        )

    rotation_residual = max(
        float(solution.rotation_orthogonality_residual),
        abs(float(solution.rotation_determinant) - 1.0),
    )
    return HabitPlaneSolution(
        habit_branch=int(solution.habit_branch),
        other_variant_volume_fraction=other_fraction,
        base_variant_volume_fraction=base_fraction,
        habit_plane_parent_crystal=_vector3(habit_plane),
        habit_normal_parent_cartesian=_vector3(habit_normal),
        shape_vector_parent_cartesian=_vector3(rank_one_vector),
        shape_vector_parent_crystal=_vector3(b_parent_crystal),
        habit_plane_product_crystal_base=_vector3(solution.habit_plane_product_crystal_base),
        frame_plane_residual=float(plane_residual),
        frame_shape_vector_residual=float(b_residual),
        rank_one_residual=float(solution.rank_one_residual),
        rotation_residual=float(rotation_residual),
        middle_stretch_residual=float(solution.middle_stretch_residual),
    )


def _canonical_projective_vector(value: Any) -> Vector3:
    vector = np.asarray(value, dtype=float).reshape(3)
    scale = float(np.max(np.abs(vector)))
    if scale <= 1.0e-15:
        raise ValueError("Cannot canonicalize a zero crystallographic vector")
    vector = vector / scale
    nonzero = np.flatnonzero(np.abs(vector) > 1.0e-12)
    if len(nonzero) and vector[nonzero[0]] < 0.0:
        vector = -vector
    vector[np.abs(vector) < 1.0e-13] = 0.0
    return _vector3(vector)


def _pair_shear_direction_product_crystal(
    relation: PTMCTwinRelation,
    base_variant: Any,
    product_metric: np.ndarray,
) -> Vector3:
    """Return the physical shear line eta1 in the base-product crystal basis.

    From R U_other - U_base = a ⊗ n, right-multiplication by U_base^-1
    gives the relative shear I + a ⊗ (U_base^-T n).  Hence ``a`` is the
    current/deformed shear direction.  The base variant's polar rotation carries
    it into the product physical frame, and M_M^(1/2) converts that physical
    vector to product crystallographic direct coordinates.
    """

    a = np.asarray(relation.a, dtype=float).reshape(3)
    norm = float(np.linalg.norm(a))
    if norm <= 1.0e-15:
        raise ValueError("Exact twin relation has a zero shear vector")
    physical_product = (
        np.asarray(base_variant.polar_rotation_product_from_parent, dtype=float)
        @ (a / norm)
    )
    B_m = metric_sqrt(product_metric)
    direct_product = np.linalg.solve(B_m, physical_product)
    return _canonical_projective_vector(direct_product)


def _family_keys(groupoid: Any) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    """Group unoriented M_i--M_j pairs by an operator and its inverse class."""

    groups: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    n = len(groupoid.variants)
    for i in range(n):
        for j in range(i + 1, n):
            forward = int(groupoid.adjacency[i][j])
            reverse = int(groupoid.adjacency[j][i])
            key = tuple(sorted({forward, reverse}))
            groups[key].append((i, j))
    return groups


def _representative_pair(pairs: Sequence[tuple[int, int]]) -> tuple[int, int]:
    with_first = [pair for pair in pairs if pair[0] == 0]
    return min(with_first or list(pairs))


def _weak_candidates_for_operator(
    *,
    operator_index: int,
    operator: Sequence[sp.Matrix],
    parent_metric: np.ndarray,
    product_metric: np.ndarray,
    correspondence: Correspondence,
    product_node_basis: BravaisNodeBasis | None,
    max_plane_index: int,
    max_planes_per_element: int,
) -> tuple[str, tuple[WeakPlaneCandidate, ...]]:
    audit = audit_operator_route(operator_index, operator, parent_metric)
    if audit.route != "axial_weak":
        return audit.route, ()
    if product_node_basis is None:
        return "requires explicit product primitive-node basis", ()

    candidates: list[WeakPlaneCandidate] = []
    failures: list[str] = []
    for element_index in audit.weak_element_indices:
        try:
            analysis = analyze_higher_order_element(
                operator[element_index],
                parent_metric,
                product_metric,
                correspondence,
                product_node_basis=product_node_basis,
                max_plane_index=max_plane_index,
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            failures.append(str(exc))
            continue

        for ranked in analysis.ranked_weak_planes[:max_planes_per_element]:
            result = ranked.result
            candidates.append(
                WeakPlaneCandidate(
                    operator_index=operator_index,
                    parent_rotation_order=int(analysis.element_audit.order),
                    parent_axis=tuple(int(x) for x in analysis.parent_axis),  # type: ignore[arg-type]
                    product_axis=tuple(int(x) for x in analysis.product_axis),  # type: ignore[arg-type]
                    plane1_primitive=tuple(int(x) for x in result.plane1_primitive),  # type: ignore[arg-type]
                    plane2_primitive=tuple(int(x) for x in result.plane2_primitive),  # type: ignore[arg-type]
                    generalized_twin_index=int(result.generalized_twin_index),
                    generalized_strain=float(result.generalized_strain),
                    generalized_shear=float(result.selected.generalized_shear),
                    intraplanar_distortion=float(ranked.geometry.intraplanar_distortion),
                    maximum_principal_intraplanar_strain=float(
                        ranked.geometry.maximum_principal_intraplanar_strain
                    ),
                    plane_complexity=int(ranked.plane_complexity),
                )
            )

    candidates.sort(
        key=lambda item: (
            item.intraplanar_distortion,
            item.plane_complexity,
            item.generalized_shear,
            item.operator_index,
            item.plane1_primitive,
            item.plane2_primitive,
        )
    )
    if candidates:
        return "calculated", tuple(candidates)
    if failures:
        return "weak route identified; plane calculation failed: " + failures[0], ()
    return "weak route identified; no candidate survived the configured search", ()


def build_twin_family_report(
    project_payload: Mapping[str, object],
    transformation_id: str,
    *,
    product_node_basis: BravaisNodeBasis | None = None,
    weak_max_plane_index: int = 6,
    weak_max_planes_per_element: int = 6,
) -> TwinFamilyReport:
    """Calculate topology, twin families and exact laminate habit planes.

    Parameters
    ----------
    project_payload:
        Standard repository ProjectState JSON payload.
    transformation_id:
        Transformation to analyze.
    product_node_basis:
        Explicit primitive-node basis used only for weak-plane enumeration.
        It is intentionally not guessed from point-group symmetry. If omitted,
        polar/higher-order weak operators are still identified, but their weak
        planes are not fabricated.
    """

    if weak_max_plane_index < 1:
        raise ValueError("weak_max_plane_index must be >= 1")
    if weak_max_planes_per_element < 1:
        raise ValueError("weak_max_planes_per_element must be >= 1")

    loaded = project_from_dict(dict(project_payload), source="twin_family_app")
    project = loaded.project
    project.validate().assert_passed()
    transformation = project.transformation(transformation_id)
    parent = project.phase(transformation.parent_phase_id)
    product = project.phase(transformation.product_phase_id)
    policy = project.numerical_policy

    parent_metric = np.asarray(parent.lattice.metric(), dtype=float)
    product_metric = np.asarray(product.lattice.metric(), dtype=float)
    parent_group = _exact_group_from_phase(parent, tolerance=float(policy.algebraic))
    product_group = _exact_group_from_phase(product, tolerance=float(policy.algebraic))
    groupoid = correspondence_groupoid(
        parent_group,
        product_group,
        transformation.correspondence,
    )

    # The two metric solvers are kept independent and are used as a cross-lock.
    ptmc = PTMCAdapter(project, transformation_id).analyze_all_twinning(
        dilatational_factor=1.0
    )
    bj = BallJamesAdapter(project, transformation_id).analyze()

    # Independent, source-free numerical certification before pairing/output.
    # Any violated tensor identity aborts; no invalid result reaches the tree.
    certify_ptmc_twinning(ptmc)

    mapping_tolerance = max(
        100.0 * float(policy.algebraic),
        100.0 * float(policy.representation),
        1.0e-8,
    )
    variant_records, correspondence_to_stretch, maximum_mapping_residual = (
        _build_correspondence_variants(
            groupoid=groupoid,
            base_correspondence=transformation.correspondence,
            parent_metric=parent_metric,
            product_metric=product_metric,
            ptmc_variants=ptmc.variants,
            mapping_tolerance=mapping_tolerance,
        )
    )

    bj_by_pair = _index_bj_twins(bj)
    ptmc_relations: dict[tuple[int, int], list[PTMCTwinRelation]] = defaultdict(list)
    for relation in ptmc.twin_relations:
        ptmc_relations[(int(relation.base_variant_index), int(relation.other_variant_index))].append(relation)

    habit_by_relation: dict[tuple[int, int, int], list[PTMCHabitSolution]] = defaultdict(list)
    for solution in ptmc.solutions:
        if solution.other_variant_index is None or solution.twin_branch is None:
            continue
        habit_by_relation[
            (
                int(solution.base_variant_index),
                int(solution.other_variant_index),
                int(solution.twin_branch),
            )
        ].append(solution)

    continuum_keys = {
        (
            int(item.base_variant_index),
            int(item.other_variant_index),
            int(item.twin_branch),
        )
        for item in ptmc.continuous_families
        if item.other_variant_index is not None and item.twin_branch is not None
    }

    family_pair_groups = _family_keys(groupoid)
    families: list[TwinFamilyRecord] = []
    warnings: list[str] = []

    for family_number, (operator_key, pairs) in enumerate(
        sorted(family_pair_groups.items(), key=lambda item: item[0]),
        start=1,
    ):
        route_audits = [
            audit_operator_route(index, groupoid.operators[index], parent_metric)
            for index in operator_key
        ]
        has_classical = any(audit.route == "classical_exact" for audit in route_audits)
        has_weak = any(audit.route == "axial_weak" for audit in route_audits)
        if has_classical:
            route = "classical_exact"
        elif has_weak:
            route = "axial_weak"
        else:
            route = "unsupported"

        classical_raw: list[CTTwin] = []
        if has_classical:
            for operator_index in operator_key:
                if audit_operator_route(
                    operator_index,
                    groupoid.operators[operator_index],
                    parent_metric,
                ).route == "classical_exact":
                    classical_raw.extend(
                        twins_from_operator(
                            groupoid.operators[operator_index],
                            parent_metric,
                            product_metric,
                            transformation.correspondence,
                        )
                    )
        classical_systems = _deduplicate_classical_systems(
            classical_raw,
            parent_metric,
            product_metric,
        )

        weak_statuses: list[str] = []
        weak_candidates: list[WeakPlaneCandidate] = []
        if has_weak and not has_classical:
            for operator_index in operator_key:
                status, candidates = _weak_candidates_for_operator(
                    operator_index=operator_index,
                    operator=groupoid.operators[operator_index],
                    parent_metric=parent_metric,
                    product_metric=product_metric,
                    correspondence=transformation.correspondence,
                    product_node_basis=product_node_basis,
                    max_plane_index=weak_max_plane_index,
                    max_planes_per_element=weak_max_planes_per_element,
                )
                weak_statuses.append(status)
                weak_candidates.extend(candidates)
            weak_candidates.sort(
                key=lambda item: (
                    item.intraplanar_distortion,
                    item.plane_complexity,
                    item.generalized_shear,
                )
            )
            deduplicated_weak: list[WeakPlaneCandidate] = []
            seen_weak: set[tuple[object, ...]] = set()
            for candidate in weak_candidates:
                key = (
                    candidate.parent_rotation_order,
                    candidate.parent_axis,
                    candidate.product_axis,
                    candidate.plane1_primitive,
                    candidate.plane2_primitive,
                )
                if key not in seen_weak:
                    seen_weak.add(key)
                    deduplicated_weak.append(candidate)
            weak_candidates = deduplicated_weak
            weak_status = "; ".join(dict.fromkeys(weak_statuses))
        elif has_classical:
            weak_status = "not applicable: classical mirror/twofold route exists in this operator family"
        else:
            weak_status = "not a supported classical or axial-weak operator family"

        pair_records: list[VariantPairRecord] = []
        for pair_number, (i, j) in enumerate(sorted(pairs), start=1):
            ui = correspondence_to_stretch.get(i)
            uj = correspondence_to_stretch.get(j)
            forward = int(groupoid.adjacency[i][j])
            reverse = int(groupoid.adjacency[j][i])

            if ui is None or uj is None:
                status = "correspondence pair has no complete orientation-preserving stretch mapping"
                constructions: tuple[PairTwinConstruction, ...] = ()
            elif ui == uj:
                status = "metric collapse: distinct correspondence variants map to the same stretch variant"
                constructions = ()
            else:
                relations = sorted(
                    ptmc_relations.get((ui, uj), []),
                    key=lambda item: int(item.branch),
                )
                if not relations:
                    status = "no exact martensite/martensite rank-one relation"
                    constructions = ()
                else:
                    status = "exact martensite/martensite rank-one relation"
                    built: list[PairTwinConstruction] = []
                    for construction_number, relation in enumerate(relations, start=1):
                        bj_branch, outer_residual, shear_residual = _match_bj_branch(
                            relation,
                            bj_by_pair.get((ui, uj), ()),
                            tolerance=float(policy.rank_one),
                        )
                        pair_plane_product = _canonical_projective_vector(
                            relation.twin_plane_product_crystal_base
                        )
                        pair_direction_product = _pair_shear_direction_product_crystal(
                            relation,
                            ptmc.variants[ui],
                            product_metric,
                        )
                        (
                            classification,
                            representations,
                            classification_status,
                            classical_system_ids,
                            discrete_plane_angle_deg,
                            discrete_direction_angle_deg,
                            discrete_shear_relative_residual,
                        ) = _pair_classification(
                            bj_branch,
                            classical_systems,
                            family_has_classical_route=has_classical,
                            shear_magnitude=float(relation.twin_shear_magnitude),
                            shear_tolerance=max(
                                100.0 * float(policy.rank_one),
                                1.0e-7,
                            ),
                            pair_plane_product=pair_plane_product,
                            pair_direction_product=pair_direction_product,
                            product_metric=product_metric,
                            geometry_tolerance_deg=max(
                                100.0 * float(policy.projective_angle_deg),
                                1.0e-5,
                            ),
                            parent_group=parent_group,
                        )

                        key = (ui, uj, int(relation.branch))
                        raw_habits = sorted(
                            habit_by_relation.get(key, ()),
                            key=lambda item: (
                                float(item.parameter_value),
                                int(item.habit_branch),
                            ),
                        )
                        exact_habits = [
                            item for item in raw_habits if bool(item.true_invariant_plane)
                        ]
                        habits = tuple(
                            _habit_solution(item, parent_metric) for item in exact_habits
                        )
                        if key in continuum_keys:
                            habit_status = "continuous exact compatible fraction family"
                        elif habits:
                            habit_status = "exact A/M habit-plane solution(s)"
                        else:
                            habit_status = "no exact A/M habit-plane solution for this twin construction"

                        built.append(
                            PairTwinConstruction(
                                construction_id=f"P{pair_number}.B{construction_number}",
                                branch=int(relation.branch),
                                classification=classification,
                                representations=representations,
                                classification_status=classification_status,
                                classical_system_ids=classical_system_ids,
                                shear_magnitude=float(relation.twin_shear_magnitude),
                                a_parent_cartesian=_vector3(relation.a),
                                n_parent_cartesian=_vector3(relation.n_reference),
                                twin_plane_product_crystal=pair_plane_product,
                                shear_direction_product_crystal=pair_direction_product,
                                rank_one_residual=float(relation.rank_one_residual),
                                rotation_residual=float(relation.rotation_residual),
                                independent_outer_product_residual=outer_residual,
                                independent_shear_relative_residual=shear_residual,
                                discrete_plane_angle_deg=discrete_plane_angle_deg,
                                discrete_direction_angle_deg=discrete_direction_angle_deg,
                                discrete_shear_relative_residual=discrete_shear_relative_residual,
                                habit_status=habit_status,
                                habit_solutions=habits,
                                continuum_fraction=key in continuum_keys,
                            )
                        )
                    constructions = tuple(built)

            pair_records.append(
                VariantPairRecord(
                    pair_id=f"M{i + 1}–M{j + 1}",
                    variant_i=i + 1,
                    variant_j=j + 1,
                    operator_forward=forward,
                    operator_reverse=reverse,
                    stretch_i=None if ui is None else ui + 1,
                    stretch_j=None if uj is None else uj + 1,
                    status=status,
                    constructions=constructions,
                )
            )

        representative = _representative_pair(pairs)
        families.append(
            TwinFamilyRecord(
                family_id=f"F{family_number}",
                operator_indices=operator_key,
                route=route,
                representative_pair=(representative[0] + 1, representative[1] + 1),
                equivalent_pairs=tuple((i + 1, j + 1) for i, j in sorted(pairs)),
                classical_systems=classical_systems,
                weak_status=weak_status,
                weak_candidates=tuple(weak_candidates),
                pair_records=tuple(pair_records),
            )
        )

    mapped = [item for item in variant_records if item.stretch_variant_index is not None]
    stretch_to_correspondence: dict[int, int] = defaultdict(int)
    for item in mapped:
        stretch_to_correspondence[int(item.stretch_variant_index)] += 1
    collapsed_count = sum(max(0, count - 1) for count in stretch_to_correspondence.values())
    unmapped_count = len(variant_records) - len(mapped)

    if unmapped_count:
        warnings.append(
            f"{unmapped_count} correspondence variant(s) have no orientation-preserving "
            "mapping to the classical stretch-variant family. They remain visible as "
            "topological variants and are not silently discarded."
        )
    if collapsed_count:
        warnings.append(
            f"{collapsed_count} correspondence variant(s) collapse onto already-used "
            "stretch variants for the supplied metric. Topological and metric variant "
            "numbers must therefore not be identified."
        )
    if any(family.route == "axial_weak" and not family.weak_candidates for family in families):
        warnings.append(
            "At least one higher-order weak family was identified without a calculated "
            "weak plane. Supply the product primitive-node basis explicitly; centering "
            "is not inferred from point-group symmetry."
        )

    return TwinFamilyReport(
        transformation_id=transformation_id,
        parent_phase_id=parent.phase_id,
        product_phase_id=product.phase_id,
        correspondence_variants=variant_records,
        families=tuple(families),
        audit=ScientificAudit(
            topology_variant_count=len(groupoid.variants),
            stretch_variant_count=len(ptmc.variants),
            correspondence_subgroup_order=len(groupoid.subgroup),
            operator_count=len(groupoid.operators),
            mapped_correspondence_variant_count=len(mapped),
            collapsed_correspondence_variant_count=int(collapsed_count),
            unmapped_correspondence_variant_count=int(unmapped_count),
            maximum_correspondence_to_stretch_residual=float(maximum_mapping_residual),
            ptmc_maximum_residual=float(ptmc.audit.maximum_residual),
            ball_james_maximum_residual=float(bj.audit.maximum_residual),
            warnings=tuple(warnings),
        ),
    )
