from __future__ import annotations

"""High-level scientific workstation orchestration.

This module intentionally contains no replacement implementation of CT, PTMC,
Ball-James, Mallard, cofactor, orientation-topology, or crystallographic metric
mathematics.  It composes the validated ``cualni_cryst`` services into
UI-facing workflows and serializes their results.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Mapping, Sequence

import numpy as np
import sympy as sp

from cualni_cryst.ball_james import mallard_law_twins
from cualni_cryst.calculation_service import (
    CalculationKind,
    CalculationService,
    TransformationBundle,
)
from cualni_cryst.calpad import CalPadService
from cualni_cryst.cofactor import evaluate_cofactor_conditions
from cualni_cryst.crystal_objects import Direction, Plane
from cualni_cryst.crystallography_console import ConsoleInputKind, parse_crystal_input
from cualni_cryst.orientation import (
    EulerConvention,
    OrientationService,
    axis_angle_from_matrix,
    euler_zxz_from_matrix,
    rotation_audit,
)
from cualni_cryst.project_io import LoadedProject, project_from_dict
from cualni_cryst.project_state import (
    OrientationDefinition,
    OrientationState,
    OrientationTheoryOrigin,
)
from cualni_cryst.ptmc import (
    solve_ptmc_laminate,
    solve_ptmc_laminate_numerical_crosscheck,
)
from cualni_cryst.representation import CartesianConvention

from .errors import ApplicationError, input_error, internal_error, scientific_domain_error


@dataclass(frozen=True)
class WorkbenchContext:
    loaded: LoadedProject
    transformation_id: str
    service: CalculationService
    bundle: TransformationBundle

    @property
    def project(self):
        return self.loaded.project

    @property
    def transformation(self):
        return self.project.transformation(self.transformation_id)

    @property
    def parent(self):
        return self.project.phase(self.transformation.parent_phase_id)

    @property
    def product(self):
        return self.project.phase(self.transformation.product_phase_id)


def _safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)




def _exact_indices_from_canonical(text: str) -> sp.Matrix:
    """Recover exact rational index values from parser-normalized notation."""
    body = text.strip()[1:-1].strip()
    tokens = body.replace(",", " ").split()
    if len(tokens) != 3:
        raise input_error(f"Expected three crystallographic indices in {text!r}.")
    values = []
    for token in tokens:
        fraction = Fraction(token)
        values.append(sp.Rational(fraction.numerator, fraction.denominator))
    return sp.Matrix(values)

def _matrix3(values: Sequence[Sequence[object]], *, name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise input_error(f"{name} must contain numeric entries.", detail=str(exc)) from exc
    if arr.shape != (3, 3):
        raise input_error(f"{name} must be exactly 3×3; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise input_error(f"{name} contains non-finite entries.")
    return arr


def _context(
    payload: Mapping[str, object],
    transformation_id: str | None = None,
) -> WorkbenchContext:
    if not isinstance(payload, Mapping):
        raise input_error("Project payload must be a JSON object.")
    try:
        loaded = project_from_dict(dict(payload), source="workbench")
    except ApplicationError:
        raise
    except (TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
        raise input_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        raise internal_error(
            "The project could not be loaded for workstation analysis.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    ids = tuple(item.transformation_id for item in loaded.project.transformations)
    if transformation_id and str(transformation_id).strip():
        selected = str(transformation_id).strip()
        if selected not in ids:
            raise input_error(
                f"Unknown transformation_id {selected!r}. Available: {', '.join(ids) or '(none)'}."
            )
    elif len(ids) == 1:
        selected = ids[0]
    elif not ids:
        raise input_error("The project contains no transformation.")
    else:
        raise input_error("Select a transformation; the project contains more than one.")

    service = CalculationService(loaded.project)
    try:
        result = service.compute(CalculationKind.TRANSFORMATION_BUNDLE, selected)
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        raise internal_error(
            "The scientific calculation service failed while building workstation context.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(result, TransformationBundle):
        raise internal_error(
            "Unexpected transformation-bundle type.",
            detail=type(result).__name__,
        )
    return WorkbenchContext(loaded, selected, service, result)


def _orientation_variant_payload(variant: object) -> dict[str, object]:
    data = variant.to_dict()  # OrientationVariant has a stable to_dict contract.
    R_parent_from_product = np.asarray(data["matrix_reference_from_moving"], dtype=float)
    reverse = R_parent_from_product.T
    forward_axis = axis_angle_from_matrix(R_parent_from_product)
    reverse_axis = axis_angle_from_matrix(reverse)
    return {
        **data,
        "R_parent_from_product": R_parent_from_product,
        "R_product_from_parent": reverse,
        "forward_axis_angle": forward_axis.to_dict(),
        "reverse_axis_angle": reverse_axis.to_dict(),
    }


def _orientation_analysis(
    ctx: WorkbenchContext,
    state: OrientationState,
    *,
    origin_note: str,
) -> dict[str, object]:
    service = OrientationService(ctx.project)
    report = service.report(state)
    variants = service.variants(state)
    operators = service.operators(state)
    return _safe(
        {
            "origin_note": origin_note,
            "state": state.to_dict(),
            "report": report.to_dict(),
            "variants": [_orientation_variant_payload(item) for item in variants],
            "operators": [item.to_dict() for item in operators],
            "convention": (
                "Each variant stores R_parent_from_product with x_parent = "
                "R_parent_from_product @ x_product.  The reverse matrix is its transpose."
            ),
        }
    )  # type: ignore[return-value]


def orientation_from_polar_correspondence(
    payload: Mapping[str, object],
    *,
    transformation_id: str | None = None,
) -> dict[str, object]:
    """Use the polar rotation of F as an explicitly labelled OR candidate.

    ``RepresentationBridge`` returns R in F=R U, i.e. it maps parent Cartesian
    vectors to product Cartesian vectors. ``OrientationState`` uses the opposite
    convention x_parent=R_parent_from_product x_product, hence the transpose.
    """

    ctx = _context(payload, transformation_id)
    transformation = ctx.transformation
    R_product_from_parent = np.asarray(ctx.bundle.representation.polar_rotation, dtype=float)
    R_parent_from_product = R_product_from_parent.T
    service = OrientationService(ctx.project)
    try:
        state = service.state_from_matrix(
            ctx.parent.phase_id,
            ctx.product.phase_id,
            R_parent_from_product,
            orientation_id=f"polar_{ctx.transformation_id}",
            label="Polar-rotation candidate from correspondence deformation",
            reference_convention=transformation.parent_cartesian_convention,
            moving_convention=transformation.product_cartesian_convention,
            definition_method=OrientationDefinition.POLAR_CORRESPONDENCE,
            theory_origin=OrientationTheoryOrigin.POLAR_CORRESPONDENCE,
            transformation_id=ctx.transformation_id,
            notes=(
                "Finite-strain polar rotation of the supplied correspondence deformation; "
                "not automatically an experimental OR, Cayron T, PTMC OR, or Ball-James OR."
            ),
        )
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc
    return _orientation_analysis(
        ctx,
        state,
        origin_note=(
            "Generated from the current transformation's polar decomposition. "
            "Treat it as a finite-strain rotation candidate, not as experimentally established OR data."
        ),
    )


def orientation_from_matrix(
    payload: Mapping[str, object],
    matrix_parent_from_product: Sequence[Sequence[object]],
    *,
    transformation_id: str | None = None,
    parent_convention: str = CartesianConvention.PTCLAB_A_X_C_XZ.value,
    product_convention: str = CartesianConvention.PTCLAB_A_X_C_XZ.value,
    repair: bool = False,
) -> dict[str, object]:
    ctx = _context(payload, transformation_id)
    matrix = _matrix3(matrix_parent_from_product, name="R(parent←product)")
    service = OrientationService(ctx.project)
    try:
        state = service.state_from_matrix(
            ctx.parent.phase_id,
            ctx.product.phase_id,
            matrix,
            orientation_id=f"manual_{ctx.transformation_id}",
            label="User-specified orientation relationship",
            reference_convention=CartesianConvention(parent_convention),
            moving_convention=CartesianConvention(product_convention),
            definition_method=OrientationDefinition.USER_MATRIX,
            theory_origin=OrientationTheoryOrigin.USER_DEFINED,
            transformation_id=ctx.transformation_id,
            repair=bool(repair),
        )
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc
    return _orientation_analysis(ctx, state, origin_note="User-specified proper rotation.")


def orientation_from_euler(
    payload: Mapping[str, object],
    phi1_deg: float,
    Phi_deg: float,
    phi2_deg: float,
    *,
    transformation_id: str | None = None,
    euler_convention: str = EulerConvention.ZXZ_ACTIVE.value,
    parent_convention: str = CartesianConvention.PTCLAB_A_X_C_XZ.value,
    product_convention: str = CartesianConvention.PTCLAB_A_X_C_XZ.value,
) -> dict[str, object]:
    ctx = _context(payload, transformation_id)
    service = OrientationService(ctx.project)
    try:
        state = service.state_from_euler(
            ctx.parent.phase_id,
            ctx.product.phase_id,
            float(phi1_deg),
            float(Phi_deg),
            float(phi2_deg),
            convention=EulerConvention(euler_convention),
            orientation_id=f"euler_{ctx.transformation_id}",
            label="User-specified ZXZ orientation relationship",
            reference_convention=CartesianConvention(parent_convention),
            moving_convention=CartesianConvention(product_convention),
            theory_origin=OrientationTheoryOrigin.USER_DEFINED,
            transformation_id=ctx.transformation_id,
        )
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc
    return _orientation_analysis(
        ctx,
        state,
        origin_note=(
            f"User ZXZ Euler input ({EulerConvention(euler_convention).value}); "
            "the selected Cartesian-frame conventions are reported explicitly."
        ),
    )


def orientation_from_parallelisms(
    payload: Mapping[str, object],
    parent_first: str,
    product_first: str,
    parent_second: str,
    product_second: str,
    *,
    transformation_id: str | None = None,
    parent_convention: str = CartesianConvention.PTCLAB_A_X_C_XZ.value,
    product_convention: str = CartesianConvention.PTCLAB_A_X_C_XZ.value,
) -> dict[str, object]:
    ctx = _context(payload, transformation_id)
    service = OrientationService(ctx.project)
    try:
        solved = service.state_from_parallelisms(
            ctx.parent.phase_id,
            ctx.product.phase_id,
            parent_first,
            product_first,
            parent_second,
            product_second,
            orientation_id=f"parallel_{ctx.transformation_id}",
            label="OR from two crystallographic parallelisms",
            unoriented_directions=True,
            reference_convention=CartesianConvention(parent_convention),
            moving_convention=CartesianConvention(product_convention),
            theory_origin=OrientationTheoryOrigin.USER_DEFINED,
        )
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc

    candidates = [
        _orientation_analysis(
            ctx,
            candidate.state,
            origin_note=(
                "Exact proper-rotation candidate satisfying the two supplied parallelisms."
            ),
        )
        for candidate in solved.candidates
    ]
    return _safe({"solve": solved.to_dict(), "candidates": candidates})  # type: ignore[return-value]


def map_correspondence_object(
    payload: Mapping[str, object],
    object_text: str,
    *,
    source_phase: str,
    transformation_id: str | None = None,
) -> dict[str, object]:
    """Map a direction or plane through the exact crystallographic correspondence."""

    ctx = _context(payload, transformation_id)
    try:
        parsed = parse_crystal_input(object_text)
        exact_indices = _exact_indices_from_canonical(parsed.canonical_text)
        C = ctx.transformation.correspondence
        source_key = source_phase.strip().lower()
        parent_tokens = {"parent", "a", ctx.parent.phase_id.lower()}
        product_tokens = {"product", "daughter", "m", ctx.product.phase_id.lower()}
        if source_key in parent_tokens:
            if parsed.kind is ConsoleInputKind.DIRECTION:
                mapped = C.map_direction_A_to_M(exact_indices)
                relation = "u_product = C(product←parent) u_parent"
                target = ctx.product.phase_id
            else:
                mapped = C.map_plane_A_to_M(exact_indices)
                relation = "p_product = C(product←parent)^(-T) p_parent"
                target = ctx.product.phase_id
            source = ctx.parent.phase_id
        elif source_key in product_tokens:
            if parsed.kind is ConsoleInputKind.DIRECTION:
                mapped = C.map_direction_M_to_A(exact_indices)
                relation = "u_parent = C(parent←product) u_product"
                target = ctx.parent.phase_id
            else:
                mapped = C.map_plane_M_to_A(exact_indices)
                relation = "p_parent = C(product←parent)^T p_product"
                target = ctx.parent.phase_id
            source = ctx.product.phase_id
        else:
            raise input_error(
                f"source_phase must identify parent ({ctx.parent.phase_id}) or product ({ctx.product.phase_id})."
            )
        exact = [str(mapped[index]) for index in range(3)]
        numeric = [float(mapped[index]) for index in range(3)]
        return {
            "source_phase_id": source,
            "target_phase_id": target,
            "object_kind": parsed.kind.value,
            "input": parsed.canonical_text,
            "mapped_exact_coefficients": exact,
            "mapped_numeric_coefficients": numeric,
            "relation": relation,
        }
    except ApplicationError:
        raise
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc


def map_orientation_object(
    payload: Mapping[str, object],
    matrix_parent_from_product: Sequence[Sequence[object]],
    object_text: str,
    *,
    source_phase: str,
    transformation_id: str | None = None,
    parent_convention: str = CartesianConvention.PTCLAB_A_X_C_XZ.value,
    product_convention: str = CartesianConvention.PTCLAB_A_X_C_XZ.value,
) -> dict[str, object]:
    """Map a physical direction/plane through a chosen OR, in either direction."""

    ctx = _context(payload, transformation_id)
    service = OrientationService(ctx.project)
    matrix = _matrix3(matrix_parent_from_product, name="R(parent←product)")
    try:
        state = service.state_from_matrix(
            ctx.parent.phase_id,
            ctx.product.phase_id,
            matrix,
            orientation_id="ui_map_or",
            reference_convention=CartesianConvention(parent_convention),
            moving_convention=CartesianConvention(product_convention),
            definition_method=OrientationDefinition.USER_MATRIX,
            theory_origin=OrientationTheoryOrigin.USER_DEFINED,
            transformation_id=ctx.transformation_id,
        )
        parsed = parse_crystal_input(object_text)
        source_key = source_phase.strip().lower()
        if source_key in {"parent", "a", ctx.parent.phase_id.lower()}:
            basis = ctx.parent.basis
            source = ctx.parent.phase_id
            target = ctx.product.phase_id
        elif source_key in {"product", "daughter", "m", ctx.product.phase_id.lower()}:
            basis = ctx.product.basis
            source = ctx.product.phase_id
            target = ctx.parent.phase_id
        else:
            raise input_error(
                f"source_phase must identify parent ({ctx.parent.phase_id}) or product ({ctx.product.phase_id})."
            )

        if parsed.kind is ConsoleInputKind.DIRECTION:
            mapped = service.map_direction(state, Direction(parsed.indices, basis))
        else:
            mapped = service.map_plane(state, Plane(parsed.indices, basis))
        return _safe(
            {
                "source_phase_id": source,
                "target_phase_id": target,
                "object_kind": parsed.kind.value,
                "input": parsed.canonical_text,
                "mapped_coefficients": mapped.array,
                "mapping": "physical orientation relationship (not correspondence mapping)",
            }
        )  # type: ignore[return-value]
    except ApplicationError:
        raise
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc


def reconstruct_sample_orientations(
    payload: Mapping[str, object],
    orientation_analysis: Mapping[str, object],
    observed_sample_from_crystal: Sequence[Sequence[object]],
    *,
    observed_phase: str,
    transformation_id: str | None = None,
) -> dict[str, object]:
    """Bidirectional parent/product orientation reconstruction.

    Input ``g`` is explicitly ``x_sample = g_sample_from_crystal @ x_crystal``
    in the same Cartesian phase frames used by the chosen OR.  This function
    deliberately does not guess a vendor EBSD Euler convention.
    """

    ctx = _context(payload, transformation_id)
    observed = _matrix3(observed_sample_from_crystal, name="g(sample←crystal)")
    audit = rotation_audit(observed)
    if audit.maximum_residual > ctx.project.numerical_policy.representation:
        raise input_error(
            "Observed sample-from-crystal matrix must be a proper rotation within the project representation tolerance.",
            detail=f"maximum residual={audit.maximum_residual:.3e}",
        )

    variants = orientation_analysis.get("variants")
    if not isinstance(variants, list) or not variants:
        raise input_error("Orientation analysis contains no variants to reconstruct.")

    phase_key = observed_phase.strip().lower()
    parent_tokens = {"parent", "a", ctx.parent.phase_id.lower()}
    product_tokens = {"product", "daughter", "m", ctx.product.phase_id.lower()}
    rows: list[dict[str, object]] = []

    for item in variants:
        if not isinstance(item, Mapping):
            continue
        R_A_from_M = np.asarray(item["R_parent_from_product"], dtype=float)
        if phase_key in parent_tokens:
            candidate = observed @ R_A_from_M
            reconstructed = ctx.product.phase_id
            relation = "g(sample←product) = g(sample←parent) · R(parent←product)"
        elif phase_key in product_tokens:
            candidate = observed @ R_A_from_M.T
            reconstructed = ctx.parent.phase_id
            relation = "g(sample←parent) = g(sample←product) · R(product←parent)"
        else:
            raise input_error(
                f"observed_phase must identify parent ({ctx.parent.phase_id}) or product ({ctx.product.phase_id})."
            )
        candidate_audit = rotation_audit(candidate)
        active = euler_zxz_from_matrix(candidate, EulerConvention.ZXZ_ACTIVE)
        passive = euler_zxz_from_matrix(candidate, EulerConvention.ZXZ_PASSIVE)
        rows.append(
            {
                "variant_index": int(item["index"]),
                "reconstructed_phase_id": reconstructed,
                "g_sample_from_reconstructed": candidate,
                "rotation_audit": candidate_audit.to_dict(),
                "euler_zxz_active": active.to_dict(),
                "euler_zxz_passive": passive.to_dict(),
            }
        )

    return _safe(
        {
            "observed_phase": ctx.parent.phase_id if phase_key in parent_tokens else ctx.product.phase_id,
            "input_convention": "x_sample = g_sample_from_crystal @ x_crystal",
            "relation": relation,
            "candidates": rows,
            "note": (
                "No EBSD vendor convention is inferred. Convert vendor Euler/orientation data "
                "to the stated sample-from-crystal Cartesian convention before using reconstruction."
            ),
        }
    )  # type: ignore[return-value]


def _analytic_rank_one_payload(solution: object) -> dict[str, object]:
    return {
        "branch": int(solution.branch),
        "rotation": solution.rotation,
        "a": solution.a,
        "n": solution.n,
        "residual": float(solution.residual),
        "eigenvalues_C": solution.eigenvalues_C,
    }


def _cofactor_payload(result: object) -> dict[str, object]:
    return {
        "lambdas": result.lambdas,
        "middle_eigenvector": result.middle_eigenvector,
        "cc1_residual": float(result.cc1_residual),
        "cc2_residual": float(result.cc2_residual),
        "cc2_simplified": float(result.cc2_simplified),
        "cc3_margin": float(result.cc3_margin),
        "cc1_satisfied": bool(result.cc1_satisfied),
        "cc2_satisfied": bool(result.cc2_satisfied),
        "cc3_satisfied": bool(result.cc3_satisfied),
        "satisfied": bool(result.satisfied),
    }


def _ptmc_payload(solution: object, *, independent_residual: float | None) -> dict[str, object]:
    return {
        "volume_fraction": float(solution.volume_fraction),
        "average_deformation": solution.average_deformation,
        "middle_stretch_residual": float(solution.middle_stretch_residual),
        "habit_connections": [
            _analytic_rank_one_payload(item) for item in solution.habit_connections
        ],
        "independent_rank_one_crosscheck_residual": independent_residual,
    }


def martensite_variant_pair_analysis(
    payload: Mapping[str, object],
    variant_i: int,
    variant_j: int,
    *,
    transformation_id: str | None = None,
    independent_crosscheck: bool = True,
) -> dict[str, object]:
    """Run Mallard -> classical PTMC -> cofactor workflow for a selected pair.

    The function searches the *registered parent proper symmetries* for an
    order-two symmetry relating the selected stretch variants, then delegates
    all twin/PTMC/cofactor equations to the scientific modules.
    """

    ctx = _context(payload, transformation_id)
    variants = tuple(np.asarray(item, dtype=float) for item in ctx.bundle.stretch_variants.variants)
    count = len(variants)
    if not 0 <= int(variant_i) < count or not 0 <= int(variant_j) < count:
        raise input_error(f"Variant indices must lie in 0..{count - 1}.")
    if int(variant_i) == int(variant_j):
        raise input_error("Choose two distinct stretch variants for a twin analysis.")

    Ui = variants[int(variant_i)]
    Uj = variants[int(variant_j)]
    orientation_service = OrientationService(ctx.project)
    proper = orientation_service.proper_symmetry_cartesian(
        ctx.parent,
        CartesianConvention.SYMMETRIC_METRIC,
    )
    policy = ctx.project.numerical_policy
    found: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()

    for symmetry_index, Q in proper:
        Q = np.asarray(Q, dtype=float)
        involution_residual = float(np.linalg.norm(Q @ Q - np.eye(3)))
        identity_residual = float(np.linalg.norm(Q - np.eye(3)))
        if involution_residual > policy.representation:
            continue
        if identity_residual <= policy.representation:
            continue
        axis_angle = axis_angle_from_matrix(Q)
        axis = np.asarray(axis_angle.axis, dtype=float)
        try:
            twins = mallard_law_twins(Ui, Uj, axis, tol=policy.representation)
        except ValueError:
            continue

        for twin in twins:
            key = (symmetry_index, str(twin.kind))
            if key in seen:
                continue
            seen.add(key)
            cofactor = evaluate_cofactor_conditions(
                Ui,
                twin.a,
                twin.n,
                tol=policy.rank_one,
            )
            ptmc_error = ""
            ptmc_rows: list[dict[str, object]] = []
            try:
                ptmc_solutions = solve_ptmc_laminate(
                    Ui,
                    twin.a,
                    twin.n,
                    tol=policy.rank_one,
                )
                for solution in ptmc_solutions:
                    cross_residual: float | None = None
                    if independent_crosscheck:
                        cross = solve_ptmc_laminate_numerical_crosscheck(
                            Ui,
                            twin.a,
                            twin.n,
                            solution.volume_fraction,
                        )
                        cross_residual = float(cross.residual)
                    ptmc_rows.append(
                        _ptmc_payload(solution, independent_residual=cross_residual)
                    )
            except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                ptmc_error = str(exc)

            found.append(
                {
                    "parent_symmetry_index": int(symmetry_index),
                    "twofold_axis_parent_symmetric_cartesian": axis,
                    "twofold_rotation_angle_deg": float(axis_angle.angle_deg),
                    "involution_residual": involution_residual,
                    "mallard_kind": twin.kind,
                    "mallard_rotation": twin.rotation,
                    "twin_a": twin.a,
                    "twin_n": twin.n,
                    "twin_shear_magnitude": float(twin.shear_magnitude),
                    "mallard_residual": float(twin.residual),
                    "cofactor": _cofactor_payload(cofactor),
                    "ptmc": ptmc_rows,
                    "ptmc_note": ptmc_error,
                }
            )

    return _safe(
        {
            "variant_i": int(variant_i),
            "variant_j": int(variant_j),
            "variant_count": count,
            "relations": found,
            "tolerance_policy": {
                "representation": float(policy.representation),
                "rank_one": float(policy.rank_one),
            },
            "note": (
                "No result is fabricated when a selected pair is not related by a registered "
                "parent order-two symmetry. PTMC and cofactor outputs are evaluated only from "
                "Mallard twin solutions returned by the scientific backend."
            ),
        }
    )  # type: ignore[return-value]


def manual_ptmc_cofactor_analysis(
    payload: Mapping[str, object],
    variant_index: int,
    twin_a: Sequence[object],
    twin_n: Sequence[object],
    *,
    transformation_id: str | None = None,
    independent_crosscheck: bool = True,
) -> dict[str, object]:
    """Advanced route for a user-specified lattice-invariant shear a⊗n."""

    ctx = _context(payload, transformation_id)
    variants = tuple(np.asarray(item, dtype=float) for item in ctx.bundle.stretch_variants.variants)
    if not 0 <= int(variant_index) < len(variants):
        raise input_error(f"variant_index must lie in 0..{len(variants) - 1}.")
    try:
        a = np.asarray(twin_a, dtype=float).reshape(3)
        n = np.asarray(twin_n, dtype=float).reshape(3)
    except (TypeError, ValueError) as exc:
        raise input_error("twin_a and twin_n must each contain three numbers.", detail=str(exc)) from exc
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(n)):
        raise input_error("twin_a and twin_n must be finite.")
    if float(np.linalg.norm(a)) == 0.0 or float(np.linalg.norm(n)) == 0.0:
        raise input_error("twin_a and twin_n must be non-zero.")

    U = variants[int(variant_index)]
    policy = ctx.project.numerical_policy
    cofactor = evaluate_cofactor_conditions(U, a, n, tol=policy.rank_one)
    ptmc_rows: list[dict[str, object]] = []
    ptmc_error = ""
    try:
        solutions = solve_ptmc_laminate(U, a, n, tol=policy.rank_one)
        for solution in solutions:
            cross_residual: float | None = None
            if independent_crosscheck:
                cross = solve_ptmc_laminate_numerical_crosscheck(
                    U, a, n, solution.volume_fraction
                )
                cross_residual = float(cross.residual)
            ptmc_rows.append(_ptmc_payload(solution, independent_residual=cross_residual))
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        ptmc_error = str(exc)

    return _safe(
        {
            "variant_index": int(variant_index),
            "twin_a": a,
            "twin_n": n,
            "cofactor": _cofactor_payload(cofactor),
            "ptmc": ptmc_rows,
            "ptmc_note": ptmc_error,
            "tolerance_policy": {"rank_one": float(policy.rank_one)},
            "note": (
                "The application does not infer a missing twin system in this manual route; "
                "a and n are exactly the user-supplied lattice-invariant shear data."
            ),
        }
    )  # type: ignore[return-value]


def calpad_phase_cell(
    payload: Mapping[str, object],
    phase_query: str,
    *,
    transformation_id: str | None = None,
) -> dict[str, object]:
    ctx = _context(payload, transformation_id)
    try:
        return _safe(CalPadService(ctx.project).phase_cell(phase_query).to_dict())  # type: ignore[return-value]
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc


def calpad_normal_conversion(
    payload: Mapping[str, object],
    phase_query: str,
    object_text: str,
    *,
    max_index: int = 12,
    transformation_id: str | None = None,
) -> dict[str, object]:
    ctx = _context(payload, transformation_id)
    try:
        report = CalPadService(ctx.project).normal_conversion(
            phase_query,
            object_text,
            max_index=int(max_index),
        )
        return _safe(report.to_dict())  # type: ignore[return-value]
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc


def calpad_low_index_table(
    payload: Mapping[str, object],
    phase_query: str,
    target_text: str,
    *,
    candidate_kind: str | None = None,
    max_index: int = 3,
    limit: int = 20,
    angle_sense: str = "projective",
    transformation_id: str | None = None,
) -> dict[str, object]:
    ctx = _context(payload, transformation_id)
    try:
        report = CalPadService(ctx.project).low_index_table(
            phase_query,
            target_text,
            candidate_kind=candidate_kind,
            max_index=int(max_index),
            limit=int(limit),
            angle_sense=angle_sense,
        )
        return _safe(report.to_dict())  # type: ignore[return-value]
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc
