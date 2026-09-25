from __future__ import annotations

"""Stable application boundary between a GUI and the verified science engine.

This file intentionally contains *no crystallographic equations*.  Its job is:

1. validate/normalize user-facing data without changing its scientific meaning;
2. build the existing ``ProjectState`` through ``project_io``;
3. call ``CalculationService``;
4. serialize the returned scientific objects for a browser/UI.

Keeping this adapter narrow means the presentation layer can later move from
Streamlit to another frontend without duplicating CMC, SMC, PTMC, Ball-James,
metric, symmetry, or variant mathematics.
"""

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from cualni_cryst.calculation_service import (
    CalculationKind,
    CalculationService,
    TransformationBundle,
)
from cualni_cryst.point_groups import point_group_table
from cualni_cryst.project_io import SCHEMA_VERSION, project_from_dict

from .errors import ApplicationError, input_error, internal_error, scientific_domain_error


MatrixText3 = tuple[
    tuple[str, str, str],
    tuple[str, str, str],
    tuple[str, str, str],
]


def _require_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise input_error(f"{name} must be finite; got {value!r}.")
    return number


def _clean_identifier(value: str, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise input_error(f"{name} must not be empty.")
    return text


def _exact_scalar_text(value: object, *, name: str) -> str:
    """Validate a user-entered exact rational scalar without converting to float."""

    if isinstance(value, bool):
        raise input_error(f"{name}: Boolean values are not valid matrix entries.")
    text = str(value).strip()
    if not text:
        raise input_error(f"{name}: matrix entry must not be empty.")
    try:
        Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise input_error(
            f"{name}: {text!r} is not an integer, decimal, or rational p/q.",
            detail=str(exc),
        ) from exc
    return text


def _matrix_text3(values: Sequence[Sequence[object]], *, name: str) -> MatrixText3:
    if len(values) != 3 or any(len(row) != 3 for row in values):
        raise input_error(f"{name} must contain exactly 3 rows and 3 columns.")
    rows: list[tuple[str, str, str]] = []
    for i, row in enumerate(values):
        clean = tuple(
            _exact_scalar_text(value, name=f"{name}[{i + 1},{j + 1}]")
            for j, value in enumerate(row)
        )
        rows.append(clean)  # type: ignore[arg-type]
    return tuple(rows)  # type: ignore[return-value]


def _json_safe(value: object) -> object:
    """Return strict-JSON-safe data without modifying scientific classifications.

    Non-finite diagnostics (for example an inapplicable route-disagreement
    residual) become JSON ``null`` rather than non-standard NaN/Infinity tokens.
    Arrays and NumPy scalars are presentation-only conversions.
    """

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class LatticeInput:
    a: float
    b: float
    c: float
    alpha_deg: float = 90.0
    beta_deg: float = 90.0
    gamma_deg: float = 90.0
    length_unit: str = "angstrom"

    def validated(self) -> "LatticeInput":
        a = _require_finite(self.a, "a")
        b = _require_finite(self.b, "b")
        c = _require_finite(self.c, "c")
        if min(a, b, c) <= 0.0:
            raise input_error("Lattice lengths a, b, and c must all be > 0.")

        alpha = _require_finite(self.alpha_deg, "alpha")
        beta = _require_finite(self.beta_deg, "beta")
        gamma = _require_finite(self.gamma_deg, "gamma")
        for name, angle in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not 0.0 < angle < 180.0:
                raise input_error(f"{name} must lie strictly between 0° and 180°.")

        unit = str(self.length_unit).strip()
        if not unit:
            raise input_error("Length unit must not be empty.")
        return LatticeInput(a, b, c, alpha, beta, gamma, unit)

    def to_payload(self) -> dict[str, object]:
        cell = self.validated()
        return {
            "a": cell.a,
            "b": cell.b,
            "c": cell.c,
            "alpha_deg": cell.alpha_deg,
            "beta_deg": cell.beta_deg,
            "gamma_deg": cell.gamma_deg,
            "length_unit": cell.length_unit,
        }


@dataclass(frozen=True)
class PhaseInput:
    phase_id: str
    label: str
    lattice: LatticeInput
    point_group: str = "1"
    physical_phase: str = ""
    cell_representation: str = "conventional"

    def to_payload(self) -> dict[str, object]:
        phase_id = _clean_identifier(self.phase_id, "phase_id")
        label = str(self.label).strip() or phase_id
        physical = str(self.physical_phase).strip() or label
        representation = str(self.cell_representation).strip() or "conventional"
        point_group = str(self.point_group).strip()
        if not point_group:
            raise input_error(f"Phase {phase_id!r}: point group must not be empty.")
        return {
            "phase_id": phase_id,
            "label": label,
            "physical_phase": physical,
            "cell_representation": representation,
            "basis_id": f"{phase_id}_basis",
            "cell": self.lattice.to_payload(),
            "point_group": point_group,
            "provenance": {
                "status": "USER_MEASURED",
                "notes": "Entered through the CuAlNi-CT public workbench.",
            },
        }


@dataclass(frozen=True)
class TransformationInput:
    transformation_id: str
    parent_phase_id: str
    product_phase_id: str
    correspondence_M_from_A: MatrixText3
    label: str = ""

    @classmethod
    def from_rows(
        cls,
        transformation_id: str,
        parent_phase_id: str,
        product_phase_id: str,
        rows: Sequence[Sequence[object]],
        *,
        label: str = "",
    ) -> "TransformationInput":
        return cls(
            transformation_id=_clean_identifier(
                transformation_id, "transformation_id"
            ),
            parent_phase_id=_clean_identifier(parent_phase_id, "parent_phase_id"),
            product_phase_id=_clean_identifier(product_phase_id, "product_phase_id"),
            correspondence_M_from_A=_matrix_text3(
                rows, name="correspondence C(M←A)"
            ),
            label=label,
        )

    def to_payload(self) -> dict[str, object]:
        transformation_id = _clean_identifier(
            self.transformation_id, "transformation_id"
        )
        parent = _clean_identifier(self.parent_phase_id, "parent_phase_id")
        product = _clean_identifier(self.product_phase_id, "product_phase_id")
        if parent == product:
            raise input_error("Parent and product phase IDs must be distinct.")
        matrix = _matrix_text3(
            self.correspondence_M_from_A,
            name="correspondence C(M←A)",
        )
        return {
            "transformation_id": transformation_id,
            "label": str(self.label).strip() or transformation_id,
            "parent_phase_id": parent,
            "product_phase_id": product,
            "correspondence_M_from_A": [list(row) for row in matrix],
            "correspondence_label": "User-entered crystallographic correspondence",
            "correspondence_source": "CuAlNi-CT public workbench",
            "correspondence_derivation": (
                "Entered explicitly by the user; no correspondence was inferred."
            ),
            "provenance": {
                "status": "USER_MEASURED",
                "notes": "User-entered transformation relation.",
            },
        }


@dataclass(frozen=True)
class CalculationRequest:
    project_id: str
    title: str
    parent: PhaseInput
    product: PhaseInput
    transformation: TransformationInput
    notes: str = ""

    def to_project_payload(self) -> dict[str, object]:
        project_id = _clean_identifier(self.project_id, "project_id")
        title = str(self.title).strip() or project_id
        parent = self.parent.to_payload()
        product = self.product.to_payload()
        if parent["phase_id"] == product["phase_id"]:
            raise input_error("Parent and product phase IDs must be distinct.")

        transformation = self.transformation.to_payload()
        if transformation["parent_phase_id"] != parent["phase_id"]:
            raise input_error(
                "Transformation parent_phase_id does not match the parent phase."
            )
        if transformation["product_phase_id"] != product["phase_id"]:
            raise input_error(
                "Transformation product_phase_id does not match the product phase."
            )

        return {
            "schema_version": SCHEMA_VERSION,
            "project_id": project_id,
            "title": title,
            "phases": [parent, product],
            "transformations": [transformation],
            "orientations": [],
            "sources": [],
            "representation_links": [],
            "notes": str(self.notes).strip(),
        }


@dataclass(frozen=True)
class CalculationResponse:
    """Strictly serializable application result plus the exact user project."""

    project_payload: dict[str, object]
    transformation_id: str
    result: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "application_schema_version": 1,
            "engine": "cualni_cryst.CalculationService",
            "transformation_id": self.transformation_id,
            "project": _json_safe(self.project_payload),
            "result": _json_safe(self.result),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    def project_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            _json_safe(self.project_payload),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )


def point_group_options() -> tuple[dict[str, object], ...]:
    """UI metadata for all conventional crystallographic point groups."""

    return point_group_table()


def _select_transformation_id(
    loaded: object,
    requested: str | None,
) -> str:
    project = loaded.project  # type: ignore[attr-defined]
    ids = tuple(item.transformation_id for item in project.transformations)
    if requested is not None and str(requested).strip():
        key = str(requested).strip()
        if key not in ids:
            raise input_error(
                f"Unknown transformation_id {key!r}. Available: {', '.join(ids) or '(none)'}."
            )
        return key
    if len(ids) == 1:
        return ids[0]
    if not ids:
        raise input_error("The project contains no transformation to calculate.")
    raise input_error(
        "The project contains multiple transformations. Select a transformation explicitly."
    )


def _serialize_bundle(
    bundle: TransformationBundle,
    service: CalculationService,
) -> dict[str, object]:
    base = bundle.to_dict()
    compatibility = bundle.am_compatibility
    ct = compatibility.ct
    analysis = ct.analysis
    metric = bundle.metric

    if compatibility.ct_exact:
        classification = "EXACT_CT_COMPATIBLE"
    elif ct.approximate.admissible_signature:
        classification = "NOT_EXACT_NEAREST_DEGENERACY_DIAGNOSTIC_AVAILABLE"
    else:
        classification = "NOT_EXACT_CT_COMPATIBLE"

    ct_detail = {
        "classification": classification,
        "exact_compatible": bool(analysis.exact_compatible),
        "degeneracy_order": int(analysis.degeneracy_order),
        "reason": analysis.reason,
        "eta_eigenvalues": analysis.eigenvalues,
        "generalized_mu": analysis.generalized_mu,
        "zero_index": analysis.zero_index,
        "nearest_zero_index": int(analysis.nearest_zero_index),
        "nearest_zero_residual": float(analysis.nearest_zero_residual),
        "inertia": analysis.inertia,
        "signature_after_nearest_zero": analysis.signature_after_nearest_zero,
        "exact_habit_planes_parent_covectors": list(ct.exact_habit_planes),
        "approximate_diagnostic": {
            "residual": float(ct.approximate.residual),
            "admissible_signature": bool(ct.approximate.admissible_signature),
            "candidate_planes_parent_covectors": list(
                ct.approximate.candidate_planes
            ),
            "explanation": ct.approximate.explanation,
        },
    }

    ball_james = {
        "lambda2_exact": bool(compatibility.ball_james_lambda2_exact),
        "solutions": [
            {
                "branch": int(solution.branch),
                "rotation": solution.rotation,
                "shape_vector_a": solution.a,
                "habit_normal_n": solution.n,
                "residual": float(solution.residual),
                "eigenvalues_C": solution.eigenvalues_C,
            }
            for solution in compatibility.ball_james_solutions
        ],
    }

    diagnostics = {
        "solver_source": analysis.solver_source,
        "precision_escalated": bool(analysis.precision_escalated),
        "generalized_eigen_residual": analysis.generalized_eigen_residual,
        "metric_orthonormality_residual": analysis.metric_orthonormality_residual,
        "route_disagreement": analysis.route_disagreement,
        "representation_parity_residual": (
            bundle.representation.maximum_parity_residual
        ),
        "exact_classification_agreement_ct_vs_ball_james": bool(
            compatibility.exact_classification_agreement
        ),
        "cache": {
            "hits": service.cache_stats.hits,
            "misses": service.cache_stats.misses,
            "entries": service.cache_stats.entries,
        },
    }

    summary = {
        "classification": classification,
        "ct_exact_compatible": bool(compatibility.ct_exact),
        "ct_reason": analysis.reason,
        "degeneracy_order": int(analysis.degeneracy_order),
        "lambda1": float(metric.principal_stretches[0]),
        "lambda2": float(metric.principal_stretches[1]),
        "lambda3": float(metric.principal_stretches[2]),
        "lambda2_residual": float(metric.lambda2_residual),
        "ct_exact_habit_plane_count": len(ct.exact_habit_planes),
        "ball_james_lambda2_exact": bool(compatibility.ball_james_lambda2_exact),
        "ball_james_solution_count": len(compatibility.ball_james_solutions),
        "variant_count": int(bundle.stretch_variants.n_variants),
        "operator_count": int(bundle.topology.n_operators),
        "topological_variant_count": int(bundle.topology.n_variants),
    }

    return _json_safe(
        {
            "summary": summary,
            **base,
            "ct_detail": ct_detail,
            "ball_james_detail": ball_james,
            "diagnostics": diagnostics,
        }
    )  # type: ignore[return-value]


def calculate_payload(
    payload: Mapping[str, object],
    *,
    transformation_id: str | None = None,
    source: str = "ui",
) -> CalculationResponse:
    """Calculate a project dictionary through the verified service boundary."""

    if not isinstance(payload, Mapping):
        raise input_error("Project payload must be a JSON object.")
    project_payload = dict(payload)

    try:
        loaded = project_from_dict(project_payload, source=source)
    except ApplicationError:
        raise
    except (TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
        raise input_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:  # defensive boundary; no silent scientific fallback
        raise internal_error(
            "The project could not be constructed.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    selected = _select_transformation_id(loaded, transformation_id)
    service = CalculationService(loaded.project)

    try:
        result = service.compute(CalculationKind.TRANSFORMATION_BUNDLE, selected)
    except ApplicationError:
        raise
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise scientific_domain_error(
            str(exc), detail=f"{type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise internal_error(
            "The verified calculation service did not complete the requested analysis.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if not isinstance(result, TransformationBundle):
        raise internal_error(
            "The calculation service returned an unexpected result type.",
            detail=type(result).__name__,
        )

    serialized = _serialize_bundle(result, service)
    return CalculationResponse(project_payload, selected, serialized)


def calculate_request(request: CalculationRequest) -> CalculationResponse:
    """Calculate a typed normal-user request through the same project pipeline."""

    try:
        payload = request.to_project_payload()
    except ApplicationError:
        raise
    except Exception as exc:
        raise input_error(str(exc), detail=f"{type(exc).__name__}: {exc}") from exc
    return calculate_payload(payload, transformation_id=request.transformation.transformation_id)
