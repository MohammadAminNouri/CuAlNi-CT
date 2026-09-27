
from __future__ import annotations

"""Systematic CT-supercompatibility versus cofactor-condition atlas.

The atlas does not assert a universal equivalence theorem.  Each supplied
ProjectState is solved independently by the existing CT and Ball--James/cofactor
backends and classified only after those calculations have completed.
"""

from dataclasses import dataclass, replace
from enum import Enum
from itertools import product as cartesian_product
from typing import Any, Iterable, Mapping

import numpy as np

from .lattice import Lattice
from .point_groups import resolve_point_group
from .theory_equivalence import AssignmentScales, MatchFamily, PhysicalBranchMatcher
from .theory_unified import (
    PTMCMode,
    PredictionKind,
    TheoryComparisonAdapter,
    TheoryKind,
)


class AtlasClass(str, Enum):
    BOTH = "both"
    CT_ONLY = "ct_only"
    COFACTOR_ONLY = "cofactor_only"
    NEITHER = "neither"
    NOT_EVALUABLE = "not_evaluable"


@dataclass(frozen=True)
class AtlasPairResult:
    """One physically matched CT M/M relation versus one Ball--James/cofactor relation."""

    ct_row_id: str
    ball_james_row_id: str
    ct_operator_index: int | None
    ct_twin_index: int | None
    ct_best_habit_index: int | None
    ct_supercompatibility_residual: float | None
    ct_supercompatible: bool
    cofactor_cc1_abs: float | None
    cofactor_cc2_abs: float | None
    cofactor_cc3_margin: float | None
    cofactor_compatible: bool
    twin_plane_angle_deg: float | None
    twin_shear_relative: float | None
    match_completeness: str
    missing_required_components: tuple[str, ...]
    classification: AtlasClass

    def to_dict(self) -> dict[str, Any]:
        return {
            "ct_row_id": self.ct_row_id,
            "ball_james_row_id": self.ball_james_row_id,
            "ct_operator_index": self.ct_operator_index,
            "ct_twin_index": self.ct_twin_index,
            "ct_best_habit_index": self.ct_best_habit_index,
            "ct_supercompatibility_residual": self.ct_supercompatibility_residual,
            "ct_supercompatible": self.ct_supercompatible,
            "cofactor_cc1_abs": self.cofactor_cc1_abs,
            "cofactor_cc2_abs": self.cofactor_cc2_abs,
            "cofactor_cc3_margin": self.cofactor_cc3_margin,
            "cofactor_compatible": self.cofactor_compatible,
            "twin_plane_angle_deg": self.twin_plane_angle_deg,
            "twin_shear_relative": self.twin_shear_relative,
            "match_completeness": self.match_completeness,
            "missing_required_components": list(self.missing_required_components),
            "classification": self.classification.value,
        }


@dataclass(frozen=True)
class AtlasStateResult:
    state_id: str
    classification: AtlasClass
    ct_am_exact_compatible: bool | None
    ct_supercompatible: bool | None
    cofactor_compatible: bool | None
    ct_best_supercompatibility_residual: float | None
    cofactor_best_cc1_abs: float | None
    cofactor_best_cc2_abs: float | None
    cofactor_best_cc3_margin: float | None
    ct_supercompatible_branch_count: int
    cofactor_compatible_branch_count: int
    matched_relation_count: int
    pair_agreement_count: int
    pair_disagreement_count: int
    pair_results: tuple[AtlasPairResult, ...]
    metadata: dict[str, Any]
    error: str = ""

    @property
    def matched_pair_counts(self) -> dict[str, int]:
        return {
            item.value: sum(pair.classification is item for pair in self.pair_results)
            for item in (
                AtlasClass.BOTH,
                AtlasClass.CT_ONLY,
                AtlasClass.COFACTOR_ONLY,
                AtlasClass.NEITHER,
            )
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "classification": self.classification.value,
            "ct_am_exact_compatible": self.ct_am_exact_compatible,
            "ct_supercompatible": self.ct_supercompatible,
            "cofactor_compatible": self.cofactor_compatible,
            "ct_best_supercompatibility_residual": (
                self.ct_best_supercompatibility_residual
            ),
            "cofactor_best_cc1_abs": self.cofactor_best_cc1_abs,
            "cofactor_best_cc2_abs": self.cofactor_best_cc2_abs,
            "cofactor_best_cc3_margin": self.cofactor_best_cc3_margin,
            "ct_supercompatible_branch_count": self.ct_supercompatible_branch_count,
            "cofactor_compatible_branch_count": self.cofactor_compatible_branch_count,
            "matched_relation_count": self.matched_relation_count,
            "pair_agreement_count": self.pair_agreement_count,
            "pair_disagreement_count": self.pair_disagreement_count,
            "matched_pair_counts": self.matched_pair_counts,
            "pair_results": [item.to_dict() for item in self.pair_results],
            "metadata": dict(self.metadata),
            "error": self.error,
        }


@dataclass(frozen=True)
class SupercompatibilityAtlasReport:
    transformation_id: str
    states: tuple[AtlasStateResult, ...]
    notes: tuple[str, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            item.value: sum(state.classification is item for state in self.states)
            for item in AtlasClass
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "counts": self.counts,
            "states": [state.to_dict() for state in self.states],
            "notes": list(self.notes),
        }


def _finite_min(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return min(finite) if finite else None


def _finite_max(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return max(finite) if finite else None


def _classification(ct_value: bool, cofactor_value: bool) -> AtlasClass:
    if ct_value and cofactor_value:
        return AtlasClass.BOTH
    if ct_value:
        return AtlasClass.CT_ONLY
    if cofactor_value:
        return AtlasClass.COFACTOR_ONLY
    return AtlasClass.NEITHER


def _matched_pair_results(
    project: Any,
    transformation_id: str,
    report: Any,
    *,
    tolerance: float,
) -> tuple[AtlasPairResult, ...]:
    """Compare CT shear/shear closure and cofactor conditions on matched M/M relations."""
    matcher = PhysicalBranchMatcher(project, transformation_id)
    ct_twins = [
        row
        for row in report.rows
        if row.prediction_kind is PredictionKind.CT_MM_TWIN
    ]
    bj_twins = [
        row
        for row in report.rows
        if row.prediction_kind is PredictionKind.BALL_JAMES_MM
    ]
    matches, _ = matcher.match_pair(
        ct_twins,
        bj_twins,
        family=MatchFamily.MM_TWIN,
        scales=AssignmentScales(),
    )

    by_id = {row.row_id: row for row in report.rows}
    super_rows = [
        row
        for row in report.rows
        if row.prediction_kind is PredictionKind.CT_SUPERCOMPATIBILITY
    ]
    output: list[AtlasPairResult] = []
    for match in matches:
        ct_row = by_id[match.left_row_id]
        bj_row = by_id[match.right_row_id]
        operator_index = ct_row.metadata.get("operator_index")
        twin_index = ct_row.metadata.get("twin_index")
        candidates = [
            row
            for row in super_rows
            if row.metadata.get("operator_index") == operator_index
            and row.metadata.get("twin_index") == twin_index
            and row.residuals.get("ct_supercompatibility_dimensionless") is not None
        ]
        best = (
            min(
                candidates,
                key=lambda row: abs(
                    float(row.residuals["ct_supercompatibility_dimensionless"])
                ),
            )
            if candidates
            else None
        )
        ct_residual = (
            None
            if best is None
            else abs(float(best.residuals["ct_supercompatibility_dimensionless"]))
        )
        ct_ok = ct_residual is not None and ct_residual <= tolerance
        bj_ok = bool(bj_row.metadata.get("cofactor_all_satisfied", False))
        cc1 = bj_row.residuals.get("cc1_lambda2_minus_one")
        cc2 = bj_row.residuals.get("cc2")
        cc3 = bj_row.residuals.get("cc3_margin")
        output.append(
            AtlasPairResult(
                ct_row_id=ct_row.row_id,
                ball_james_row_id=bj_row.row_id,
                ct_operator_index=(
                    None if operator_index is None else int(operator_index)
                ),
                ct_twin_index=None if twin_index is None else int(twin_index),
                ct_best_habit_index=(
                    None if best is None else int(best.metadata["habit_index"])
                ),
                ct_supercompatibility_residual=ct_residual,
                ct_supercompatible=bool(ct_ok),
                cofactor_cc1_abs=None if cc1 is None else abs(float(cc1)),
                cofactor_cc2_abs=None if cc2 is None else abs(float(cc2)),
                cofactor_cc3_margin=None if cc3 is None else float(cc3),
                cofactor_compatible=bj_ok,
                twin_plane_angle_deg=match.residuals.twin_plane_angle_deg,
                twin_shear_relative=match.residuals.shear_relative,
                match_completeness=match.completeness.value,
                missing_required_components=match.missing_required_components,
                classification=_classification(bool(ct_ok), bj_ok),
            )
        )
    return tuple(output)


def evaluate_supercompatibility_state(
    project: Any,
    transformation_id: str,
    *,
    state_id: str = "",
    metadata: dict[str, Any] | None = None,
    ct_tolerance: float | None = None,
) -> AtlasStateResult:
    """Evaluate one state without using one theory to construct the other."""

    project.validate().assert_passed()
    tolerance = (
        max(project.numerical_policy.rank_one, project.numerical_policy.algebraic)
        if ct_tolerance is None
        else float(ct_tolerance)
    )
    if tolerance <= 0.0:
        raise ValueError("ct_tolerance must be positive")

    report = TheoryComparisonAdapter(project, transformation_id).compare(
        ptmc_mode=PTMCMode.NONE,
        include_ct_closing_gap=False,
        include_ct_supercompatibility=True,
        ball_james_fraction_samples=101,
    )

    ct_super_rows = [
        row
        for row in report.rows
        if row.prediction_kind is PredictionKind.CT_SUPERCOMPATIBILITY
    ]
    ct_residuals = [
        row.residuals.get("ct_supercompatibility_dimensionless")
        for row in ct_super_rows
    ]
    ct_satisfied_rows = [
        row
        for row in ct_super_rows
        if row.residuals.get("ct_supercompatibility_dimensionless") is not None
        and abs(float(row.residuals["ct_supercompatibility_dimensionless"]))
        <= tolerance
    ]

    cofactor_rows = [
        row
        for row in report.rows
        if row.prediction_kind is PredictionKind.BALL_JAMES_MM
    ]
    cofactor_satisfied_rows = [
        row
        for row in cofactor_rows
        if bool(row.metadata.get("cofactor_all_satisfied", False))
    ]

    pair_results = _matched_pair_results(
        project,
        transformation_id,
        report,
        tolerance=tolerance,
    )
    # State-level set membership remains descriptive, while pair_results records
    # whether the same physically matched M/M relation agrees or disagrees.
    ct_super = bool(ct_satisfied_rows)
    cofactor = bool(cofactor_satisfied_rows)
    classification = _classification(ct_super, cofactor)

    cc1 = [
        abs(float(row.residuals["cc1_lambda2_minus_one"]))
        for row in cofactor_rows
        if "cc1_lambda2_minus_one" in row.residuals
    ]
    cc2 = [
        abs(float(row.residuals["cc2"]))
        for row in cofactor_rows
        if "cc2" in row.residuals
    ]
    cc3 = [
        float(row.residuals["cc3_margin"])
        for row in cofactor_rows
        if "cc3_margin" in row.residuals
    ]

    return AtlasStateResult(
        state_id=state_id or project.project_id,
        classification=classification,
        ct_am_exact_compatible=bool(report.ct_report.analysis.exact_compatible),
        ct_supercompatible=ct_super,
        cofactor_compatible=cofactor,
        ct_best_supercompatibility_residual=_finite_min(ct_residuals),
        cofactor_best_cc1_abs=_finite_min(cc1),
        cofactor_best_cc2_abs=_finite_min(cc2),
        cofactor_best_cc3_margin=_finite_max(cc3),
        ct_supercompatible_branch_count=len(ct_satisfied_rows),
        cofactor_compatible_branch_count=len(cofactor_satisfied_rows),
        matched_relation_count=len(pair_results),
        pair_agreement_count=sum(
            item.ct_supercompatible == item.cofactor_compatible
            for item in pair_results
        ),
        pair_disagreement_count=sum(
            item.ct_supercompatible != item.cofactor_compatible
            for item in pair_results
        ),
        pair_results=pair_results,
        metadata={} if metadata is None else dict(metadata),
    )


def evaluate_supercompatibility_atlas(
    states: Iterable[tuple[str, Any, dict[str, Any]]],
    transformation_id: str,
    *,
    ct_tolerance: float | None = None,
    preserve_failures: bool = True,
) -> SupercompatibilityAtlasReport:
    """Evaluate an explicit deterministic family of ProjectState objects.

    ``states`` contains ``(state_id, project, metadata)`` tuples.  The atlas
    never perturbs a project silently.
    """

    output: list[AtlasStateResult] = []
    for state_id, project, metadata in states:
        try:
            output.append(
                evaluate_supercompatibility_state(
                    project,
                    transformation_id,
                    state_id=state_id,
                    metadata=metadata,
                    ct_tolerance=ct_tolerance,
                )
            )
        except Exception as exc:
            if not preserve_failures:
                raise
            output.append(
                AtlasStateResult(
                    state_id=state_id,
                    classification=AtlasClass.NOT_EVALUABLE,
                    ct_am_exact_compatible=None,
                    ct_supercompatible=None,
                    cofactor_compatible=None,
                    ct_best_supercompatibility_residual=None,
                    cofactor_best_cc1_abs=None,
                    cofactor_best_cc2_abs=None,
                    cofactor_best_cc3_margin=None,
                    ct_supercompatible_branch_count=0,
                    cofactor_compatible_branch_count=0,
                    matched_relation_count=0,
                    pair_agreement_count=0,
                    pair_disagreement_count=0,
                    pair_results=(),
                    metadata=dict(metadata),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return SupercompatibilityAtlasReport(
        transformation_id=transformation_id,
        states=tuple(output),
        notes=(
            "CT supercompatibility and cofactor conditions are evaluated independently.",
            "The four-way state classification is descriptive; it is not an equivalence theorem.",
            (
                "pair_results uses the globally assigned common M/M observables to pair "
                "CT and Ball-James relations. Because CT reference-configuration twin "
                "directions and Ball-James current-configuration shear directions are not "
                "directly comparable, those pairings are explicitly marked partial; full "
                "CT/Mallard/Ball-James M/M equivalence remains the independent audit."
            ),
            "NOT_EVALUABLE states are retained rather than removed from the sweep.",
        ),
    )


def uniform_product_scale_sweep(
    project: Any,
    transformation_id: str,
    scale_factors: Iterable[float],
) -> tuple[tuple[str, Any, dict[str, Any]], ...]:
    """Build symmetry-preserving states by uniformly scaling product lengths.

    Uniform scaling preserves all cell angles and the point-group metric
    constraints.  It is therefore a safe generic one-parameter family for the
    app.  More specialized studies should construct explicit ProjectState
    families and pass them to :func:`evaluate_supercompatibility_atlas`.
    """

    transformation = project.transformation(transformation_id)
    product = project.phase(transformation.product_phase_id)
    output = []
    for raw in scale_factors:
        scale = float(raw)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale_factors must be finite and positive")
        old = product.lattice
        lattice = replace(
            old,
            a=old.a * scale,
            b=old.b * scale,
            c=old.c * scale,
            label=(old.label + f" [uniform scale {scale:.12g}]").strip(),
        )
        scaled_product = replace(product, lattice=lattice)
        phases = tuple(
            scaled_product if phase.phase_id == product.phase_id else phase
            for phase in project.phases
        )
        scaled_project = replace(
            project,
            project_id=f"{project.project_id}__product_scale_{scale:.12g}",
            phases=phases,
        )
        scaled_project.validate().assert_passed()
        output.append(
            (
                f"scale={scale:.12g}",
                scaled_project,
                {
                    "sweep_parameter": "uniform_product_length_scale",
                    "scale_factor": scale,
                    "product_a": lattice.a,
                    "product_b": lattice.b,
                    "product_c": lattice.c,
                    "alpha_deg": lattice.alpha_deg,
                    "beta_deg": lattice.beta_deg,
                    "gamma_deg": lattice.gamma_deg,
                },
            )
        )
    return tuple(output)


def product_lattice_grid(
    project: Any,
    transformation_id: str,
    *,
    a_scales: Iterable[float] = (1.0,),
    b_scales: Iterable[float] = (1.0,),
    c_scales: Iterable[float] = (1.0,),
    alpha_offsets_deg: Iterable[float] = (0.0,),
    beta_offsets_deg: Iterable[float] = (0.0,),
    gamma_offsets_deg: Iterable[float] = (0.0,),
    max_states: int = 2000,
) -> tuple[tuple[str, Any, dict[str, Any]], ...]:
    """Build an explicit deterministic product-cell parameter grid.

    This helper never relaxes point-group/metric validation.  A grid point that
    violates the registered phase symmetry is left as an invalid ProjectState
    and will be reported as ``NOT_EVALUABLE`` by
    :func:`evaluate_supercompatibility_atlas` when ``preserve_failures=True``.
    That behaviour is intentional: the atlas must not silently alter a user's
    requested state to recover symmetry.
    """

    axes = [
        tuple(float(x) for x in a_scales),
        tuple(float(x) for x in b_scales),
        tuple(float(x) for x in c_scales),
        tuple(float(x) for x in alpha_offsets_deg),
        tuple(float(x) for x in beta_offsets_deg),
        tuple(float(x) for x in gamma_offsets_deg),
    ]
    if any(not values for values in axes):
        raise ValueError("Every product-lattice grid axis must contain at least one value")
    if any(
        not np.isfinite(value)
        for values in axes
        for value in values
    ):
        raise ValueError("Product-lattice grid values must be finite")
    if any(value <= 0.0 for values in axes[:3] for value in values):
        raise ValueError("Length scale factors must be positive")

    count = int(np.prod([len(values) for values in axes]))
    if count > int(max_states):
        raise ValueError(
            f"Requested product-lattice grid has {count} states; max_states={max_states}"
        )

    transformation = project.transformation(transformation_id)
    product_phase = project.phase(transformation.product_phase_id)
    old = product_phase.lattice
    output = []
    for a_scale, b_scale, c_scale, da, db, dg in cartesian_product(*axes):
        lattice = replace(
            old,
            a=old.a * a_scale,
            b=old.b * b_scale,
            c=old.c * c_scale,
            alpha_deg=old.alpha_deg + da,
            beta_deg=old.beta_deg + db,
            gamma_deg=old.gamma_deg + dg,
            label=(old.label + " [compatibility atlas grid]").strip(),
        )
        changed_product = replace(product_phase, lattice=lattice)
        phases = tuple(
            changed_product if phase.phase_id == product_phase.phase_id else phase
            for phase in project.phases
        )
        state_id = (
            f"a={a_scale:.9g},b={b_scale:.9g},c={c_scale:.9g},"
            f"da={da:.9g},db={db:.9g},dg={dg:.9g}"
        )
        changed_project = replace(
            project,
            project_id=f"{project.project_id}__atlas__{len(output):04d}",
            phases=phases,
        )
        output.append(
            (
                state_id,
                changed_project,
                {
                    "a_scale": a_scale,
                    "b_scale": b_scale,
                    "c_scale": c_scale,
                    "alpha_offset_deg": da,
                    "beta_offset_deg": db,
                    "gamma_offset_deg": dg,
                    "product_a": lattice.a,
                    "product_b": lattice.b,
                    "product_c": lattice.c,
                    "product_alpha_deg": lattice.alpha_deg,
                    "product_beta_deg": lattice.beta_deg,
                    "product_gamma_deg": lattice.gamma_deg,
                },
            )
        )
    return tuple(output)


_PARAMETER_NEUTRAL: dict[str, float] = {
    "a_scale": 1.0,
    "b_scale": 1.0,
    "c_scale": 1.0,
    "alpha_offset_deg": 0.0,
    "beta_offset_deg": 0.0,
    "gamma_offset_deg": 0.0,
}


def _registered_point_group_definition(phase: Any):
    """Resolve a PhaseState point-group label without discarding setting metadata.

    Modern ProjectState/JSON phases store canonical Hermann--Mauguin symbols
    (for example ``2/m``).  Some truth-locked legacy/reference phases predate
    that registry and include the conventional setting in the same display
    string, for example ``2/m (unique b)``.

    We accept that older form *only* when the parenthetical suffix exactly
    matches the registry's conventional setting for the resolved base symbol.
    This is intentionally stricter than blindly stripping parentheses: an
    inconsistent or genuinely non-standard setting must continue to fail
    loudly instead of being assigned the wrong crystal family.
    """

    raw = str(phase.point_group_symbol).strip()
    try:
        return resolve_point_group(raw)
    except ValueError as original_error:
        if not raw.endswith(")") or "(" not in raw:
            raise

        base, suffix = raw.rsplit("(", 1)
        base = base.strip()
        suffix = suffix[:-1].strip()
        if not base or not suffix:
            raise original_error

        try:
            definition = resolve_point_group(base)
        except ValueError:
            raise original_error

        normalize = lambda text: " ".join(str(text).casefold().split())
        if normalize(suffix) != normalize(definition.conventional_setting):
            raise ValueError(
                "Point-group label contains a setting annotation that does not "
                "match the registered conventional setting: "
                f"label={raw!r}, registered_setting={definition.conventional_setting!r}. "
                "Refusing to infer a crystal family from an inconsistent setting."
            ) from original_error
        return definition


def independent_product_lattice_parameters(
    project: Any,
    transformation_id: str,
) -> tuple[str, ...]:
    """Independent conventional-cell parameters for the registered product setting.

    The returned names are deliberately phrased as scale factors / angle offsets
    so a sweep is defined relative to the frozen ProjectState rather than by
    silently replacing its absolute lattice constants.
    """

    transformation = project.transformation(transformation_id)
    phase = project.phase(transformation.product_phase_id)
    family = _registered_point_group_definition(phase).crystal_family
    if family == "cubic":
        return ("a_scale",)
    if family in {"tetragonal", "hexagonal", "trigonal"}:
        return ("a_scale", "c_scale")
    if family == "orthorhombic":
        return ("a_scale", "b_scale", "c_scale")
    if family == "monoclinic":
        return ("a_scale", "b_scale", "c_scale", "beta_offset_deg")
    if family == "triclinic":
        return (
            "a_scale",
            "b_scale",
            "c_scale",
            "alpha_offset_deg",
            "beta_offset_deg",
            "gamma_offset_deg",
        )
    raise ValueError(f"Unsupported product crystal family {family!r}")


def symmetry_preserving_product_grid(
    project: Any,
    transformation_id: str,
    axes: Mapping[str, Iterable[float]],
    *,
    max_states: int = 2000,
) -> tuple[tuple[str, Any, dict[str, Any]], ...]:
    """Build a deterministic cell grid while preserving conventional symmetry constraints.

    ``axes`` may contain only parameters returned by
    :func:`independent_product_lattice_parameters`.  Omitted independent
    parameters remain at their neutral values (length scale 1, angle offset 0).
    Derived parameters are reconstructed from the crystal family, so a cubic
    sweep can never accidentally make ``a != b`` and a monoclinic unique-b
    sweep can never alter alpha or gamma away from 90 degrees.
    """

    if max_states < 1:
        raise ValueError("max_states must be positive")
    allowed = independent_product_lattice_parameters(project, transformation_id)
    unknown = sorted(set(axes).difference(allowed))
    if unknown:
        raise ValueError(
            f"Parameters {unknown} are not independent for this product setting; "
            f"allowed={list(allowed)}"
        )

    values: dict[str, tuple[float, ...]] = {}
    for name in allowed:
        raw = tuple(float(value) for value in axes.get(name, (_PARAMETER_NEUTRAL[name],)))
        if not raw:
            raise ValueError(f"Grid axis {name!r} is empty")
        if any(not np.isfinite(value) for value in raw):
            raise ValueError(f"Grid axis {name!r} contains a non-finite value")
        if name.endswith("_scale") and any(value <= 0.0 for value in raw):
            raise ValueError(f"Grid axis {name!r} must contain positive scale factors")
        values[name] = raw

    state_count = int(np.prod([len(values[name]) for name in allowed], dtype=int))
    if state_count > max_states:
        raise ValueError(
            f"Requested symmetry-preserving grid has {state_count} states; "
            f"max_states={max_states}"
        )

    transformation = project.transformation(transformation_id)
    product_phase = project.phase(transformation.product_phase_id)
    family = _registered_point_group_definition(product_phase).crystal_family
    old = product_phase.lattice
    output: list[tuple[str, Any, dict[str, Any]]] = []

    for coordinate in cartesian_product(*(values[name] for name in allowed)):
        point = dict(_PARAMETER_NEUTRAL)
        point.update(dict(zip(allowed, coordinate, strict=True)))
        a_scale = point["a_scale"]
        b_scale = point["b_scale"]
        c_scale = point["c_scale"]
        da = point["alpha_offset_deg"]
        db = point["beta_offset_deg"]
        dg = point["gamma_offset_deg"]

        if family == "cubic":
            a = old.a * a_scale
            lattice = replace(old, a=a, b=a, c=a, alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0)
        elif family == "tetragonal":
            a = old.a * a_scale
            lattice = replace(old, a=a, b=a, c=old.c * c_scale, alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0)
        elif family in {"hexagonal", "trigonal"}:
            a = old.a * a_scale
            lattice = replace(old, a=a, b=a, c=old.c * c_scale, alpha_deg=90.0, beta_deg=90.0, gamma_deg=120.0)
        elif family == "orthorhombic":
            lattice = replace(old, a=old.a * a_scale, b=old.b * b_scale, c=old.c * c_scale, alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0)
        elif family == "monoclinic":
            lattice = replace(old, a=old.a * a_scale, b=old.b * b_scale, c=old.c * c_scale, alpha_deg=90.0, beta_deg=old.beta_deg + db, gamma_deg=90.0)
        elif family == "triclinic":
            lattice = replace(
                old,
                a=old.a * a_scale,
                b=old.b * b_scale,
                c=old.c * c_scale,
                alpha_deg=old.alpha_deg + da,
                beta_deg=old.beta_deg + db,
                gamma_deg=old.gamma_deg + dg,
            )
        else:
            raise AssertionError(f"Unhandled crystal family {family!r}")

        changed_product = replace(product_phase, lattice=lattice)
        phases = tuple(
            changed_product if phase.phase_id == product_phase.phase_id else phase
            for phase in project.phases
        )
        changed_project = replace(
            project,
            project_id=f"{project.project_id}__compat_atlas_{len(output):05d}",
            phases=phases,
        )
        metadata = {
            "crystal_family": family,
            **{name: float(point[name]) for name in allowed},
            "product_a": lattice.a,
            "product_b": lattice.b,
            "product_c": lattice.c,
            "product_alpha_deg": lattice.alpha_deg,
            "product_beta_deg": lattice.beta_deg,
            "product_gamma_deg": lattice.gamma_deg,
        }
        state_id = ",".join(f"{name}={point[name]:.9g}" for name in allowed)
        output.append((state_id, changed_project, metadata))

    return tuple(output)
