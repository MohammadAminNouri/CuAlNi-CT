from __future__ import annotations

"""Fail-closed preflight validation for the Streamlit EBSD pipeline boundary.

The native EBSD engine remains authoritative for scientific calculations.  This
module only refuses ambiguous/incomplete UI configuration before the expensive
pipeline is entered.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class EBSDPreflightError(ValueError):
    pass


@dataclass(frozen=True)
class EBSDPreflightReport:
    status: str
    checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "checks": list(self.checks)}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EBSDPreflightError(f"{name} must be a JSON object.")
    return value


def _finite_positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EBSDPreflightError(f"{name} must be numeric.") from exc
    if not np.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise EBSDPreflightError(f"{name} must be finite and {qualifier}.")
    return number


def _vector3(value: Any, name: str, *, nonzero: bool = True) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError) as exc:
        raise EBSDPreflightError(f"{name} must contain exactly three numeric components.") from exc
    if not np.all(np.isfinite(vector)):
        raise EBSDPreflightError(f"{name} contains non-finite values.")
    if nonzero and float(np.linalg.norm(vector)) <= 1.0e-15:
        raise EBSDPreflightError(f"{name} must be non-zero.")
    return vector


def _proper_rotation(value: Any, name: str, *, tolerance: float = 2.0e-6) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float).reshape(3, 3)
    except (TypeError, ValueError) as exc:
        raise EBSDPreflightError(f"{name} must be a numeric 3×3 matrix.") from exc
    if not np.all(np.isfinite(matrix)):
        raise EBSDPreflightError(f"{name} contains non-finite values.")
    orth = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(matrix))
    if orth > tolerance or abs(determinant - 1.0) > tolerance:
        raise EBSDPreflightError(
            f"{name} is not a proper rotation within tolerance: "
            f"orthogonality residual={orth:.3e}, det={determinant:.12g}."
        )
    return matrix


def _strictly_increasing_positive(values: Any, name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EBSDPreflightError(f"{name} must be a non-empty numeric array.")
    result = tuple(_finite_positive(item, f"{name}[{index}]") for index, item in enumerate(values))
    if not result:
        raise EBSDPreflightError(f"{name} must not be empty.")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise EBSDPreflightError(f"{name} must be strictly increasing.")
    return result


def _validate_advanced_input(input_spec: Mapping[str, Any], format_name: str) -> None:
    advanced = {"csv", "tsv", "h5", "hdf5", "h5ebsd", "h5oina"}
    if format_name not in advanced:
        return
    convention = _mapping(input_spec.get("convention"), "input.convention")
    schema = _mapping(input_spec.get("schema"), "input.schema")

    sequence = str(convention.get("scipy_sequence", "")).strip()
    if not sequence:
        raise EBSDPreflightError("input.convention.scipy_sequence must be explicit.")
    angle_unit = str(convention.get("angle_unit", "")).strip().lower()
    if angle_unit not in {"degree", "degrees", "radian", "radians"}:
        raise EBSDPreflightError("input.convention.angle_unit must explicitly be degree or radian.")
    direction = str(convention.get("raw_matrix_direction", "")).strip().lower()
    if direction not in {"crystal_to_sample", "sample_to_crystal"}:
        raise EBSDPreflightError(
            "input.convention.raw_matrix_direction must be crystal_to_sample or sample_to_crystal."
        )
    _proper_rotation(convention.get("sample_correction"), "input.convention.sample_correction")
    _proper_rotation(convention.get("crystal_correction"), "input.convention.crystal_correction")

    for required in ("phase", "x", "y"):
        if schema.get(required) in (None, ""):
            raise EBSDPreflightError(f"input.schema.{required} must be explicit.")
    representations = (schema.get("euler"), schema.get("quaternion"), schema.get("matrix"))
    if all(value in (None, [], "") for value in representations):
        raise EBSDPreflightError(
            "input.schema must declare at least one orientation representation: euler, quaternion or matrix."
        )


def validate_ebsd_run_config(config: Mapping[str, Any]) -> EBSDPreflightReport:
    if not isinstance(config, Mapping):
        raise EBSDPreflightError("EBSD configuration must be a JSON object.")
    checks: list[str] = []

    input_spec = _mapping(config.get("input"), "input")
    format_name = str(input_spec.get("format", "")).strip().lower()
    supported = {"ang", "ctf", "csv", "tsv", "h5", "hdf5", "h5ebsd", "h5oina"}
    if format_name not in supported:
        raise EBSDPreflightError(f"Unsupported or missing input.format: {format_name!r}.")
    if not str(input_spec.get("path", "")).strip():
        raise EBSDPreflightError("input.path must be explicit.")
    if format_name == "ctf" and "three_dimensional_angle_unit" in input_spec:
        unit = str(input_spec["three_dimensional_angle_unit"]).strip().lower()
        if unit not in {"degree", "radian"}:
            raise EBSDPreflightError("input.three_dimensional_angle_unit must be degree or radian.")
    _validate_advanced_input(input_spec, format_name)
    checks.append("input format / orientation convention / schema explicit")

    phases = config.get("phases")
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)) or len(phases) < 2:
        raise EBSDPreflightError("phases must contain at least parent and product phase definitions.")
    phase_ids: set[int] = set()
    for index, raw in enumerate(phases):
        phase = _mapping(raw, f"phases[{index}]")
        try:
            pid = int(phase.get("id"))
        except (TypeError, ValueError) as exc:
            raise EBSDPreflightError(f"phases[{index}].id must be an integer.") from exc
        if pid in phase_ids:
            raise EBSDPreflightError(f"Duplicate EBSD phase ID {pid} in phases.")
        phase_ids.add(pid)
        if not str(phase.get("point_group", "")).strip():
            raise EBSDPreflightError(f"phases[{index}].point_group must be explicit.")
        lattice = _mapping(phase.get("lattice"), f"phases[{index}].lattice")
        if not str(lattice.get("length_unit", "")).strip():
            raise EBSDPreflightError(f"phases[{index}].lattice.length_unit must be explicit.")
        for key in ("a", "b", "c"):
            _finite_positive(lattice.get(key), f"phases[{index}].lattice.{key}")
        for key in ("alpha_deg", "beta_deg", "gamma_deg"):
            angle = _finite_positive(lattice.get(key), f"phases[{index}].lattice.{key}")
            if angle >= 180.0:
                raise EBSDPreflightError(f"phases[{index}].lattice.{key} must be < 180°.")

    try:
        parent_id = int(config.get("parent_phase_id"))
        product_id = int(config.get("product_phase_id"))
    except (TypeError, ValueError) as exc:
        raise EBSDPreflightError("parent_phase_id and product_phase_id must be explicit integers.") from exc
    if parent_id == product_id:
        raise EBSDPreflightError("Parent and product EBSD phase IDs must be distinct.")
    if parent_id not in phase_ids or product_id not in phase_ids:
        raise EBSDPreflightError("Parent/product phase IDs must reference entries in phases.")
    checks.append("phase mapping explicit and parent/product IDs distinct")

    orientation = _mapping(config.get("orientation_relationship"), "orientation_relationship")
    if str(orientation.get("mode", "")).strip().lower() != "matrix":
        raise EBSDPreflightError("orientation_relationship.mode must explicitly be matrix for this UI route.")
    if str(orientation.get("matrix_direction", "")).strip().lower() != "parent_from_product":
        raise EBSDPreflightError(
            "orientation_relationship.matrix_direction must be parent_from_product."
        )
    _proper_rotation(orientation.get("matrix"), "orientation_relationship.matrix")
    checks.append("initial OR direction and SO(3) validity explicit")

    quality = config.get("quality_filters", [])
    if not isinstance(quality, list):
        raise EBSDPreflightError("quality_filters must be a JSON array.")
    if any(not isinstance(item, Mapping) for item in quality):
        raise EBSDPreflightError("Every quality filter must be a JSON object.")
    checks.append("quality filtering explicit; absent rules mean no quality field is guessed")

    segmentation = _mapping(config.get("segmentation"), "segmentation")
    _finite_positive(segmentation.get("main_threshold_deg"), "segmentation.main_threshold_deg")
    _strictly_increasing_positive(
        segmentation.get("sweep_thresholds_deg"), "segmentation.sweep_thresholds_deg"
    )
    if int(segmentation.get("minimum_grain_points", 0)) < 1:
        raise EBSDPreflightError("segmentation.minimum_grain_points must be >= 1.")
    _finite_positive(segmentation.get("neighbor_radius_factor"), "segmentation.neighbor_radius_factor")
    _finite_positive(
        segmentation.get("kam_max_neighbor_misorientation_deg"),
        "segmentation.kam_max_neighbor_misorientation_deg",
    )
    checks.append("segmentation and sensitivity sweep explicit")

    refinement = _mapping(config.get("or_refinement"), "or_refinement")
    refine_enabled = bool(refinement.get("enabled", False))
    use_refined = bool(refinement.get("use_if_accepted", False))
    if use_refined and not refine_enabled:
        raise EBSDPreflightError("A refined OR cannot be used when OR refinement is disabled.")
    _finite_positive(refinement.get("maximum_correction_deg"), "or_refinement.maximum_correction_deg")
    trim = _finite_positive(refinement.get("trim_fraction"), "or_refinement.trim_fraction")
    if trim > 1.0:
        raise EBSDPreflightError("or_refinement.trim_fraction must be <= 1.")
    _finite_positive(
        refinement.get("minimum_improvement_deg2", 0.0),
        "or_refinement.minimum_improvement_deg2",
        allow_zero=True,
    )
    checks.append("OR refinement is bounded and opt-in")

    reconstruction = _mapping(config.get("parent_reconstruction"), "parent_reconstruction")
    if bool(reconstruction.get("enabled", False)):
        _finite_positive(reconstruction.get("link_tolerance_deg"), "parent_reconstruction.link_tolerance_deg")
        _finite_positive(
            reconstruction.get("reconstruction_tolerance_deg"),
            "parent_reconstruction.reconstruction_tolerance_deg",
        )
        if int(reconstruction.get("minimum_grains", 0)) < 1:
            raise EBSDPreflightError("parent_reconstruction.minimum_grains must be >= 1.")
    checks.append("parent reconstruction tolerances explicit")

    boundary = _mapping(config.get("boundary_classification"), "boundary_classification")
    _finite_positive(boundary.get("maximum_residual_deg"), "boundary_classification.maximum_residual_deg")
    _finite_positive(
        boundary.get("minimum_margin_deg", 0.0),
        "boundary_classification.minimum_margin_deg",
        allow_zero=True,
    )
    checks.append("boundary acceptance and ambiguity margin explicit")

    trace = _mapping(config.get("trace_validation", {"enabled": False}), "trace_validation")
    if bool(trace.get("enabled", False)):
        _vector3(trace.get("surface_normal_sample"), "trace_validation.surface_normal_sample")
        linearity = _finite_positive(
            trace.get("minimum_linearity"), "trace_validation.minimum_linearity", allow_zero=True
        )
        if linearity > 1.0:
            raise EBSDPreflightError("trace_validation.minimum_linearity must be <= 1.")
        hypotheses = trace.get("hypotheses")
        if not isinstance(hypotheses, list) or not hypotheses:
            raise EBSDPreflightError(
                "Trace validation requires at least one explicit plane hypothesis."
            )
        for index, raw in enumerate(hypotheses):
            hypothesis = _mapping(raw, f"trace_validation.hypotheses[{index}]")
            if not str(hypothesis.get("label", "")).strip():
                raise EBSDPreflightError(
                    f"trace_validation.hypotheses[{index}].label must be explicit."
                )
            _vector3(
                hypothesis.get("plane_side1_crystal"),
                f"trace_validation.hypotheses[{index}].plane_side1_crystal",
            )
            _vector3(
                hypothesis.get("plane_side2_crystal"),
                f"trace_validation.hypotheses[{index}].plane_side2_crystal",
            )
        checks.append("trace analysis gated by explicit non-zero surface normal and plane hypotheses")
    else:
        checks.append("trace analysis disabled; no surface normal is inferred")

    return EBSDPreflightReport(status="ready", checks=tuple(checks))


def validate_ebsd_config_file(path: str | Path) -> EBSDPreflightReport:
    source = Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EBSDPreflightError(f"EBSD configuration could not be read: {exc}") from exc
    if not isinstance(config, Mapping):
        raise EBSDPreflightError("EBSD configuration top level must be a JSON object.")
    return validate_ebsd_run_config(config)
