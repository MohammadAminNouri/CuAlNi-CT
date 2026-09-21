from __future__ import annotations

"""Reproducible end-to-end execution of the frozen EBSD crystallography stack.

This module is deliberately orchestration-only.  It does not reimplement CT,
orientation topology, weak-twin equations, EBSD disorientation, OR refinement,
or parent reconstruction.

Scientific contract
-------------------
A run is controlled by one strict JSON configuration.  The pipeline never
guesses:

* parent/product phase IDs;
* lattice parameters or point groups;
* Euler/matrix/quaternion direction conventions;
* reference-frame corrections;
* EBSD quality cutoffs;
* the grain-segmentation threshold;
* which projective OR branch is intended;
* whether an experimentally refined OR should replace the supplied OR;
* a specimen surface normal;
* how overlapping parent-domain candidates should be resolved.

Ambiguity is retained in the output instead of being collapsed to one answer.

The output directory is written through a staging directory and promoted only
after every requested calculation and file write succeeds.
"""

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from fractions import Fraction
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .ebsd_analysis import (
    Grain,
    GrainSegmentation,
    build_neighbor_graph,
    grain_statistics,
    kernel_average_misorientation,
    segment_grains,
)
from .ebsd_io import (
    DelimitedSchema,
    HDF5Schema,
    load_ang,
    load_ctf,
    load_delimited,
    load_hdf5_explicit,
)
from .ebsd_map import (
    AngleUnit,
    EBSDMap,
    EBSDPhase,
    MatrixDirection,
    OrientationConvention,
    audit_map,
)
from .ebsd_phase_forensics import (
    InsufficientPhaseEvidenceError,
    assess_grain_route_reliability,
)
from .ebsd_reconstruction import (
    VariantAssignment,
    reconstruct_parent,
)
from .ebsd_theory_bridge import (
    TheoryLibrary,
    build_theory_library,
)
from .ebsd_trace_validation import (
    BoundaryTraceEstimate,
    TwinPlaneHypothesis,
    estimate_boundary_trace,
    rank_trace_hypotheses,
)
from .ebsd_validation import (
    ConventionHypothesisRanking,
    ORRefinementResult,
    PreparedBoundaryOperatorKernel,
    SegmentationSweep,
    TheoryConsistencyScore,
    VariantGraphReport,
    rank_orientation_hypotheses,
    reconstruct_variant_graph_candidates,
    refine_orientation_relationship_from_child_boundaries,
    score_theory_consistency,
    sweep_segmentation_thresholds,
    unique_grain_adjacency,
)
from .lattice import Lattice
from .orientation_kernel import (
    OrientationKernel,
    require_so3,
)
from .point_groups import point_group_operations


SCHEMA_VERSION = 1


class PipelineConfigError(ValueError):
    pass


def _reject_unknown(
    mapping: Mapping[str, Any],
    allowed: set[str],
    *,
    context: str,
) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise PipelineConfigError(
            f"{context} contains unknown keys: {unknown}. "
            "Configuration is strict so misspellings cannot be ignored."
        )


def _require_keys(
    mapping: Mapping[str, Any],
    required: set[str],
    *,
    context: str,
) -> None:
    missing = sorted(required.difference(mapping))
    if missing:
        raise PipelineConfigError(
            f"{context} is missing required keys: {missing}"
        )


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise PipelineConfigError(f"{name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            try:
                result = float(Fraction(value.strip()))
            except (ValueError, ZeroDivisionError) as exc:
                raise PipelineConfigError(
                    f"{name} must be numeric or a rational string such as '1/2'"
                ) from exc
        else:
            raise PipelineConfigError(f"{name} must be numeric")
    if not math.isfinite(result):
        raise PipelineConfigError(f"{name} must be finite")
    return result


def _positive(value: Any, *, name: str) -> float:
    result = _number(value, name=name)
    if result <= 0.0:
        raise PipelineConfigError(f"{name} must be positive")
    return result


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise PipelineConfigError(f"{name} must be an integer, not boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineConfigError(f"{name} must be an integer") from exc
    if float(result) != float(value):
        raise PipelineConfigError(f"{name} must be an exact integer")
    if minimum is not None and result < minimum:
        raise PipelineConfigError(f"{name} must be >= {minimum}")
    return result


def _numeric_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Coerce a numeric/rational nested sequence to a finite ndarray.

    JSON naturally supplies lists, while internal defaults/tests may use tuples
    or NumPy arrays.  All are accepted if and only if the exact requested shape
    is present.  Rational strings such as ``"1/2"`` are handled by `_number`.
    """

    if isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, (list, tuple)):
        raw = value
    else:
        raise PipelineConfigError(
            f"{name} must have exact shape {shape}; got {type(value).__name__}"
        )

    def convert(item: Any, depth: int, prefix: str) -> Any:
        if depth == len(shape):
            return _number(item, name=prefix)
        if not isinstance(item, (list, tuple)) or len(item) != shape[depth]:
            raise PipelineConfigError(
                f"{name} must have exact shape {shape}"
            )
        return [
            convert(child, depth + 1, f"{prefix}[{index}]")
            for index, child in enumerate(item)
        ]

    array = np.asarray(convert(raw, 0, name), dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise PipelineConfigError(
            f"{name} must be finite with exact shape {shape}"
        )
    return array


def _vector3(value: Any, *, name: str) -> np.ndarray:
    array = _numeric_array(value, shape=(3,), name=name)
    if float(np.linalg.norm(array)) <= 1.0e-15:
        raise PipelineConfigError(f"{name} must be nonzero")
    return array


def _matrix3(value: Any, *, name: str) -> np.ndarray:
    return _numeric_array(value, shape=(3, 3), name=name)

def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        # Do not use dataclasses.asdict here: it deep-copies field values and
        # MappingProxyType is intentionally used by the EBSD data model.
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    tmp.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(tmp, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _git_revision() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            result = subprocess.run(
                list(args),
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.strip()
        except Exception:
            return None
        return result or None

    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "describe": command("git", "describe", "--tags", "--always", "--dirty"),
    }


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    try:
        config = json.loads(config_path.read_text())
    except FileNotFoundError:
        raise PipelineConfigError(f"configuration file not found: {config_path}")
    except json.JSONDecodeError as exc:
        raise PipelineConfigError(
            f"invalid JSON configuration at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    if not isinstance(config, dict):
        raise PipelineConfigError("top-level configuration must be a JSON object")

    allowed = {
        "schema_version",
        "input",
        "phases",
        "parent_phase_id",
        "product_phase_id",
        "orientation_relationship",
        "quality_filters",
        "segmentation",
        "theory",
        "or_refinement",
        "parent_reconstruction",
        "boundary_classification",
        "trace_validation",
        "convention_hypotheses",
        "output",
    }
    required = {
        "schema_version",
        "input",
        "phases",
        "parent_phase_id",
        "product_phase_id",
        "orientation_relationship",
        "segmentation",
        "output",
    }
    _reject_unknown(config, allowed, context="root")
    _require_keys(config, required, context="root")

    version = _integer(
        config["schema_version"], name="schema_version", minimum=1
    )
    if version != SCHEMA_VERSION:
        raise PipelineConfigError(
            f"unsupported schema_version={version}; expected {SCHEMA_VERSION}"
        )

    return config, config_path


def _resolve_relative(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def lattice_from_config(spec: Mapping[str, Any], *, context: str) -> Lattice:
    allowed = {
        "a",
        "b",
        "c",
        "alpha_deg",
        "beta_deg",
        "gamma_deg",
        "length_unit",
        "label",
    }
    required = {"a", "b", "c"}
    _reject_unknown(spec, allowed, context=context)
    _require_keys(spec, required, context=context)
    return Lattice(
        a=_positive(spec["a"], name=f"{context}.a"),
        b=_positive(spec["b"], name=f"{context}.b"),
        c=_positive(spec["c"], name=f"{context}.c"),
        alpha_deg=_number(spec.get("alpha_deg", 90.0), name=f"{context}.alpha_deg"),
        beta_deg=_number(spec.get("beta_deg", 90.0), name=f"{context}.beta_deg"),
        gamma_deg=_number(spec.get("gamma_deg", 90.0), name=f"{context}.gamma_deg"),
        label=str(spec.get("label", "")),
        length_unit=str(spec.get("length_unit", "")),
    )


def phases_from_config(
    entries: Any,
) -> dict[int, EBSDPhase]:
    if not isinstance(entries, list) or not entries:
        raise PipelineConfigError("phases must be a nonempty JSON array")

    phases: dict[int, EBSDPhase] = {}
    for index, entry in enumerate(entries):
        context = f"phases[{index}]"
        if not isinstance(entry, dict):
            raise PipelineConfigError(f"{context} must be an object")
        allowed = {"id", "name", "point_group", "lattice"}
        required = allowed
        _reject_unknown(entry, allowed, context=context)
        _require_keys(entry, required, context=context)
        phase_id = _integer(entry["id"], name=f"{context}.id", minimum=1)
        if phase_id in phases:
            raise PipelineConfigError(f"duplicate phase id {phase_id}")
        lattice_spec = entry["lattice"]
        if not isinstance(lattice_spec, dict):
            raise PipelineConfigError(f"{context}.lattice must be an object")
        phases[phase_id] = EBSDPhase.from_point_group(
            phase_id=phase_id,
            name=str(entry["name"]),
            lattice=lattice_from_config(
                lattice_spec, context=f"{context}.lattice"
            ),
            point_group=str(entry["point_group"]),
        )
    return phases


def orientation_convention_from_config(
    spec: Mapping[str, Any] | None,
    *,
    required: bool,
    context: str,
) -> OrientationConvention | None:
    if spec is None:
        if required:
            raise PipelineConfigError(
                f"{context} is required for this input format"
            )
        return None
    if not isinstance(spec, Mapping):
        raise PipelineConfigError(f"{context} must be an object")

    allowed = {
        "scipy_sequence",
        "angle_unit",
        "raw_matrix_direction",
        "sample_correction",
        "crystal_correction",
        "label",
    }
    _reject_unknown(spec, allowed, context=context)

    sequence = str(spec.get("scipy_sequence", "ZXZ"))
    try:
        # Static validation of the exact SciPy Euler convention before a large
        # map is touched.  This catches invalid axis strings, mixed intrinsic /
        # extrinsic cases, and repeated-consecutive-axis errors immediately.
        Rotation.from_euler(sequence, [0.0, 0.0, 0.0], degrees=True)
    except ValueError as exc:
        raise PipelineConfigError(
            f"{context}.scipy_sequence={sequence!r} is not a valid "
            "SciPy Euler sequence"
        ) from exc

    try:
        angle_unit = AngleUnit(str(spec.get("angle_unit", "degree")))
    except ValueError as exc:
        raise PipelineConfigError(
            f"{context}.angle_unit must be 'degree' or 'radian'"
        ) from exc

    try:
        direction = MatrixDirection(
            str(spec.get("raw_matrix_direction", "crystal_to_sample"))
        )
    except ValueError as exc:
        raise PipelineConfigError(
            f"{context}.raw_matrix_direction must be "
            "'crystal_to_sample' or 'sample_to_crystal'"
        ) from exc

    # Use an explicit immutable identity default, not an ndarray-only special
    # case.  `_matrix3` nevertheless accepts list/tuple/ndarray uniformly.
    identity = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    sample = _matrix3(
        spec.get("sample_correction", identity),
        name=f"{context}.sample_correction",
    )
    crystal = _matrix3(
        spec.get("crystal_correction", identity),
        name=f"{context}.crystal_correction",
    )

    try:
        return OrientationConvention(
            scipy_sequence=sequence,
            angle_unit=angle_unit,
            raw_matrix_direction=direction,
            sample_correction=tuple(
                tuple(float(x) for x in row) for row in sample
            ),
            crystal_correction=tuple(
                tuple(float(x) for x in row) for row in crystal
            ),
            label=str(spec.get("label", "explicit pipeline convention")),
        )
    except ValueError as exc:
        # Configuration errors must not leak backend ValueErrors: callers need
        # one stable exception type for malformed scientific input.
        raise PipelineConfigError(
            f"{context} is not a physically valid orientation convention: {exc}"
        ) from exc

def validate_input_config_static(
    spec: Mapping[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    """Validate input format/convention/schema without opening the EBSD file."""

    if not isinstance(spec, Mapping):
        raise PipelineConfigError("input must be an object")

    allowed = {
        "path",
        "format",
        "convention",
        "schema",
        "separator",
        "three_dimensional_angle_unit",
    }
    required = {"path", "format"}
    _reject_unknown(spec, allowed, context="input")
    _require_keys(spec, required, context="input")

    path_text = str(spec["path"]).strip()
    if not path_text:
        raise PipelineConfigError("input.path must not be empty")
    path = _resolve_relative(path_text, base_dir)
    format_name = str(spec["format"]).lower()

    convention_spec = spec.get("convention")
    if convention_spec is not None and not isinstance(convention_spec, dict):
        raise PipelineConfigError("input.convention must be an object")

    if format_name == "ang":
        orientation_convention_from_config(
            convention_spec,
            required=False,
            context="input.convention",
        )
        if spec.get("schema") is not None:
            raise PipelineConfigError(
                "input.schema is not used for native ANG input"
            )
        return {"path": path, "format": format_name}

    if format_name == "ctf":
        orientation_convention_from_config(
            convention_spec,
            required=False,
            context="input.convention",
        )
        unit = spec.get("three_dimensional_angle_unit")
        if unit is not None:
            try:
                AngleUnit(str(unit))
            except ValueError as exc:
                raise PipelineConfigError(
                    "input.three_dimensional_angle_unit must be "
                    "'degree' or 'radian'"
                ) from exc
        if spec.get("schema") is not None:
            raise PipelineConfigError(
                "input.schema is not used for native CTF input"
            )
        return {"path": path, "format": format_name}

    if format_name in {"csv", "tsv", "delimited"}:
        orientation_convention_from_config(
            convention_spec,
            required=True,
            context="input.convention",
        )
        schema_spec = spec.get("schema")
        if not isinstance(schema_spec, dict):
            raise PipelineConfigError(
                "input.schema is required for delimited input"
            )
        allowed_schema = {
            "phase",
            "x",
            "y",
            "z",
            "indexed",
            "euler",
            "quaternion",
            "matrix",
            "quaternion_order",
            "quality",
        }
        _reject_unknown(
            schema_spec, allowed_schema, context="input.schema"
        )
        _require_keys(
            schema_spec, {"phase", "x", "y"}, context="input.schema"
        )
        representations = [
            schema_spec.get("euler"),
            schema_spec.get("quaternion"),
            schema_spec.get("matrix"),
        ]
        if sum(item is not None for item in representations) != 1:
            raise PipelineConfigError(
                "input.schema must specify exactly one of "
                "euler/quaternion/matrix"
            )
        if schema_spec.get("euler") is not None:
            value = schema_spec["euler"]
            if not isinstance(value, list) or len(value) != 3:
                raise PipelineConfigError(
                    "input.schema.euler must name exactly 3 columns"
                )
        if schema_spec.get("quaternion") is not None:
            value = schema_spec["quaternion"]
            if not isinstance(value, list) or len(value) != 4:
                raise PipelineConfigError(
                    "input.schema.quaternion must name exactly 4 columns"
                )
        if schema_spec.get("matrix") is not None:
            value = schema_spec["matrix"]
            if not isinstance(value, list) or len(value) != 9:
                raise PipelineConfigError(
                    "input.schema.matrix must name exactly 9 columns"
                )
        quality = schema_spec.get("quality", {})
        if not isinstance(quality, dict):
            raise PipelineConfigError(
                "input.schema.quality must be an object"
            )
        return {"path": path, "format": format_name}

    if format_name in {"h5", "hdf5", "h5ebsd", "h5oina"}:
        orientation_convention_from_config(
            convention_spec,
            required=True,
            context="input.convention",
        )
        schema_spec = spec.get("schema")
        if not isinstance(schema_spec, dict):
            raise PipelineConfigError(
                "input.schema is required for explicit HDF5 input"
            )
        allowed_schema = {
            "phase",
            "x",
            "y",
            "z",
            "indexed",
            "euler",
            "quaternion",
            "matrix",
            "quaternion_order",
            "quality",
        }
        _reject_unknown(
            schema_spec, allowed_schema, context="input.schema"
        )
        _require_keys(
            schema_spec, {"phase", "x", "y"}, context="input.schema"
        )
        representations = [
            schema_spec.get("euler"),
            schema_spec.get("quaternion"),
            schema_spec.get("matrix"),
        ]
        if sum(item is not None for item in representations) != 1:
            raise PipelineConfigError(
                "input.schema must specify exactly one HDF5 orientation "
                "representation: euler/quaternion/matrix"
            )
        if schema_spec.get("euler") is not None:
            value = schema_spec["euler"]
            if not isinstance(value, list) or len(value) != 3:
                raise PipelineConfigError(
                    "input.schema.euler must name exactly 3 HDF5 datasets"
                )
        if schema_spec.get("quaternion") is not None and not isinstance(
            schema_spec["quaternion"], str
        ):
            raise PipelineConfigError(
                "input.schema.quaternion must be one HDF5 dataset path"
            )
        if schema_spec.get("matrix") is not None and not isinstance(
            schema_spec["matrix"], str
        ):
            raise PipelineConfigError(
                "input.schema.matrix must be one HDF5 dataset path"
            )
        quality = schema_spec.get("quality", {})
        if not isinstance(quality, dict):
            raise PipelineConfigError(
                "input.schema.quality must be an object"
            )
        return {"path": path, "format": format_name}

    raise PipelineConfigError(
        "input.format must be one of: ang, ctf, csv, tsv, delimited, "
        "h5, hdf5, h5ebsd, h5oina"
    )

def load_ebsd_from_config(
    spec: Mapping[str, Any],
    *,
    base_dir: Path,
) -> tuple[EBSDMap, Path]:
    validate_input_config_static(spec, base_dir=base_dir)

    allowed = {
        "path",
        "format",
        "convention",
        "schema",
        "separator",
        "three_dimensional_angle_unit",
    }
    required = {"path", "format"}
    _reject_unknown(spec, allowed, context="input")
    _require_keys(spec, required, context="input")

    path = _resolve_relative(str(spec["path"]), base_dir)
    if not path.is_file():
        raise PipelineConfigError(f"input EBSD file not found: {path}")

    format_name = str(spec["format"]).lower()
    convention_spec = spec.get("convention")
    if convention_spec is not None and not isinstance(convention_spec, dict):
        raise PipelineConfigError("input.convention must be an object")

    if format_name == "ang":
        convention = orientation_convention_from_config(
            convention_spec,
            required=False,
            context="input.convention",
        )
        return load_ang(path, convention=convention), path

    if format_name == "ctf":
        convention = orientation_convention_from_config(
            convention_spec,
            required=False,
            context="input.convention",
        )
        unit = spec.get("three_dimensional_angle_unit")
        parsed_unit = None
        if unit is not None:
            try:
                parsed_unit = AngleUnit(str(unit))
            except ValueError as exc:
                raise PipelineConfigError(
                    "input.three_dimensional_angle_unit must be "
                    "'degree' or 'radian'"
                ) from exc
        return (
            load_ctf(
                path,
                convention=convention,
                three_dimensional_angle_unit=parsed_unit,
            ),
            path,
        )

    if format_name in {"csv", "tsv", "delimited"}:
        convention = orientation_convention_from_config(
            convention_spec,
            required=True,
            context="input.convention",
        )
        assert convention is not None
        schema_spec = spec.get("schema")
        if not isinstance(schema_spec, dict):
            raise PipelineConfigError(
                "input.schema is required for delimited input"
            )
        allowed_schema = {
            "phase",
            "x",
            "y",
            "z",
            "indexed",
            "euler",
            "quaternion",
            "matrix",
            "quaternion_order",
            "quality",
        }
        _reject_unknown(
            schema_spec, allowed_schema, context="input.schema"
        )
        _require_keys(
            schema_spec, {"phase", "x", "y"}, context="input.schema"
        )

        quality = schema_spec.get("quality", {})
        if not isinstance(quality, dict):
            raise PipelineConfigError("input.schema.quality must be an object")

        schema = DelimitedSchema(
            phase=str(schema_spec["phase"]),
            x=str(schema_spec["x"]),
            y=str(schema_spec["y"]),
            z=(
                str(schema_spec["z"])
                if schema_spec.get("z") is not None
                else None
            ),
            indexed=(
                str(schema_spec["indexed"])
                if schema_spec.get("indexed") is not None
                else None
            ),
            euler=(
                tuple(str(x) for x in schema_spec["euler"])
                if schema_spec.get("euler") is not None
                else None
            ),
            quaternion=(
                tuple(str(x) for x in schema_spec["quaternion"])
                if schema_spec.get("quaternion") is not None
                else None
            ),
            matrix=(
                tuple(str(x) for x in schema_spec["matrix"])
                if schema_spec.get("matrix") is not None
                else None
            ),
            quaternion_order=str(
                schema_spec.get("quaternion_order", "wxyz")
            ),
            quality={str(k): str(v) for k, v in quality.items()},
        )
        separator = spec.get("separator")
        if format_name == "tsv" and separator is None:
            separator = "\t"
        return (
            load_delimited(
                path,
                schema,
                convention=convention,
                separator=(str(separator) if separator is not None else None),
            ),
            path,
        )

    if format_name in {"h5", "hdf5", "h5ebsd", "h5oina"}:
        convention = orientation_convention_from_config(
            convention_spec,
            required=True,
            context="input.convention",
        )
        assert convention is not None
        schema_spec = spec.get("schema")
        if not isinstance(schema_spec, dict):
            raise PipelineConfigError(
                "input.schema is required for explicit HDF5 input"
            )
        allowed_schema = {
            "phase",
            "x",
            "y",
            "z",
            "indexed",
            "euler",
            "quaternion",
            "matrix",
            "quaternion_order",
            "quality",
        }
        _reject_unknown(
            schema_spec, allowed_schema, context="input.schema"
        )
        _require_keys(
            schema_spec, {"phase", "x", "y"}, context="input.schema"
        )
        quality = schema_spec.get("quality", {})
        if not isinstance(quality, dict):
            raise PipelineConfigError("input.schema.quality must be an object")

        schema = HDF5Schema(
            phase=str(schema_spec["phase"]),
            x=str(schema_spec["x"]),
            y=str(schema_spec["y"]),
            z=(
                str(schema_spec["z"])
                if schema_spec.get("z") is not None
                else None
            ),
            indexed=(
                str(schema_spec["indexed"])
                if schema_spec.get("indexed") is not None
                else None
            ),
            euler=(
                tuple(str(x) for x in schema_spec["euler"])
                if schema_spec.get("euler") is not None
                else None
            ),
            quaternion=(
                str(schema_spec["quaternion"])
                if schema_spec.get("quaternion") is not None
                else None
            ),
            matrix=(
                str(schema_spec["matrix"])
                if schema_spec.get("matrix") is not None
                else None
            ),
            quaternion_order=str(
                schema_spec.get("quaternion_order", "wxyz")
            ),
            quality={str(k): str(v) for k, v in quality.items()},
        )
        return (
            load_hdf5_explicit(
                path,
                schema,
                convention=convention,
            ),
            path,
        )

    raise PipelineConfigError(
        "input.format must be one of: ang, ctf, csv, tsv, delimited, "
        "h5, hdf5, h5ebsd, h5oina"
    )


def validate_quality_filter_rules_static(
    rules: Any,
) -> tuple[dict[str, Any], ...]:
    """Validate rule syntax without requiring a concrete vendor quality field."""

    if rules is None:
        return tuple()
    if not isinstance(rules, list):
        raise PipelineConfigError("quality_filters must be an array")

    parsed: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        context = f"quality_filters[{index}]"
        if not isinstance(rule, Mapping):
            raise PipelineConfigError(f"{context} must be an object")
        allowed = {"field", "op", "value"}
        _reject_unknown(rule, allowed, context=context)
        _require_keys(rule, allowed, context=context)

        field = str(rule["field"]).strip()
        if not field:
            raise PipelineConfigError(
                f"{context}.field must be a nonempty quality-field name"
            )
        op = str(rule["op"])
        if op not in {">=", ">", "<=", "<", "==", "!="}:
            raise PipelineConfigError(
                f"{context}.op must be one of >=, >, <=, <, ==, !="
            )
        parsed.append(
            {
                "field": field,
                "op": op,
                "value": _number(rule["value"], name=f"{context}.value"),
            }
        )
    return tuple(parsed)


@dataclass(frozen=True)
class QualityFilterAudit:
    input_indexed_points: int
    retained_indexed_points: int
    rejected_indexed_points: int
    retained_fraction_of_indexed: float
    rules: tuple[dict[str, Any], ...]


def apply_quality_filters(
    data: EBSDMap,
    rules: Any,
) -> tuple[EBSDMap, QualityFilterAudit]:
    parsed_rules = validate_quality_filter_rules_static(rules)

    mask = data.indexed.copy()
    audited_rules: list[dict[str, Any]] = []

    for index, rule in enumerate(parsed_rules):
        context = f"quality_filters[{index}]"
        field = rule["field"]
        if field not in data.quality:
            raise PipelineConfigError(
                f"{context}.field={field!r} is not present. "
                f"Available quality fields: {sorted(data.quality)}"
            )
        values = np.asarray(data.quality[field], dtype=float)
        threshold = float(rule["value"])
        op = rule["op"]
        finite = np.isfinite(values)

        if op == ">=":
            local = finite & (values >= threshold)
        elif op == ">":
            local = finite & (values > threshold)
        elif op == "<=":
            local = finite & (values <= threshold)
        elif op == "<":
            local = finite & (values < threshold)
        elif op == "==":
            local = finite & (values == threshold)
        else:
            assert op == "!="
            local = finite & (values != threshold)

        mask &= local
        audited_rules.append(dict(rule))

    orientations = data.orientations.copy()
    phase_id = data.phase_id.copy()
    orientations[~mask] = np.nan
    phase_id[~mask] = 0

    metadata = dict(data.metadata)
    metadata["pipeline_quality_filters"] = tuple(
        json.dumps(rule, sort_keys=True) for rule in audited_rules
    )

    filtered = EBSDMap(
        orientations=orientations,
        phase_id=phase_id,
        indexed=mask,
        x=data.x.copy(),
        y=data.y.copy(),
        z=data.z.copy(),
        quality={
            key: np.asarray(value).copy()
            for key, value in data.quality.items()
        },
        metadata=metadata,
    )

    input_count = int(np.count_nonzero(data.indexed))
    retained_count = int(np.count_nonzero(mask))
    return filtered, QualityFilterAudit(
        input_indexed_points=input_count,
        retained_indexed_points=retained_count,
        rejected_indexed_points=input_count - retained_count,
        retained_fraction_of_indexed=(
            retained_count / input_count if input_count else 0.0
        ),
        rules=tuple(audited_rules),
    )


@dataclass(frozen=True)
class ResolvedOR:
    R_parent_from_product: np.ndarray
    source_mode: str
    selected_candidate_index: int | None
    candidate_count: int
    candidate_residuals_deg: tuple[tuple[float, float], ...]
    details: Mapping[str, Any]


def resolve_orientation_relationship(
    spec: Mapping[str, Any],
    *,
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
) -> ResolvedOR:
    allowed = {
        "mode",
        "matrix",
        "matrix_direction",
        "parallelisms",
        "candidate_index",
        "parallelism_tolerance_deg",
    }
    _reject_unknown(spec, allowed, context="orientation_relationship")
    _require_keys(spec, {"mode"}, context="orientation_relationship")
    mode = str(spec["mode"])

    if mode == "matrix":
        _require_keys(
            spec, {"matrix"}, context="orientation_relationship"
        )
        matrix = require_so3(
            _matrix3(
                spec["matrix"],
                name="orientation_relationship.matrix",
            ),
            tolerance=2.0e-8,
            name="configured OR matrix",
        )
        direction = str(
            spec.get("matrix_direction", "parent_from_product")
        )
        if direction == "parent_from_product":
            resolved = matrix
        elif direction == "product_from_parent":
            resolved = matrix.T
        else:
            raise PipelineConfigError(
                "orientation_relationship.matrix_direction must be "
                "'parent_from_product' or 'product_from_parent'"
            )
        return ResolvedOR(
            R_parent_from_product=resolved,
            source_mode="matrix",
            selected_candidate_index=None,
            candidate_count=1,
            candidate_residuals_deg=((0.0, 0.0),),
            details={"matrix_direction": direction},
        )

    if mode != "parallelisms":
        raise PipelineConfigError(
            "orientation_relationship.mode must be 'matrix' or 'parallelisms'"
        )

    relations = spec.get("parallelisms")
    if not isinstance(relations, list) or len(relations) != 2:
        raise PipelineConfigError(
            "orientation_relationship.parallelisms must contain exactly "
            "two independent crystallographic relations"
        )

    parsed = []
    for index, relation in enumerate(relations):
        context = f"orientation_relationship.parallelisms[{index}]"
        if not isinstance(relation, dict):
            raise PipelineConfigError(f"{context} must be an object")
        allowed_relation = {"parent", "product", "projective"}
        _reject_unknown(
            relation, allowed_relation, context=context
        )
        _require_keys(
            relation, {"parent", "product"}, context=context
        )

        sides = []
        for side_name in ("parent", "product"):
            side = relation[side_name]
            if not isinstance(side, dict):
                raise PipelineConfigError(
                    f"{context}.{side_name} must be an object"
                )
            _reject_unknown(
                side, {"kind", "indices"}, context=f"{context}.{side_name}"
            )
            _require_keys(
                side, {"kind", "indices"}, context=f"{context}.{side_name}"
            )
            kind = str(side["kind"])
            if kind not in {"direction", "plane"}:
                raise PipelineConfigError(
                    f"{context}.{side_name}.kind must be 'direction' or 'plane'"
                )
            indices = _vector3(
                side["indices"],
                name=f"{context}.{side_name}.indices",
            )
            sides.append((kind, indices))
        parsed.append(
            {
                "parent_kind": sides[0][0],
                "parent_indices": sides[0][1],
                "product_kind": sides[1][0],
                "product_indices": sides[1][1],
                "projective": bool(relation.get("projective", True)),
            }
        )

    kernel = OrientationKernel(
        parent_phase.lattice.metric(),
        product_phase.lattice.metric(),
        point_group_operations(parent_phase.point_group),
        point_group_operations(product_phase.point_group),
    )
    tolerance = _positive(
        spec.get("parallelism_tolerance_deg", 1.0e-7),
        name="orientation_relationship.parallelism_tolerance_deg",
    )
    report = kernel.parallelism_orientations(
        parsed[0]["parent_indices"],
        parsed[0]["product_indices"],
        parsed[1]["parent_indices"],
        parsed[1]["product_indices"],
        reference_first_kind=parsed[0]["parent_kind"],
        moving_first_kind=parsed[0]["product_kind"],
        reference_second_kind=parsed[1]["parent_kind"],
        moving_second_kind=parsed[1]["product_kind"],
        projective_first=parsed[0]["projective"],
        projective_second=parsed[1]["projective"],
        tolerance_deg=tolerance,
    )

    if len(report.candidates) == 1:
        selected_index = 0
    else:
        if "candidate_index" not in spec:
            raise PipelineConfigError(
                f"the two projective parallelisms generated "
                f"{len(report.candidates)} proper OR branches. "
                "Set orientation_relationship.candidate_index explicitly; "
                "the pipeline will not choose a branch from experimental fit "
                "without being asked."
            )
        selected_index = _integer(
            spec["candidate_index"],
            name="orientation_relationship.candidate_index",
            minimum=0,
        )
        if selected_index >= len(report.candidates):
            raise PipelineConfigError(
                "orientation_relationship.candidate_index is out of range: "
                f"{selected_index} >= {len(report.candidates)}"
            )

    selected = report.candidates[selected_index]
    return ResolvedOR(
        R_parent_from_product=selected.R_A_from_M,
        source_mode="parallelisms",
        selected_candidate_index=selected_index,
        candidate_count=len(report.candidates),
        candidate_residuals_deg=tuple(
            (
                float(candidate.first_residual_deg),
                float(candidate.second_residual_deg),
            )
            for candidate in report.candidates
        ),
        details={
            "reference_internal_angle_deg": report.reference_internal_angle_deg,
            "product_internal_angle_deg": report.moving_internal_angle_deg,
            "projective_first": report.projective_first,
            "projective_second": report.projective_second,
        },
    )


def _parse_segmentation(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {
        "main_threshold_deg",
        "sweep_thresholds_deg",
        "minimum_grain_points",
        "neighbor_radius",
        "neighbor_radius_factor",
        "kam_max_neighbor_misorientation_deg",
    }
    required = {"main_threshold_deg", "sweep_thresholds_deg"}
    _reject_unknown(spec, allowed, context="segmentation")
    _require_keys(spec, required, context="segmentation")

    sweep = [
        _positive(value, name="segmentation.sweep_thresholds_deg[]")
        for value in spec["sweep_thresholds_deg"]
    ]
    if not sweep:
        raise PipelineConfigError(
            "segmentation.sweep_thresholds_deg must not be empty"
        )
    if any(second <= first for first, second in zip(sweep, sweep[1:])):
        raise PipelineConfigError(
            "segmentation.sweep_thresholds_deg must be strictly increasing"
        )

    radius = spec.get("neighbor_radius")
    return {
        "main_threshold_deg": _positive(
            spec["main_threshold_deg"],
            name="segmentation.main_threshold_deg",
        ),
        "sweep_thresholds_deg": sweep,
        "minimum_grain_points": _integer(
            spec.get("minimum_grain_points", 1),
            name="segmentation.minimum_grain_points",
            minimum=1,
        ),
        "neighbor_radius": (
            _positive(radius, name="segmentation.neighbor_radius")
            if radius is not None
            else None
        ),
        "neighbor_radius_factor": _positive(
            spec.get("neighbor_radius_factor", 1.15),
            name="segmentation.neighbor_radius_factor",
        ),
        "kam_max_neighbor_misorientation_deg": (
            None
            if spec.get("kam_max_neighbor_misorientation_deg", 5.0) is None
            else _positive(
                spec.get("kam_max_neighbor_misorientation_deg", 5.0),
                name="segmentation.kam_max_neighbor_misorientation_deg",
            )
        ),
    }


def _parse_theory(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    if spec is None:
        spec = {}
    allowed = {
        "quotient_tolerance_deg",
        "operator_equivalence_tolerance_deg",
        "crosscheck_topology",
    }
    _reject_unknown(spec, allowed, context="theory")
    return {
        "quotient_tolerance_deg": _positive(
            spec.get("quotient_tolerance_deg", 2.0e-7),
            name="theory.quotient_tolerance_deg",
        ),
        "operator_equivalence_tolerance_deg": _positive(
            spec.get("operator_equivalence_tolerance_deg", 2.0e-6),
            name="theory.operator_equivalence_tolerance_deg",
        ),
        "crosscheck_topology": bool(spec.get("crosscheck_topology", True)),
    }


def _parse_refinement(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    if spec is None:
        spec = {}
    allowed = {
        "enabled",
        "use_if_accepted",
        "maximum_correction_deg",
        "trim_fraction",
        "huber_delta_deg",
        "minimum_improvement_deg2",
        "multi_start_step_deg",
        "maximum_iterations",
    }
    _reject_unknown(spec, allowed, context="or_refinement")
    trim = _number(
        spec.get("trim_fraction", 0.65),
        name="or_refinement.trim_fraction",
    )
    if not 0.0 < trim <= 1.0:
        raise PipelineConfigError(
            "or_refinement.trim_fraction must lie in (0,1]"
        )
    return {
        "enabled": bool(spec.get("enabled", False)),
        "use_if_accepted": bool(spec.get("use_if_accepted", False)),
        "maximum_correction_deg": _positive(
            spec.get("maximum_correction_deg", 5.0),
            name="or_refinement.maximum_correction_deg",
        ),
        "trim_fraction": trim,
        "huber_delta_deg": _positive(
            spec.get("huber_delta_deg", 2.0),
            name="or_refinement.huber_delta_deg",
        ),
        "minimum_improvement_deg2": _positive(
            spec.get("minimum_improvement_deg2", 0.02),
            name="or_refinement.minimum_improvement_deg2",
        ),
        "multi_start_step_deg": _positive(
            spec.get("multi_start_step_deg", 0.75),
            name="or_refinement.multi_start_step_deg",
        ),
        "maximum_iterations": _integer(
            spec.get("maximum_iterations", 120),
            name="or_refinement.maximum_iterations",
            minimum=1,
        ),
    }


def _parse_parent_reconstruction(
    spec: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if spec is None:
        spec = {}
    allowed = {
        "enabled",
        "link_tolerance_deg",
        "reconstruction_tolerance_deg",
        "minimum_grains",
    }
    _reject_unknown(spec, allowed, context="parent_reconstruction")
    return {
        "enabled": bool(spec.get("enabled", True)),
        "link_tolerance_deg": _positive(
            spec.get("link_tolerance_deg", 3.0),
            name="parent_reconstruction.link_tolerance_deg",
        ),
        "reconstruction_tolerance_deg": _positive(
            spec.get("reconstruction_tolerance_deg", 3.0),
            name="parent_reconstruction.reconstruction_tolerance_deg",
        ),
        "minimum_grains": _integer(
            spec.get("minimum_grains", 2),
            name="parent_reconstruction.minimum_grains",
            minimum=2,
        ),
    }


def _parse_boundary_classification(
    spec: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if spec is None:
        spec = {}
    allowed = {"maximum_residual_deg", "minimum_margin_deg"}
    _reject_unknown(spec, allowed, context="boundary_classification")
    margin = _number(
        spec.get("minimum_margin_deg", 0.5),
        name="boundary_classification.minimum_margin_deg",
    )
    if margin < 0.0:
        raise PipelineConfigError(
            "boundary_classification.minimum_margin_deg must be >= 0"
        )
    return {
        "maximum_residual_deg": _positive(
            spec.get("maximum_residual_deg", 3.0),
            name="boundary_classification.maximum_residual_deg",
        ),
        "minimum_margin_deg": margin,
    }


def _parse_trace(
    spec: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if spec is None:
        spec = {}
    allowed = {
        "enabled",
        "surface_normal_sample",
        "minimum_linearity",
        "hypotheses",
    }
    _reject_unknown(spec, allowed, context="trace_validation")
    enabled = bool(spec.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "surface_normal_sample": None,
            "minimum_linearity": 0.90,
            "hypotheses": tuple(),
        }

    _require_keys(
        spec,
        {"surface_normal_sample", "hypotheses"},
        context="trace_validation",
    )
    minimum_linearity = _number(
        spec.get("minimum_linearity", 0.90),
        name="trace_validation.minimum_linearity",
    )
    if not 0.0 <= minimum_linearity <= 1.0:
        raise PipelineConfigError(
            "trace_validation.minimum_linearity must lie in [0,1]"
        )
    hypotheses_spec = spec["hypotheses"]
    if not isinstance(hypotheses_spec, list) or not hypotheses_spec:
        raise PipelineConfigError(
            "trace_validation.hypotheses must be a nonempty array"
        )
    hypotheses = []
    for index, item in enumerate(hypotheses_spec):
        context = f"trace_validation.hypotheses[{index}]"
        if not isinstance(item, dict):
            raise PipelineConfigError(f"{context} must be an object")
        allowed_item = {
            "label",
            "plane_side1_crystal",
            "plane_side2_crystal",
            "operator_index",
            "source",
        }
        required_item = {
            "label",
            "plane_side1_crystal",
            "plane_side2_crystal",
        }
        _reject_unknown(item, allowed_item, context=context)
        _require_keys(item, required_item, context=context)
        hypotheses.append(
            TwinPlaneHypothesis(
                label=str(item["label"]),
                plane_side1_crystal=_vector3(
                    item["plane_side1_crystal"],
                    name=f"{context}.plane_side1_crystal",
                ),
                plane_side2_crystal=_vector3(
                    item["plane_side2_crystal"],
                    name=f"{context}.plane_side2_crystal",
                ),
                operator_index=(
                    _integer(
                        item["operator_index"],
                        name=f"{context}.operator_index",
                        minimum=0,
                    )
                    if item.get("operator_index") is not None
                    else None
                ),
                source=str(item.get("source", "configured")),
            )
        )

    return {
        "enabled": True,
        "surface_normal_sample": _vector3(
            spec["surface_normal_sample"],
            name="trace_validation.surface_normal_sample",
        ),
        "minimum_linearity": minimum_linearity,
        "hypotheses": tuple(hypotheses),
    }


def _parse_output(
    spec: Mapping[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    allowed = {"directory", "run_name", "overwrite"}
    _reject_unknown(spec, allowed, context="output")
    _require_keys(spec, {"directory"}, context="output")
    directory = _resolve_relative(str(spec["directory"]), base_dir)
    run_name = str(spec.get("run_name", "")).strip()
    if not run_name:
        run_name = datetime.now(timezone.utc).strftime(
            "ebsd-run-%Y%m%dT%H%M%SZ"
        )
    if any(character in run_name for character in ("/", "\\", "\0")):
        raise PipelineConfigError(
            "output.run_name must be one directory name, not a path"
        )
    return {
        "directory": directory,
        "run_name": run_name,
        "overwrite": bool(spec.get("overwrite", False)),
    }


def _phase_rows(phases: Mapping[int, EBSDPhase]) -> list[dict[str, Any]]:
    rows = []
    for phase_id in sorted(phases):
        phase = phases[phase_id]
        lattice = phase.lattice
        rows.append(
            {
                "phase_id": phase_id,
                "name": phase.name,
                "point_group": phase.point_group,
                "a": lattice.a,
                "b": lattice.b,
                "c": lattice.c,
                "alpha_deg": lattice.alpha_deg,
                "beta_deg": lattice.beta_deg,
                "gamma_deg": lattice.gamma_deg,
                "length_unit": lattice.length_unit,
                "proper_symmetry_order": len(
                    phase.proper_symmetry_cartesian
                ),
            }
        )
    return rows


def _grain_rows(grains: Sequence[Grain]) -> list[dict[str, Any]]:
    rows = []
    for grain in grains:
        matrix = np.asarray(grain.mean_orientation, dtype=float)
        row: dict[str, Any] = {
            "grain_id": grain.grain_id,
            "phase_id": grain.phase_id,
            "size": grain.size,
            "centroid_x": grain.centroid[0],
            "centroid_y": grain.centroid[1],
            "centroid_z": grain.centroid[2],
            "gos_deg": grain.gos_deg,
            "maximum_spread_deg": grain.maximum_spread_deg,
        }
        for i in range(3):
            for j in range(3):
                row[f"g{i+1}{j+1}"] = matrix[i, j]
        rows.append(row)
    return rows


def _score_dict(score: TheoryConsistencyScore | None) -> Any:
    if score is None:
        return None
    return {
        "n_boundaries": score.n_boundaries,
        "operator_median_deg": score.operator_median_deg,
        "operator_p90_deg": score.operator_p90_deg,
        "parent_median_deg": score.parent_median_deg,
        "parent_p90_deg": score.parent_p90_deg,
        "combined_robust_score_deg": score.combined_robust_score_deg,
        "operator_residuals_deg": score.operator_residuals_deg,
        "parent_compatibility_residuals_deg": (
            score.parent_compatibility_residuals_deg
        ),
    }


@dataclass(frozen=True)
class ProductGrainContext:
    grains: tuple[Grain, ...]
    global_ids: tuple[int, ...]
    global_to_local: Mapping[int, int]
    adjacency_global: tuple[tuple[int, int], ...]
    adjacency_local: tuple[tuple[int, int], ...]
    orientations: np.ndarray


def _product_grain_context(
    grains: Sequence[Grain],
    segmentation: GrainSegmentation,
    *,
    product_phase_id: int,
) -> ProductGrainContext:
    product = tuple(
        grain for grain in grains if grain.phase_id == product_phase_id
    )
    if not product:
        raise ValueError(
            f"no reconstructed grains belong to product phase {product_phase_id}"
        )
    global_ids = tuple(grain.grain_id for grain in product)
    mapping = {grain_id: i for i, grain_id in enumerate(global_ids)}
    global_pairs = unique_grain_adjacency(
        segmentation,
        allowed_grains=set(global_ids),
    )
    local_pairs = tuple(
        (mapping[a], mapping[b]) for a, b in global_pairs
    )
    orientations = np.asarray(
        [grain.mean_orientation for grain in product],
        dtype=float,
    )
    return ProductGrainContext(
        grains=product,
        global_ids=global_ids,
        global_to_local=mapping,
        adjacency_global=global_pairs,
        adjacency_local=local_pairs,
        orientations=orientations,
    )


def _build_theory(
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
    R: np.ndarray,
    settings: Mapping[str, Any],
) -> TheoryLibrary:
    return build_theory_library(
        parent_phase,
        product_phase,
        R,
        quotient_tolerance_deg=settings["quotient_tolerance_deg"],
        operator_equivalence_tolerance_deg=settings[
            "operator_equivalence_tolerance_deg"
        ],
        crosscheck_topology=settings["crosscheck_topology"],
    )


def _boundary_rows(
    context: ProductGrainContext,
    theory: TheoryLibrary,
    product_phase: EBSDPhase,
    settings: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    if not context.adjacency_local:
        return [], None

    kernel = PreparedBoundaryOperatorKernel.prepare(
        context.orientations,
        context.adjacency_local,
        product_phase.proper_symmetry_cartesian,
    )
    residual_matrix = kernel.class_residuals_deg(
        theory.boundary_operators
    )

    rows: list[dict[str, Any]] = []
    for boundary_index, ((local_a, local_b), residuals) in enumerate(
        zip(context.adjacency_local, residual_matrix, strict=True)
    ):
        order = np.argsort(residuals)
        best_position = int(order[0])
        best = float(residuals[best_position])
        second = (
            float(residuals[int(order[1])])
            if len(order) > 1
            else 180.0
        )
        gap = second - best
        accepted = best <= settings["maximum_residual_deg"]
        ambiguous = accepted and gap < settings["minimum_margin_deg"]
        global_a = context.global_ids[local_a]
        global_b = context.global_ids[local_b]
        rows.append(
            {
                "boundary_index": boundary_index,
                "grain_a": min(global_a, global_b),
                "grain_b": max(global_a, global_b),
                "local_product_grain_a": local_a,
                "local_product_grain_b": local_b,
                "operator_index": (
                    theory.boundary_operators[
                        best_position
                    ].operator_index
                    if accepted
                    else None
                ),
                "best_residual_deg": best,
                "second_best_residual_deg": second,
                "ambiguity_gap_deg": gap,
                "accepted": accepted,
                "ambiguous": ambiguous,
            }
        )
    return rows, residual_matrix


def _domain_outputs(
    context: ProductGrainContext,
    theory: TheoryLibrary,
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
    settings: Mapping[str, Any],
) -> tuple[
    VariantGraphReport | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n = len(context.grains)
    unique_domain = np.full(n, -1, dtype=int)
    unique_variant = np.full(n, -1, dtype=int)
    unique_variant_residual = np.full(n, np.nan, dtype=float)

    if not settings["enabled"] or not context.adjacency_local:
        return (
            None,
            [],
            [],
            unique_domain,
            unique_variant,
            unique_variant_residual,
        )

    graph = reconstruct_variant_graph_candidates(
        context.orientations,
        context.adjacency_local,
        theory.variant_set.variants,
        product_phase.proper_symmetry_cartesian,
        parent_phase.proper_symmetry_cartesian,
        grain_weights=np.asarray(
            [grain.size for grain in context.grains],
            dtype=float,
        ),
        link_tolerance_deg=settings["link_tolerance_deg"],
        reconstruction_tolerance_deg=settings[
            "reconstruction_tolerance_deg"
        ],
        minimum_grains=settings["minimum_grains"],
    )

    domain_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    per_grain_domains: dict[int, list[int]] = {
        index: [] for index in range(n)
    }

    for domain_index, candidate in enumerate(graph.domain_candidates):
        local_indices = np.asarray(candidate.grain_indices, dtype=int)
        if len(local_indices) == 0:
            continue
        reconstruction = reconstruct_parent(
            context.orientations[local_indices],
            theory.variant_set.variants,
            product_phase.proper_symmetry_cartesian,
            parent_phase.proper_symmetry_cartesian,
            weights=np.asarray(
                [context.grains[i].size for i in local_indices],
                dtype=float,
            ),
            inlier_tolerance_deg=settings[
                "reconstruction_tolerance_deg"
            ],
            variant_acceptance_deg=max(
                5.0, settings["reconstruction_tolerance_deg"]
            ),
        )
        parent = reconstruction.parent_orientation
        domain_row: dict[str, Any] = {
            "domain_index": domain_index,
            "n_grains": len(local_indices),
            "support_weight": candidate.support_weight,
            "weighted_inlier_fraction": reconstruction.weighted_inlier_fraction,
            "mean_inlier_residual_deg": reconstruction.mean_inlier_residual_deg,
            "maximum_inlier_residual_deg": reconstruction.maximum_inlier_residual_deg,
            "iterations": reconstruction.iterations,
        }
        for i in range(3):
            for j in range(3):
                domain_row[f"parent_g{i+1}{j+1}"] = parent[i, j]
        domain_rows.append(domain_row)

        for position, local_index in enumerate(local_indices):
            per_grain_domains[int(local_index)].append(domain_index)
            assignment = reconstruction.variant_assignments[position]
            assignment_rows.append(
                {
                    "domain_index": domain_index,
                    "local_product_grain": int(local_index),
                    "grain_id": context.global_ids[int(local_index)],
                    "variant_index": assignment.variant_index,
                    "best_residual_deg": assignment.best_residual_deg,
                    "second_best_residual_deg": (
                        assignment.second_best_residual_deg
                    ),
                    "ambiguity_gap_deg": assignment.ambiguity_gap_deg,
                    "accepted": assignment.accepted,
                    "ambiguous_variant": assignment.ambiguous,
                    "domain_reconstruction_inlier": bool(
                        reconstruction.inlier_mask[position]
                    ),
                }
            )

    for local_index, domains in per_grain_domains.items():
        if len(domains) == 0:
            unique_domain[local_index] = -1
        elif len(domains) > 1:
            unique_domain[local_index] = -2
            unique_variant[local_index] = -2
        else:
            domain_id = domains[0]
            unique_domain[local_index] = domain_id
            matches = [
                row
                for row in assignment_rows
                if row["domain_index"] == domain_id
                and row["local_product_grain"] == local_index
            ]
            if len(matches) == 1 and matches[0]["accepted"]:
                unique_variant[local_index] = int(
                    matches[0]["variant_index"]
                )
                unique_variant_residual[local_index] = float(
                    matches[0]["best_residual_deg"]
                )

    return (
        graph,
        domain_rows,
        assignment_rows,
        unique_domain,
        unique_variant,
        unique_variant_residual,
    )


def _trace_rows(
    data: EBSDMap,
    segmentation: GrainSegmentation,
    context: ProductGrainContext,
    product_phase: EBSDPhase,
    boundary_rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not settings["enabled"]:
        return []

    by_pair = {
        (int(row["grain_a"]), int(row["grain_b"])): row
        for row in boundary_rows
    }
    output: list[dict[str, Any]] = []

    for local_a, local_b in context.adjacency_local:
        global_a = context.global_ids[local_a]
        global_b = context.global_ids[local_b]
        pair = tuple(sorted((global_a, global_b)))
        boundary = by_pair.get(pair)
        operator_index = (
            boundary.get("operator_index") if boundary is not None else None
        )

        base_row = {
            "grain_a": pair[0],
            "grain_b": pair[1],
            "operator_index": operator_index,
        }

        try:
            estimate = estimate_boundary_trace(
                data,
                segmentation,
                pair[0],
                pair[1],
                surface_normal_sample=settings[
                    "surface_normal_sample"
                ],
            )
        except ValueError as exc:
            output.append(
                {
                    **base_row,
                    "status": "unresolved_trace_geometry",
                    "message": str(exc),
                }
            )
            continue

        candidate_hypotheses = [
            hypothesis
            for hypothesis in settings["hypotheses"]
            if hypothesis.operator_index is None
            or operator_index is None
            or hypothesis.operator_index == operator_index
        ]
        if not candidate_hypotheses:
            output.append(
                {
                    **base_row,
                    "status": "no_matching_hypothesis",
                    "linearity": estimate.linearity,
                    "n_boundary_edges": estimate.n_boundary_edges,
                    "n_unique_midpoints": estimate.n_unique_midpoints,
                    "projected_span": estimate.projected_span,
                }
            )
            continue

        expanded: list[TwinPlaneHypothesis] = []
        metadata: dict[str, tuple[str, str]] = {}
        for hypothesis in candidate_hypotheses:
            label_a = f"{hypothesis.label}::as_configured"
            label_b = f"{hypothesis.label}::swapped"
            expanded.append(
                TwinPlaneHypothesis(
                    label=label_a,
                    plane_side1_crystal=hypothesis.plane_side1_crystal,
                    plane_side2_crystal=hypothesis.plane_side2_crystal,
                    operator_index=hypothesis.operator_index,
                    source=hypothesis.source,
                )
            )
            expanded.append(
                TwinPlaneHypothesis(
                    label=label_b,
                    plane_side1_crystal=hypothesis.plane_side2_crystal,
                    plane_side2_crystal=hypothesis.plane_side1_crystal,
                    operator_index=hypothesis.operator_index,
                    source=hypothesis.source,
                )
            )
            metadata[label_a] = (hypothesis.label, "as_configured")
            metadata[label_b] = (hypothesis.label, "swapped")

        try:
            ranked = rank_trace_hypotheses(
                estimate,
                context.orientations[local_a],
                context.orientations[local_b],
                product_phase,
                expanded,
                minimum_linearity=settings["minimum_linearity"],
            )
        except ValueError as exc:
            output.append(
                {
                    **base_row,
                    "status": "rejected_trace_shape",
                    "message": str(exc),
                    "linearity": estimate.linearity,
                    "n_boundary_edges": estimate.n_boundary_edges,
                    "n_unique_midpoints": estimate.n_unique_midpoints,
                    "projected_span": estimate.projected_span,
                }
            )
            continue

        best = ranked[0]
        original_label, side_assignment = metadata[
            best.hypothesis.label
        ]
        output.append(
            {
                **base_row,
                "status": "evaluated",
                "hypothesis": original_label,
                "side_assignment": side_assignment,
                "linearity": estimate.linearity,
                "n_boundary_edges": estimate.n_boundary_edges,
                "n_unique_midpoints": estimate.n_unique_midpoints,
                "projected_span": estimate.projected_span,
                "side1_trace_residual_deg": (
                    best.side1_trace_residual_deg
                ),
                "side2_trace_residual_deg": (
                    best.side2_trace_residual_deg
                ),
                "plane_normal_coherence_deg": (
                    best.plane_normal_coherence_deg
                ),
                "maximum_trace_residual_deg": (
                    best.maximum_trace_residual_deg
                ),
                "score_deg": best.score_deg,
            }
        )

    return output


def _convention_hypothesis_ranking(
    spec: Any,
    *,
    context: ProductGrainContext,
    theory: TheoryLibrary,
    parent_phase: EBSDPhase,
    product_phase: EBSDPhase,
) -> ConventionHypothesisRanking | None:
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise PipelineConfigError(
            "convention_hypotheses must be an object"
        )
    allowed = {
        "minimum_identifiable_gap_deg",
        "tie_tolerance_deg",
        "hypotheses",
    }
    _reject_unknown(spec, allowed, context="convention_hypotheses")
    hypotheses_spec = spec.get("hypotheses", [])
    if not hypotheses_spec:
        return None
    if not context.adjacency_local:
        return None

    if not isinstance(hypotheses_spec, list) or len(hypotheses_spec) < 2:
        raise PipelineConfigError(
            "convention_hypotheses.hypotheses must contain at least "
            "two explicit orientation transforms"
        )

    hypotheses: dict[str, np.ndarray] = {}
    for index, item in enumerate(hypotheses_spec):
        c = f"convention_hypotheses.hypotheses[{index}]"
        if not isinstance(item, dict):
            raise PipelineConfigError(f"{c} must be an object")
        _reject_unknown(
            item,
            {"label", "left_sample_rotation", "right_crystal_rotation", "transpose"},
            context=c,
        )
        _require_keys(item, {"label"}, context=c)
        label = str(item["label"])
        values = context.orientations.copy()
        if bool(item.get("transpose", False)):
            values = np.transpose(values, (0, 2, 1))
        if item.get("left_sample_rotation") is not None:
            Q = require_so3(
                _matrix3(
                    item["left_sample_rotation"],
                    name=f"{c}.left_sample_rotation",
                ),
                tolerance=2.0e-8,
                name=f"{c}.left_sample_rotation",
            )
            values = np.einsum("ij,njk->nik", Q, values)
        if item.get("right_crystal_rotation") is not None:
            Q = require_so3(
                _matrix3(
                    item["right_crystal_rotation"],
                    name=f"{c}.right_crystal_rotation",
                ),
                tolerance=2.0e-8,
                name=f"{c}.right_crystal_rotation",
            )
            values = np.einsum("nij,jk->nik", values, Q)
        if label in hypotheses:
            raise PipelineConfigError(
                f"duplicate convention hypothesis label {label!r}"
            )
        hypotheses[label] = values

    return rank_orientation_hypotheses(
        hypotheses,
        context.adjacency_local,
        theory,
        parent_phase.proper_symmetry_cartesian,
        product_phase.proper_symmetry_cartesian,
        tie_tolerance_deg=_positive(
            spec.get("tie_tolerance_deg", 1.0e-8),
            name="convention_hypotheses.tie_tolerance_deg",
        ),
        minimum_identifiable_gap_deg=_positive(
            spec.get("minimum_identifiable_gap_deg", 0.25),
            name="convention_hypotheses.minimum_identifiable_gap_deg",
        ),
    )


@dataclass(frozen=True)
class PipelineResult:
    run_directory: Path
    summary: Mapping[str, Any]


def run_pipeline(config_path: str | Path) -> PipelineResult:
    config, config_file = load_config(config_path)
    base_dir = config_file.parent

    phases = phases_from_config(config["phases"])
    parent_phase_id = _integer(
        config["parent_phase_id"],
        name="parent_phase_id",
        minimum=1,
    )
    product_phase_id = _integer(
        config["product_phase_id"],
        name="product_phase_id",
        minimum=1,
    )
    if parent_phase_id == product_phase_id:
        raise PipelineConfigError(
            "parent_phase_id and product_phase_id must be distinct"
        )
    if parent_phase_id not in phases:
        raise PipelineConfigError(
            f"parent_phase_id={parent_phase_id} is not defined in phases"
        )
    if product_phase_id not in phases:
        raise PipelineConfigError(
            f"product_phase_id={product_phase_id} is not defined in phases"
        )

    raw_map, input_path = load_ebsd_from_config(
        config["input"], base_dir=base_dir
    )
    original_audit = audit_map(raw_map)

    indexed_phase_ids = set(
        int(value) for value in np.unique(raw_map.phase_id[raw_map.indexed])
    )
    undefined = sorted(indexed_phase_ids.difference(phases))
    if undefined:
        raise PipelineConfigError(
            f"indexed EBSD map contains phase IDs not defined in config: {undefined}"
        )

    working_map, filter_audit = apply_quality_filters(
        raw_map, config.get("quality_filters", [])
    )
    if filter_audit.retained_indexed_points == 0:
        raise ValueError("quality filtering rejected every indexed point")
    working_audit = audit_map(working_map)

    segmentation_settings = _parse_segmentation(config["segmentation"])
    theory_settings = _parse_theory(config.get("theory"))
    refinement_settings = _parse_refinement(config.get("or_refinement"))
    parent_settings = _parse_parent_reconstruction(
        config.get("parent_reconstruction")
    )
    boundary_settings = _parse_boundary_classification(
        config.get("boundary_classification")
    )
    trace_settings = _parse_trace(config.get("trace_validation"))
    output_settings = _parse_output(
        config["output"], base_dir=base_dir
    )

    neighbor_graph = build_neighbor_graph(
        working_map,
        radius=segmentation_settings["neighbor_radius"],
        radius_factor=segmentation_settings["neighbor_radius_factor"],
    )
    segmentation = segment_grains(
        working_map,
        phases,
        threshold_deg=segmentation_settings["main_threshold_deg"],
        neighbor_graph=neighbor_graph,
        minimum_points=segmentation_settings["minimum_grain_points"],
    )
    grains = grain_statistics(
        working_map,
        segmentation,
        phases,
    )
    if not grains:
        raise ValueError(
            "grain segmentation produced no grains after filtering/minimum-size rules"
        )

    minimum_product_grains = max(
        2,
        (
            int(parent_settings["minimum_grains"])
            if parent_settings["enabled"]
            else 2
        ),
    )
    product_reliability = assess_grain_route_reliability(
        working_map,
        segmentation,
        grains,
        phase_id=product_phase_id,
        minimum_required_grains=minimum_product_grains,
    )
    if not product_reliability.allowed:
        raise InsufficientPhaseEvidenceError(
            "configured product phase failed the mandatory grain-level "
            "experimental reliability gate before OR/CT map inference: "
            + ",".join(product_reliability.reason_codes)
        )

    sweep = sweep_segmentation_thresholds(
        working_map,
        phases,
        segmentation_settings["sweep_thresholds_deg"],
        neighbor_graph=neighbor_graph,
        minimum_points=segmentation_settings["minimum_grain_points"],
    )
    kam = kernel_average_misorientation(
        working_map,
        phases,
        neighbor_graph=neighbor_graph,
        maximum_neighbor_misorientation_deg=segmentation_settings[
            "kam_max_neighbor_misorientation_deg"
        ],
    )

    context = _product_grain_context(
        grains,
        segmentation,
        product_phase_id=product_phase_id,
    )

    parent_phase = phases[parent_phase_id]
    product_phase = phases[product_phase_id]
    or_spec = config["orientation_relationship"]
    if not isinstance(or_spec, dict):
        raise PipelineConfigError(
            "orientation_relationship must be an object"
        )
    resolved_or = resolve_orientation_relationship(
        or_spec,
        parent_phase=parent_phase,
        product_phase=product_phase,
    )
    initial_theory = _build_theory(
        parent_phase,
        product_phase,
        resolved_or.R_parent_from_product,
        theory_settings,
    )

    initial_score = None
    if context.adjacency_local:
        initial_score = score_theory_consistency(
            context.orientations,
            context.adjacency_local,
            initial_theory,
            parent_phase.proper_symmetry_cartesian,
            product_phase.proper_symmetry_cartesian,
        )

    refinement: ORRefinementResult | None = None
    final_R = resolved_or.R_parent_from_product
    final_or_source = "configured"

    if refinement_settings["enabled"]:
        if not context.adjacency_local:
            raise ValueError(
                "OR refinement was enabled but the product-grain network "
                "contains no product/product boundaries"
            )
        refinement = refine_orientation_relationship_from_child_boundaries(
            context.orientations,
            context.adjacency_local,
            parent_phase,
            product_phase,
            resolved_or.R_parent_from_product,
            maximum_correction_deg=refinement_settings[
                "maximum_correction_deg"
            ],
            trim_fraction=refinement_settings["trim_fraction"],
            huber_delta_deg=refinement_settings["huber_delta_deg"],
            minimum_improvement_deg2=refinement_settings[
                "minimum_improvement_deg2"
            ],
            multi_start_step_deg=refinement_settings[
                "multi_start_step_deg"
            ],
            maximum_iterations=refinement_settings["maximum_iterations"],
        )
        if (
            refinement_settings["use_if_accepted"]
            and refinement.accepted_improvement
        ):
            final_R = refinement.fitted_R_parent_from_product
            final_or_source = "experimentally_refined_explicitly_enabled"

    final_theory = _build_theory(
        parent_phase,
        product_phase,
        final_R,
        theory_settings,
    )
    final_score = None
    if context.adjacency_local:
        final_score = score_theory_consistency(
            context.orientations,
            context.adjacency_local,
            final_theory,
            parent_phase.proper_symmetry_cartesian,
            product_phase.proper_symmetry_cartesian,
        )

    convention_ranking = _convention_hypothesis_ranking(
        config.get("convention_hypotheses"),
        context=context,
        theory=final_theory,
        parent_phase=parent_phase,
        product_phase=product_phase,
    )

    boundary_rows, _ = _boundary_rows(
        context,
        final_theory,
        product_phase,
        boundary_settings,
    )

    (
        variant_graph,
        domain_rows,
        assignment_rows,
        unique_domain_by_product_grain,
        unique_variant_by_product_grain,
        unique_variant_residual_by_product_grain,
    ) = _domain_outputs(
        context,
        final_theory,
        parent_phase,
        product_phase,
        parent_settings,
    )

    trace_rows = _trace_rows(
        working_map,
        segmentation,
        context,
        product_phase,
        boundary_rows,
        trace_settings,
    )

    point_domain = np.full(working_map.n_points, -1, dtype=int)
    point_variant = np.full(working_map.n_points, -1, dtype=int)
    point_variant_residual = np.full(
        working_map.n_points, np.nan, dtype=float
    )
    product_global_to_local = {
        grain_id: local
        for local, grain_id in enumerate(context.global_ids)
    }
    for point_index, grain_id in enumerate(segmentation.grain_id):
        if grain_id < 0:
            continue
        local = product_global_to_local.get(int(grain_id))
        if local is None:
            continue
        point_domain[point_index] = unique_domain_by_product_grain[local]
        point_variant[point_index] = unique_variant_by_product_grain[local]
        point_variant_residual[point_index] = (
            unique_variant_residual_by_product_grain[local]
        )

    output_root: Path = output_settings["directory"]
    target = output_root / output_settings["run_name"]
    output_root.mkdir(parents=True, exist_ok=True)
    if target.exists() and not output_settings["overwrite"]:
        raise FileExistsError(
            f"output run directory already exists: {target}. "
            "Set output.overwrite=true only if replacement is intentional."
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_settings['run_name']}.staging-",
            dir=output_root,
        )
    )
    backup: Path | None = None
    promoted = False

    try:
        _atomic_csv(
            staging / "phases.csv",
            pd.DataFrame(_phase_rows(phases)),
        )
        _atomic_csv(
            staging / "grains.csv",
            pd.DataFrame(_grain_rows(grains)),
        )
        _atomic_csv(
            staging / "boundaries.csv",
            pd.DataFrame(boundary_rows),
        )
        _atomic_csv(
            staging / "parent_domains.csv",
            pd.DataFrame(domain_rows),
        )
        _atomic_csv(
            staging / "grain_variant_assignments.csv",
            pd.DataFrame(assignment_rows),
        )
        _atomic_csv(
            staging / "trace_validation.csv",
            pd.DataFrame(trace_rows),
        )

        sweep_rows = []
        for index, entry in enumerate(sweep.entries):
            sweep_rows.append(
                {
                    "threshold_deg": entry.threshold_deg,
                    "n_grains": entry.n_grains,
                    "retained_fraction": entry.retained_fraction,
                    "median_gos_deg": entry.median_gos_deg,
                    "p95_gos_deg": entry.p95_gos_deg,
                    "adjusted_rand_to_next": (
                        sweep.consecutive_adjusted_rand[index]
                        if index < len(sweep.consecutive_adjusted_rand)
                        else None
                    ),
                }
            )
        _atomic_csv(
            staging / "segmentation_sweep.csv",
            pd.DataFrame(sweep_rows),
        )

        np.savez_compressed(
            staging / "map_fields.npz",
            x=working_map.x,
            y=working_map.y,
            z=working_map.z,
            indexed=working_map.indexed,
            phase_id=working_map.phase_id,
            grain_id=segmentation.grain_id,
            kam_deg=kam,
            parent_domain_id=point_domain,
            variant_id=point_variant,
            variant_residual_deg=point_variant_residual,
        )

        theory_payload = {
            "final_or_source": final_or_source,
            "R_parent_from_product": final_R,
            "n_variants": final_theory.n_variants,
            "n_boundary_operators": final_theory.n_boundary_operators,
            "variants": [
                {
                    "variant_index": item.variant_index,
                    "label": item.label,
                    "R_parent_from_product": item.R_parent_from_product,
                }
                for item in final_theory.variant_set.variants
            ],
            "operators": [
                {
                    "operator_index": operator.operator_index,
                    "representative": operator.representative,
                    "variant_pairs": operator.variant_pairs,
                }
                for operator in final_theory.boundary_operators
            ],
        }
        _atomic_json(staging / "theory.json", theory_payload)

        resolved_config = json.loads(json.dumps(config))
        resolved_config["input"]["path"] = str(input_path)
        resolved_config["output"]["directory"] = str(output_root)
        _atomic_json(
            staging / "resolved_config.json",
            resolved_config,
        )

        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "config_path": str(config_file),
                "config_sha256": sha256_json(config),
                "input_path": str(input_path),
                "input_sha256": sha256_file(input_path),
                "git": _git_revision(),
            },
            "phase_roles": {
                "parent_phase_id": parent_phase_id,
                "product_phase_id": product_phase_id,
            },
            "map_audit_original": original_audit,
            "quality_filter_audit": filter_audit,
            "map_audit_working": working_audit,
            "segmentation": {
                "main_threshold_deg": segmentation_settings[
                    "main_threshold_deg"
                ],
                "n_grains": segmentation.n_grains,
                "minimum_grain_points": segmentation_settings[
                    "minimum_grain_points"
                ],
                "neighbor_radius": neighbor_graph.radius,
                "coordinate_dimension": neighbor_graph.coordinate_dimension,
                "sweep": sweep_rows,
            },
            "phase_reliability": {
                "product_grain_route": product_reliability,
            },
            "orientation_relationship": {
                "configured": resolved_or,
                "initial_theory": {
                    "n_variants": initial_theory.n_variants,
                    "n_boundary_operators": initial_theory.n_boundary_operators,
                    "consistency": _score_dict(initial_score),
                },
                "refinement_enabled": refinement_settings["enabled"],
                "refinement": refinement,
                "refined_used_for_final_theory": (
                    final_or_source
                    == "experimentally_refined_explicitly_enabled"
                ),
                "final_source": final_or_source,
                "final_R_parent_from_product": final_R,
                "final_consistency": _score_dict(final_score),
            },
            "convention_hypothesis_ranking": convention_ranking,
            "parent_reconstruction": {
                "enabled": parent_settings["enabled"],
                "n_domain_candidates": (
                    len(variant_graph.domain_candidates)
                    if variant_graph is not None
                    else 0
                ),
                "unassigned_product_grains": (
                    variant_graph.unassigned_grains
                    if variant_graph is not None
                    else ()
                ),
                "ambiguous_product_grains": (
                    variant_graph.ambiguous_grains
                    if variant_graph is not None
                    else ()
                ),
                "unique_domain_code_semantics": {
                    "-1": "no accepted parent-domain candidate",
                    "-2": "multiple accepted parent-domain candidates; ambiguous",
                    ">=0": "unique domain index",
                },
            },
            "boundaries": {
                "n_product_product_boundaries": len(boundary_rows),
                "n_operator_accepted": sum(
                    bool(row["accepted"]) for row in boundary_rows
                ),
                "n_operator_ambiguous": sum(
                    bool(row["ambiguous"]) for row in boundary_rows
                ),
            },
            "traces": {
                "enabled": trace_settings["enabled"],
                "n_rows": len(trace_rows),
                "n_evaluated": sum(
                    row.get("status") == "evaluated"
                    for row in trace_rows
                ),
            },
            "files": {
                "phases": "phases.csv",
                "grains": "grains.csv",
                "boundaries": "boundaries.csv",
                "parent_domains": "parent_domains.csv",
                "grain_variant_assignments": (
                    "grain_variant_assignments.csv"
                ),
                "trace_validation": "trace_validation.csv",
                "segmentation_sweep": "segmentation_sweep.csv",
                "map_fields": "map_fields.npz",
                "theory": "theory.json",
                "resolved_config": "resolved_config.json",
            },
        }
        _atomic_json(staging / "summary.json", summary)

        if target.exists():
            backup = target.with_name(
                target.name + f".backup-{uuid.uuid4().hex}"
            )
            os.replace(target, backup)
        os.replace(staging, target)
        promoted = True
        if backup is not None:
            shutil.rmtree(backup)
        return PipelineResult(
            run_directory=target,
            summary=_json_safe(summary),
        )

    except Exception:
        if not promoted and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise


def validate_config_only(config_path: str | Path) -> dict[str, Any]:
    """Parse all static configuration sections without running the EBSD analysis."""

    config, config_file = load_config(config_path)
    base_dir = config_file.parent
    phases = phases_from_config(config["phases"])

    parent_id = _integer(
        config["parent_phase_id"], name="parent_phase_id", minimum=1
    )
    product_id = _integer(
        config["product_phase_id"], name="product_phase_id", minimum=1
    )
    if parent_id == product_id:
        raise PipelineConfigError(
            "parent_phase_id and product_phase_id must be distinct"
        )
    if parent_id not in phases or product_id not in phases:
        raise PipelineConfigError(
            "parent/product phase IDs must both exist in phases"
        )

    _parse_segmentation(config["segmentation"])
    _parse_theory(config.get("theory"))
    _parse_refinement(config.get("or_refinement"))
    _parse_parent_reconstruction(config.get("parent_reconstruction"))
    _parse_boundary_classification(config.get("boundary_classification"))
    _parse_trace(config.get("trace_validation"))
    _parse_output(config["output"], base_dir=base_dir)
    validate_quality_filter_rules_static(
        config.get("quality_filters", [])
    )

    or_spec = config["orientation_relationship"]
    if not isinstance(or_spec, dict):
        raise PipelineConfigError(
            "orientation_relationship must be an object"
        )
    resolved = resolve_orientation_relationship(
        or_spec,
        parent_phase=phases[parent_id],
        product_phase=phases[product_id],
    )

    input_spec = config["input"]
    input_audit = validate_input_config_static(
        input_spec,
        base_dir=base_dir,
    )
    input_path = input_audit["path"]

    return {
        "config_path": str(config_file),
        "config_sha256": sha256_json(config),
        "input_path": str(input_path),
        "input_exists": input_path.is_file(),
        "parent_phase_id": parent_id,
        "product_phase_id": product_id,
        "or_source_mode": resolved.source_mode,
        "or_candidate_count": resolved.candidate_count,
        "or_selected_candidate_index": resolved.selected_candidate_index,
        "R_parent_from_product": resolved.R_parent_from_product.tolist(),
    }
