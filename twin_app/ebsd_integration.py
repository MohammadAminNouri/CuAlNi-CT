from __future__ import annotations

"""CTF-to-EBSD experimental analysis for the dedicated Twin App.

This is not a twin classifier. It does not silently map OR-variant numbers
onto correspondence/stretch-variant numbers or identify 2H versus 18R
from already indexed CTF phase labels. It never changes PTMC/BJ/CT equations.
"""
from dataclasses import dataclass
import hashlib
import io
import json
import zipfile
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cualni_cryst.ebsd_pipeline import run_pipeline
from cualni_cryst.orientation_kernel import require_so3
from cualni_cryst.project_io import project_from_dict

MAX_CTF_BYTES = 35 * 1024 * 1024


@dataclass(frozen=True)
class EBSDObservation:
    sha256: str
    summary: dict[str, Any]
    boundaries: tuple[dict[str, Any], ...]
    grain_assignments: tuple[dict[str, Any], ...]
    physical_twin_evidence: str = "NOT_CERTIFIED_FROM_CTF_ALONE"
    evidence_archive: bytes = b""


def parse_explicit_or(text: str) -> np.ndarray:
    """Require a supplied proper parent<-product rotation; no identity default."""
    rows = [row.strip().split() for row in text.strip().splitlines() if row.strip()]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("Provide exactly three lines containing three numbers each")
    try:
        matrix = np.array([[float(x) for x in row] for row in rows], dtype=float)
    except ValueError as exc:
        raise ValueError("OR contains nonnumeric values") from exc
    return require_so3(matrix, tolerance=2.0e-8, name="explicit parent<-product OR")


def _phase(phase: Any, phase_id: int) -> dict[str, Any]:
    l = phase.lattice
    return {
        "id": phase_id,
        "name": phase.label or phase.phase_id,
        "point_group": phase.point_group_symbol,
        "lattice": {
            "a": l.a, "b": l.b, "c": l.c,
            "alpha_deg": l.alpha_deg, "beta_deg": l.beta_deg,
            "gamma_deg": l.gamma_deg,
            "length_unit": l.length_unit,
        },
    }


def build_ctf_config(
    project_payload: Mapping[str, Any], transformation_id: str,
    *, parent_phase_id: int, product_phase_id: int,
    verified_or: np.ndarray, file_name: str = "uploaded.ctf",
    minimum_grain_points: int = 3,
    segmentation_threshold_deg: float = 3.0,
) -> dict[str, Any]:
    if parent_phase_id < 1 or product_phase_id < 1 or parent_phase_id == product_phase_id:
        raise ValueError("Distinct positive AZtec phase IDs are required")
    if not (1 <= minimum_grain_points <= 10000):
        raise ValueError("Minimum grain point count must be between 1 and 10000")
    if not np.isfinite(segmentation_threshold_deg) or segmentation_threshold_deg <= 0:
        raise ValueError("Segmentation threshold must be positive")
    if Path(file_name).suffix.lower() != ".ctf" or Path(file_name).name != file_name:
        raise ValueError("A simple .ctf filename is required")
    R = require_so3(verified_or, tolerance=2e-8, name="verified OR")
    state = project_from_dict(dict(project_payload), source="twin_app:explicit_ctf")
    transformation = state.project.transformation(transformation_id)
    parent = state.project.phase(transformation.parent_phase_id)
    product = state.project.phase(transformation.product_phase_id)
    return {
        "schema_version": 1,
        "input": {"path": file_name, "format": "ctf"},
        "phases": [_phase(parent, parent_phase_id), _phase(product, product_phase_id)],
        "parent_phase_id": int(parent_phase_id),
        "product_phase_id": int(product_phase_id),
        "orientation_relationship": {
            "mode": "matrix", "matrix": R.tolist(),
            "matrix_direction": "parent_from_product",
        },
        "segmentation": {
            "main_threshold_deg": float(segmentation_threshold_deg),
            "sweep_thresholds_deg": [1.0, 2.0, 3.0, 5.0, 8.0],
            "minimum_grain_points": int(minimum_grain_points),
            "neighbor_radius_factor": 1.15,
            "kam_max_neighbor_misorientation_deg": 5.0,
        },
        "theory": {"crosscheck_topology": True},
        "or_refinement": {"enabled": False},
        "parent_reconstruction": {"enabled": True, "minimum_grains": 2},
        "boundary_classification": {"maximum_residual_deg": 3.0, "minimum_margin_deg": 0.5},
        "trace_validation": {"enabled": False},
        "output": {"directory": "results", "run_name": "ctf-experimental", "overwrite": False},
    }


def inspect_ctf_upload(file_bytes: bytes, *, filename: str) -> dict[str, Any]:
    """Inspect real CTF orientations and phase IDs without requiring an OR.

    This is an input-readiness diagnostic, NOT grain segmentation, OR discovery
    or certification of any twin. Uses the SAME native loader as run_pipeline.
    """
    if not file_bytes or len(file_bytes) > MAX_CTF_BYTES:
        raise ValueError("CTF is empty or exceeds the 35 MB analysis limit")
    if not file_bytes.lstrip().startswith(b"Channel Text File"):
        raise ValueError("Expected an Oxford/AZtec Channel Text File header")
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".ctf":
        raise ValueError("A simple .ctf filename is required")
    from cualni_cryst.ebsd_io import load_ctf

    with tempfile.TemporaryDirectory(prefix="cualni-ctf-preview-") as tmp:
        path = Path(tmp) / filename
        path.write_bytes(file_bytes)
        mapped = load_ctf(path)
    indexed = np.asarray(mapped.indexed, dtype=bool)
    phase_ids, counts = np.unique(mapped.phase_id[indexed], return_counts=True)
    return {
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
        "filename": filename,
        "total_pixels": int(len(mapped.phase_id)),
        "indexed_pixels": int(np.count_nonzero(indexed)),
        "phase_counts": {int(p): int(n) for p, n in zip(phase_ids, counts)},
        "orientation_convention": str(mapped.metadata.get("orientation_convention", "")),
        "phase_identification": "CTF phase IDs are labels; 2H/18R identity not verified.",
        "independent_or_available": False,
    }


def ctf_analysis_signature(
    file_bytes: bytes, project_signature: str, matrix_text: str,
    parent_phase_id: int, product_phase_id: int, minimum_grain_points: int,
    segmentation_threshold_deg: float = 3.0,
) -> str:
    """Bind conclusions to ALL scientific inputs, excluding UI-only controls."""
    payload = {
        "ctf_sha256": hashlib.sha256(file_bytes).hexdigest(),
        "model_signature": project_signature,
        "or_text": matrix_text.strip(),
        "parent_phase_id": int(parent_phase_id),
        "product_phase_id": int(product_phase_id),
        "minimum_grain_points": int(minimum_grain_points),
        "segmentation_threshold_deg": float(segmentation_threshold_deg),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def run_ctf_observation(
    file_bytes: bytes, project_payload: Mapping[str, Any], transformation_id: str,
    *, parent_phase_id: int, product_phase_id: int, verified_or: np.ndarray,
    filename: str, minimum_grain_points: int = 3,
    segmentation_threshold_deg: float = 3.0,
) -> EBSDObservation:
    if not file_bytes or len(file_bytes) > MAX_CTF_BYTES:
        raise ValueError("CTF is empty or exceeds the 35 MB per-analysis safety limit")
    if not file_bytes.lstrip().startswith(b"Channel Text File"):
        raise ValueError("Expected an Oxford/AZtec Channel Text File header; do not upload a CIF here")
    # The backend must independently enforce the same CTF-to-model
    # crystallographic setting check as the UI, not trust a prior click.
    # No automatic basis reorientation, OR or parent correspondence is inferred.
    from .ctf_independent import (
        declared_ctf_phases, compare_ctf_model_settings,
    )
    declared = declared_ctf_phases(file_bytes)
    if declared:
        setting_audit = compare_ctf_model_settings(
            declared, project_payload, transformation_id,
            parent_phase_id, product_phase_id,
        )
        if not setting_audit["pass"]:
            raise ValueError(
                "CTF phase cell/point-group settings conflict with the "
                "selected crystallographic model. Verify conventional "
                "axes and correspondence before any operator matching."
            )
    config = build_ctf_config(
        project_payload, transformation_id,
        parent_phase_id=parent_phase_id, product_phase_id=product_phase_id,
        verified_or=verified_or, file_name=filename,
        minimum_grain_points=minimum_grain_points,
        segmentation_threshold_deg=segmentation_threshold_deg,
    )
    with tempfile.TemporaryDirectory(prefix="cualni-twin-ctf-") as tmp:
        root = Path(tmp)
        (root / filename).write_bytes(file_bytes)
        (root / "pipeline.json").write_text(json.dumps(config), encoding="utf-8")
        result = run_pipeline(root / "pipeline.json")
        summary = dict(result.summary)
        def read_rows(name: str) -> tuple[dict[str, Any], ...]:
            p = result.run_directory / name
            if not p.is_file() or p.stat().st_size == 0:
                return ()
            frame = pd.read_csv(p)
            return tuple(frame.where(pd.notna(frame), None).to_dict(orient="records"))
        # Retain EVERY scientific output before the temporary working
        # directory disappears: map_fields.npz, grain/boundary tables,
        # variant assignments, config, provenance and theory library.
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(result.run_directory.iterdir()):
                if path.is_file():
                    archive.write(path, arcname=path.name)
        return EBSDObservation(
            sha256=hashlib.sha256(file_bytes).hexdigest(),
            evidence_archive=output.getvalue(),
            summary=summary,
            boundaries=read_rows("boundaries.csv"),
            grain_assignments=read_rows("grain_variant_assignments.csv"),
        )


def summarize_twin_candidates(report: Any, observation: EBSDObservation) -> dict[str, Any]:
    """Report theoretical candidates BESIDE data, never assert correspondence=OR identity."""
    theoretical_pairs = {
        tuple(sorted((p.variant_i, p.variant_j)))
        for family in report.families for p in family.pair_records
        if p.constructions
    }
    accepted = sum(bool(row.get("accepted")) for row in observation.boundaries)
    ambiguous = sum(bool(row.get("ambiguous")) for row in observation.boundaries)
    return {
        "physical_twin_certification": observation.physical_twin_evidence,
        "reason": (
            "Experimental OR-variant IDs cannot be equated with correspondence "
            "or metric-stretch IDs without an independently verified map. "
            "A boundary-operator fit alone is not a rank-one twin certificate."
        ),
        "theoretical_twin_pairs": len(theoretical_pairs),
        "observed_boundaries": len(observation.boundaries),
        "accepted_operator_fits": accepted,
        "ambiguous_operator_fits": ambiguous,
        "unresolved_2h_18r_indexing": True,
        "three_dimensional_twin_plane_measured": False,
    }
