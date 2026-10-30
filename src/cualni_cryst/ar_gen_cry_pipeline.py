from __future__ import annotations

"""Strict, standalone Cry -> Gen -> AR scientific workflow.

No legacy Python-2 GUI imports. No implicit orientation-relationship inference.
Every stage has explicit inputs and can fail without contaminating later stages.
"""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .ebsd_h5oina import load_h5oina_sample

from .legacy_suite_bridge import (
    Phase, crystal_geometry, inspect_ebsd, reconstruct_ebsd_parent, variant_groupoid,
)
from .parent_or_evidence import compare_parent_to_anchor, audit_anchor_status
from .ebsd_map import EBSDPhase
from .orientation_kernel import require_so3

MAX_EBSD_BYTES = 8 * 1024 * 1024


def cry_stage(parent: Phase, product: Phase, direction: tuple[float, float, float],
              plane: tuple[float, float, float]) -> dict:
    """Inspect both metrics, preserving the chosen basis and unit conventions."""
    if len(direction) != 3 or len(plane) != 3:
        raise ValueError("Direction and plane must each have three components")
    return {
        "stage": "Cry",
        "parent_phase": asdict(parent),
        "product_phase": asdict(product),
        "direction_uvw": list(direction),
        "plane_hkl": list(plane),
        "parent": crystal_geometry(parent, direction, plane),
        "product": crystal_geometry(product, direction, plane),
        "provenance": "user input; not a literature-validated phase assignment",
    }


def gen_stage(cry: dict, correspondence_text: str) -> dict:
    """Calculate exact groupoid for the lattice models accepted in Cry."""
    if not cry or cry.get("stage") != "Cry":
        raise ValueError("Complete Cry before Gen")
    p = Phase(**cry["parent_phase"])
    m = Phase(**cry["product_phase"])
    result = variant_groupoid(p, m, correspondence_text)
    order = int(result["subgroup_order"])
    if not order or len(result["adjacency"]) != result["variants"]:
        raise AssertionError("Invalid groupoid dimensions")
    if int(result["variants"]) * order != int(p.verified_symmetry().__len__()):
        raise AssertionError("Lagrange invariant failed")
    if sum(result["class_sizes"]) != len(p.verified_symmetry()):
        raise AssertionError("Operator partition invariant failed")
    if result["burnside_count"] != result["operator_classes"]:
        raise AssertionError("Independent operator count differs")
    return {
        "stage": "Gen",
        "cry_snapshot": {"parent_phase": cry["parent_phase"], "product_phase": cry["product_phase"]},
        **result,
        "provenance": "theoretical lattice correspondence; does not establish experimental OR",
    }


def ar_stage(filename: str, data: bytes) -> dict:
    """Strict text EBSD upload; never deserialize executable legacy formats."""
    if not isinstance(data, bytes) or not data:
        raise ValueError("Upload a nonempty EBSD file")
    if Path(filename).suffix.lower() == ".h5oina":
        _, summary = load_h5oina_sample(filename, data, maximum_sample=12000)
        return {
            "stage": "AR",
            "filename": Path(filename).name,
            "file_size_bytes": len(data),
            **summary,
            "provenance": "Oxford H5OINA; phase-cell metadata and raw radians; sample surface frame uncorrected",
        }

    if len(data) > MAX_EBSD_BYTES:
        raise ValueError("EBSD file exceeds the 8 MiB interactive safety limit")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".ang", ".ctf"}:
        raise ValueError("AR accepts only .ang and .ctf EBSD text files")
    if b"\x00" in data[:4096]:
        raise ValueError("Binary content is not supported")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / ("ebsd" + suffix)
        path.write_bytes(data)
        _, report = inspect_ebsd(path)
    return {
        "stage": "AR",
        "filename": Path(filename).name,
        "file_size_bytes": len(data),
        **report,
        "provenance": "EBSD vendor angles; unverified sample/crystal frame until separately checked",
    }


def ar_parent_fit(filename: str, data: bytes, cry: dict, product_phase_id: int,
                  verified_or_json: str, *, max_points: int = 12,
                  parent_anchor_orientation_json: str | None = None,
                  indexed_18r_basis_verified: bool = False,
                  specimen_frame_verified: bool = False) -> dict:
    """Exploratory fit ONLY after explicit experimental OR data are supplied."""
    audit = ar_stage(filename, data)
    if not cry or cry.get("stage") != "Cry":
        raise ValueError("Cry phase models are required")
    try:
        parsed = json.loads(verified_or_json)
        array = np.asarray(parsed, dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Provide a JSON list of verified proper 3x3 rotations") from exc
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError("Expected list of proper 3x3 OR rotation matrices")
    if not np.isfinite(array).all():
        raise ValueError("OR matrices contain nonfinite entries")
    # Validation of SO(3), symmetry and the actual parent fit is performed by
    # VariantOR and reconstruct_parent in the existing scientific backend.
    suffix = Path(filename).suffix.lower()
    if suffix == ".h5oina":
        ebsd, _ = load_h5oina_sample(filename, data, maximum_sample=12000)
        geometry = [item for item in audit["phase_geometry"]
                    if item["phase_id"] == int(product_phase_id)]
        if len(geometry) != 1:
            raise ValueError("Selected martensite phase is absent in H5OINA")
        cell = Phase(**cry["product_phase"])
        measured = np.array(geometry[0]["abc_angstrom"] + geometry[0]["angles_deg"], float)
        declared = np.array([cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma], float)
        # Metadata cell is authoritative for this indexed EBSD basis.
        if not np.allclose(measured[:3], declared[:3], atol=0.02, rtol=0) or not np.allclose(
            measured[3:], declared[3:], atol=0.2, rtol=0
        ):
            raise ValueError(
                "Cry product cell does not match the indexed H5OINA phase. "
                "Use exactly the 18R reduced cell setting stored in the file."
            )
        fit = reconstruct_ebsd_parent(
            ebsd, Phase(**cry["parent_phase"]), cell,
            product_phase_id, array, maximum_points=max_points,
        )
    else:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ("ebsd" + suffix)
            path.write_bytes(data)
            ebsd, _ = inspect_ebsd(path)
            fit = reconstruct_ebsd_parent(
                ebsd, Phase(**cry["parent_phase"]), Phase(**cry["product_phase"]),
                product_phase_id, array, maximum_points=max_points,
            )
    anchored = None
    anchor_residual = None
    if parent_anchor_orientation_json and parent_anchor_orientation_json.strip():
        try:
            known = np.asarray(json.loads(parent_anchor_orientation_json), float)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Measured parent orientation must be a JSON 3x3 matrix") from exc
        known = require_so3(known, tolerance=2e-8, name="measured beta parent orientation")
        pa = Phase(**cry["parent_phase"])
        sym = EBSDPhase.from_point_group(
            1, pa.name, pa.lattice(), pa.point_group
        ).proper_symmetry_cartesian
        anchor_residual = compare_parent_to_anchor(
            np.asarray(fit["parent_crystal_to_sample_rotation"], float), known, sym
        )
        anchored = {
            "parent_orientation_matrix": known.tolist(),
            "evidence_status": "user-supplied, not independently verified by software",
            "candidate_vs_measured_parent_deg": anchor_residual,
            "agrees_with_anchor_within_3deg": anchor_residual <= 3.0,
        }
    gate = audit_anchor_status(
        measured_parent_orientation=anchored is not None,
        independent_direction_count=0,
        indexed_parent_phase_present=False,
        verified_change_of_basis=indexed_18r_basis_verified,
        scan_frame_verified=specimen_frame_verified,
    )
    # A user checkbox is an external evidence claim, not experimental
    # certification. The software independently verifies geometry and SO3 only.
    if anchor_residual is not None and anchor_residual > 3.0:
        gate["blocking_requirements"].append("supplied anchor is inconsistent with fitted parent orientation")
    return {
        "stage": "AR parent fit",
        "experimental_or_user_supplied": True,
        "parent_anchor_comparison": anchored,
        "parent_anchor_residual_deg": anchor_residual,
        "evidence_gate": gate,
        "experimental_certification": False,
        **fit,
    }


def report_json(cry: dict | None, gen: dict | None,
                ar: dict | None, fit: dict | None = None) -> str:
    """Export only the explicitly calculated stages. Never create missing results."""
    if not cry:
        raise ValueError("Run Cry before exporting a report")
    report = {
        "software": "AR · Gen · Cry",
        "schema_version": 1,
        "scientific_status": "exploratory: no automatic experimental OR identification",
        "cry": cry,
        "gen": gen,
        "ar": ar,
        "ar_parent_fit": fit,
    }
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
