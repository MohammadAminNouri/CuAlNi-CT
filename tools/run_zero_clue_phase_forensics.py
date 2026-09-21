from __future__ import annotations

"""Run a sealed zero-clue EBSD forensic audit with compact scientific outputs.

Human-readable JSON contains summaries only. Point/component label arrays are
stored in a compressed NPZ sidecar and covered by a SHA-256 manifest.
"""

from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

from cualni_cryst.ebsd_analysis import build_neighbor_graph
from cualni_cryst.ebsd_map import EBSDMap, EBSDPhase, audit_map
from cualni_cryst.ebsd_phase_forensics import (
    PhaseForensicsSettings,
    assert_vectorized_disorientation_parity,
    compact_phase_forensics_report,
    phase_forensics_array_payload,
    run_phase_forensics,
)
from cualni_cryst.lattice import Lattice


EXPECTED_CHALLENGE_SHA256 = (
    "c51345e32d09e5b4ca9d4a2db7ec0f3a172fc3e78f100ece16b50497c35b5b21"
)
EXPECTED_OBSERVATIONS_SHA256 = (
    "db172ea6c8bdf283bd7503fd0bd25bb7cced687aa71d0597ca8cabb81d749484"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    serializable = json_safe(payload)
    data = (
        json.dumps(
            serializable,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, data)


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp-",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(fd)
    try:
        with open(tmp_name, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def validate_contract(challenge: Mapping[str, Any]) -> None:
    required_root = {
        "schema_version",
        "data_file",
        "orientation_convention",
        "phases",
        "grid_metadata",
        "analysis_contract",
        "settings",
    }
    if set(challenge) != required_root:
        raise RuntimeError(
            "sealed challenge root keys changed; refusing to reinterpret it"
        )

    contract = challenge["analysis_contract"]
    required_contract = {
        "phase_names_withheld": True,
        "parent_product_roles_withheld": True,
        "orientation_relationship_withheld": True,
        "correspondence_withheld": True,
        "expected_operator_families_withheld": True,
        "literature_labels_withheld": True,
        "chemical_identity_withheld": True,
        "solver_must_test_all_unordered_phase_pairs": True,
    }
    if contract != required_contract:
        raise RuntimeError(
            "zero-clue analysis contract differs from sealed specification"
        )

    forbidden_exact_keys = {
        "orientation_relationship",
        "expected_or",
        "true_or",
        "oracle",
        "answer",
        "parent_phase_id",
        "product_phase_id",
    }

    def walk(value: Any, path: str = "$") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) in forbidden_exact_keys:
                    raise RuntimeError(
                        f"answer-bearing key {key!r} found at {path}"
                    )
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(challenge)


def phase_from_spec(spec: Mapping[str, Any]) -> EBSDPhase:
    allowed = {"id", "anonymous_label", "lattice", "point_group"}
    if set(spec) != allowed:
        raise RuntimeError(
            "anonymous phase specification changed from sealed schema"
        )
    lattice = spec["lattice"]
    return EBSDPhase.from_point_group(
        int(spec["id"]),
        str(spec["anonymous_label"]),
        Lattice(
            float(lattice["a"]),
            float(lattice["b"]),
            float(lattice["c"]),
            float(lattice.get("alpha_deg", 90.0)),
            float(lattice.get("beta_deg", 90.0)),
            float(lattice.get("gamma_deg", 90.0)),
            label=str(spec["anonymous_label"]),
        ),
        str(spec["point_group"]),
    )


def load_challenge(root: Path):
    challenge_path = root / "challenge.json"
    observations_path = root / "observations.npz"
    if not challenge_path.is_file() or not observations_path.is_file():
        raise FileNotFoundError(
            "challenge directory must contain challenge.json and observations.npz"
        )
    if sha256_file(challenge_path) != EXPECTED_CHALLENGE_SHA256:
        raise RuntimeError("challenge.json hash mismatch")
    if sha256_file(observations_path) != EXPECTED_OBSERVATIONS_SHA256:
        raise RuntimeError("observations.npz hash mismatch")

    challenge = json.loads(challenge_path.read_text())
    validate_contract(challenge)
    phases = {
        int(spec["id"]): phase_from_spec(spec)
        for spec in challenge["phases"]
    }

    with np.load(observations_path) as payload:
        quality = {
            key[len("quality_"):]: np.asarray(payload[key])
            for key in payload.files
            if key.startswith("quality_")
        }
        data = EBSDMap(
            orientations=np.asarray(payload["orientations"], dtype=float),
            phase_id=np.asarray(payload["phase_id"], dtype=int),
            indexed=np.asarray(payload["indexed"], dtype=bool),
            x=np.asarray(payload["x"], dtype=float),
            y=np.asarray(payload["y"], dtype=float),
            z=np.asarray(payload["z"], dtype=float),
            quality=quality,
            metadata=challenge.get("grid_metadata", {}),
        )
    return challenge, data, phases


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print(
            "usage: python tools/run_zero_clue_phase_forensics.py "
            "data/real_blind_challenge [summary.json]",
            file=sys.stderr,
        )
        return 2

    root = Path(sys.argv[1]).resolve()
    summary_path = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) == 3
        else Path("zero_clue_phase_forensics.json").resolve()
    )
    if summary_path.suffix.lower() != ".json":
        raise ValueError("output path must end in .json")

    arrays_path = summary_path.with_name(
        summary_path.stem + "_arrays.npz"
    )
    manifest_path = summary_path.with_name(
        summary_path.stem + "_manifest.json"
    )

    challenge, data, phases = load_challenge(root)
    graph = build_neighbor_graph(data)

    print("SEALED ZERO-CLUE CHALLENGE VERIFIED", flush=True)
    print(
        f"points={data.n_points:,}; indexed={np.count_nonzero(data.indexed):,}; "
        f"neighbor_edges={len(graph.edges):,}",
        flush=True,
    )
    print("anonymous phases:", sorted(phases), flush=True)
    print(
        "No material identity / parent-product role / OR / correspondence supplied.",
        flush=True,
    )

    print("Cross-locking vectorized edge crystallography...", flush=True)
    assert_vectorized_disorientation_parity(
        data,
        phases,
        graph,
        maximum_edges_per_phase=24,
    )

    settings = PhaseForensicsSettings(
        orientation_thresholds_deg=tuple(
            float(x)
            for x in challenge["settings"]["segmentation_thresholds_deg"]
        ),
        minimum_component_sizes=(1, 2, 3, 5, 10, 20),
        relation_minimum_interface_units=8,
        relation_seed_cap=96,
        relation_refine_seeds=4,
        relation_local_max_iterations=35,
        null_permutations=int(
            challenge["settings"]["null_permutations"]
        ),
        random_seed=int(challenge["settings"]["random_seed"]),
        reliability_component_min_points=2,
        reliability_minimum_spatial_supported_pixel_fraction=0.20,
        reliability_minimum_orientation_supported_pixel_fraction=0.20,
        reliability_minimum_supported_components=8,
    )

    print("Running reliability-gated phase forensics...", flush=True)
    report = run_phase_forensics(
        data,
        phases,
        graph,
        settings=settings,
    )

    compact = compact_phase_forensics_report(report)
    payload = {
        "schema_version": 2,
        "challenge_integrity": {
            "challenge_json_sha256": EXPECTED_CHALLENGE_SHA256,
            "observations_npz_sha256": EXPECTED_OBSERVATIONS_SHA256,
            "zero_clue_contract": "PASS",
        },
        "map_audit": audit_map(data),
        "forensics": compact,
        "interpretation_contract": {
            "phase_extinction_is_never_silent": True,
            "minimum_size_is_not_auto_lowered": True,
            "direct_interface_route_requires_phase_reliability_gate": True,
            "singleton_dominated_populations_are_not_fit": True,
            "direct_interface_route_does_not_assign_transformation_direction": True,
            "fit_holdout_separation": True,
            "holdout_pairing_permutation_null": True,
            "large_point_and_component_arrays_are_not_serialized_to_json": True,
            "all_phase_names_and_roles_remain_anonymous": True,
        },
    }
    serializable = json_safe(payload)
    encoded = (
        json.dumps(
            serializable,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    forbidden_large_array_keys = (
        b'"component_id"',
        b'"component_sizes"',
        b'"component_phase_id"',
    )
    leaked = [
        token.decode("utf-8")
        for token in forbidden_large_array_keys
        if token in encoded
    ]
    if leaked:
        raise RuntimeError(
            "compact summary attempted to serialize component arrays: "
            + ", ".join(leaked)
        )
    if len(encoded) > 2_000_000:
        raise RuntimeError(
            f"compact summary unexpectedly exceeds 2 MB ({len(encoded)} bytes)"
        )

    arrays = phase_forensics_array_payload(report)
    atomic_write_bytes(summary_path, encoded)
    atomic_write_npz(arrays_path, arrays)

    manifest = {
        "schema_version": 1,
        "summary": {
            "path": summary_path.name,
            "sha256": sha256_file(summary_path),
            "bytes": summary_path.stat().st_size,
        },
        "arrays": {
            "path": arrays_path.name,
            "sha256": sha256_file(arrays_path),
            "bytes": arrays_path.stat().st_size,
            "keys": sorted(arrays),
        },
        "challenge": {
            "challenge_json_sha256": EXPECTED_CHALLENGE_SHA256,
            "observations_npz_sha256": EXPECTED_OBSERVATIONS_SHA256,
        },
    }
    atomic_write_json(manifest_path, manifest)

    print("\n=== RELIABILITY-GATED FORENSIC AUDIT COMPLETE ===", flush=True)
    for item in report.phase_survival:
        print(
            f"phase_{item.phase_id}: pixels={item.raw_indexed_pixels:,}, "
            f"raw_components={item.raw_spatial_components:,}, "
            f"largest={item.largest_raw_spatial_component}",
            flush=True,
        )

    for threshold in report.thresholds:
        print(f"\nthreshold={threshold.threshold_deg:g} deg", flush=True)
        for gate in threshold.phase_reliability:
            print(
                f"  phase_{gate.phase_id}: interface_gate="
                f"{'PASS' if gate.interface_inference_allowed else 'REJECT'}; "
                f"raw_supported_fraction={gate.raw_supported_pixel_fraction:.3f}; "
                f"orientation_supported_fraction="
                f"{gate.orientation_supported_pixel_fraction:.3f}; "
                f"supported_components={gate.orientation_supported_components}",
                flush=True,
            )
            if gate.reason_codes:
                print(
                    "    reasons=" + ",".join(gate.reason_codes),
                    flush=True,
                )
        for relation in threshold.direct_relations:
            print(
                f"  pair {relation.phase_a}-{relation.phase_b}: "
                f"{relation.status}",
                flush=True,
            )
            if relation.best is not None:
                print(
                    f"    holdout_median={relation.best.holdout_median_deg:.3f} deg; "
                    f"support={relation.best.holdout_support_fraction:.3f}; "
                    f"p={relation.best.null_empirical_p_value:.5f}",
                    flush=True,
                )

    print("\nsummary:", summary_path, flush=True)
    print("arrays: ", arrays_path, flush=True)
    print("manifest:", manifest_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
