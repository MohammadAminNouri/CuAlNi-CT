from __future__ import annotations

"""Run phase-survival forensics on a sealed anonymous EBSD challenge."""

from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from cualni_cryst.ebsd_analysis import build_neighbor_graph
from cualni_cryst.ebsd_map import EBSDMap, EBSDPhase, audit_map
from cualni_cryst.ebsd_phase_forensics import (
    PhaseForensicsSettings,
    assert_component_segmentation_parity,
    assert_vectorized_disorientation_parity,
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
        return float(value)
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
            orientations=np.asarray(
                payload["orientations"], dtype=float
            ),
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
            "data/real_blind_challenge [report.json]",
            file=sys.stderr,
        )
        return 2

    root = Path(sys.argv[1]).resolve()
    output = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) == 3
        else Path("zero_clue_phase_forensics.json").resolve()
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

    # Gate the optimized forensic primitives against the frozen scalar backend
    # on this real map before trusting the full report.
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
    )

    print("Running phase-survival + direct-interface forensics...", flush=True)
    report = run_phase_forensics(
        data,
        phases,
        graph,
        settings=settings,
    )

    payload = {
        "schema_version": 1,
        "challenge_integrity": {
            "challenge_json_sha256": EXPECTED_CHALLENGE_SHA256,
            "observations_npz_sha256": EXPECTED_OBSERVATIONS_SHA256,
            "zero_clue_contract": "PASS",
        },
        "map_audit": audit_map(data),
        "forensics": report,
        "interpretation_contract": {
            "phase_extinction_is_never_silent": True,
            "minimum_size_is_not_auto_lowered": True,
            "direct_interface_route_does_not_assign_transformation_direction": True,
            "fit_holdout_separation": True,
            "holdout_pairing_permutation_null": True,
            "all_phase_names_and_roles_remain_anonymous": True,
        },
    }
    output.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"
    )

    print("\n=== FORENSIC AUDIT COMPLETE ===", flush=True)
    for item in report.phase_survival:
        print(
            f"phase_{item.phase_id}: pixels={item.raw_indexed_pixels:,}, "
            f"raw_components={item.raw_spatial_components:,}, "
            f"largest={item.largest_raw_spatial_component}",
            flush=True,
        )
    for warning in report.warnings:
        print("WARNING:", warning, flush=True)

    for threshold in report.thresholds:
        print(
            f"\nthreshold={threshold.threshold_deg:g} deg",
            flush=True,
        )
        for route in threshold.pair_routes:
            print(
                f"  pair {route.phase_a}-{route.phase_b}: "
                f"raw_edges={route.raw_cross_phase_pixel_edges:,}, "
                f"interface_units={route.orientation_interface_units:,}, "
                f"direct_route={route.direct_interface_route_possible}",
                flush=True,
            )
        for relation in threshold.direct_relations:
            if relation.best is None:
                print(
                    f"  relation {relation.phase_a}-{relation.phase_b}: "
                    f"{relation.status}",
                    flush=True,
                )
            else:
                print(
                    f"  relation {relation.phase_a}-{relation.phase_b}: "
                    f"{relation.status}; "
                    f"fit_med={relation.best.fit_median_deg:.3f} deg; "
                    f"holdout_med={relation.best.holdout_median_deg:.3f} deg; "
                    f"holdout_support={relation.best.holdout_support_fraction:.3f}; "
                    f"p={relation.best.null_empirical_p_value:.5f}",
                    flush=True,
                )

    print("\nreport:", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
