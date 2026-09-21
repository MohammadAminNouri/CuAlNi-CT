from __future__ import annotations

"""Sealed subprocess campaign for blind OR discovery.

The oracle OR never appears in the challenge JSON/NPZ and is never passed to
the solver subprocess.  It remains only in this evaluator process until after
the solver has exited.

The evaluator scores the inverse problem in the child/child-observable subgroup
embedding quotient, not the narrower physical OR quotient.  The ordinary OR
distance is still reported diagnostically so parent-normalizer gauge shifts are
visible rather than hidden.
"""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.ebsd_theory_bridge import (
    build_theory_library,
    orientation_relationship_distance_deg,
)
from cualni_cryst.ebsd_blind_discovery import (
    blind_observable_embedding_distance_deg,
)
from cualni_cryst.lattice import Lattice


def phase_to_json(phase):
    L = phase.lattice
    return {
        "id": phase.phase_id,
        "name": phase.name,
        "point_group": phase.point_group,
        "lattice": {
            "a": L.a,
            "b": L.b,
            "c": L.c,
            "alpha_deg": L.alpha_deg,
            "beta_deg": L.beta_deg,
            "gamma_deg": L.gamma_deg,
            "length_unit": L.length_unit,
        },
    }


def hidden_case(parent, product, *, seed, n_variants, noise_deg):
    rng = np.random.default_rng(seed)
    oracle_or = Rotation.random(random_state=rng).as_matrix()
    theory = build_theory_library(parent, product, oracle_or)
    count = min(n_variants, theory.n_variants)
    if count < 3:
        raise RuntimeError("scenario has too few variants for a blind campaign")

    parent_g = Rotation.random(random_state=rng).as_matrix()
    indices = np.linspace(0, theory.n_variants - 1, count, dtype=int)
    orientations = []
    for index in indices:
        exact = (
            parent_g
            @ theory.variant_set.variants[int(index)].R_parent_from_product
        )
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        amplitude = rng.normal(scale=np.deg2rad(noise_deg))
        orientations.append(
            Rotation.from_rotvec(amplitude * axis).as_matrix() @ exact
        )
    orientations = np.asarray(orientations)
    adjacency = np.asarray(
        list(itertools.combinations(range(count), 2)),
        dtype=int,
    )
    return oracle_or, orientations, adjacency


def scenarios():
    return [
        (
            "tetragonal_to_orthorhombic",
            EBSDPhase.from_point_group(
                1, "A", Lattice(3.1, 3.1, 5.0), "4/mmm"
            ),
            EBSDPhase.from_point_group(
                2, "M", Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm"
            ),
            dict(seed=151, n_variants=7, noise_deg=0.15),
            dict(global_samples=768, coarse_keep=32, exact_keep=10,
                 local_radius_deg=30.0, local_max_iterations=70,
                 acceptance_median_deg=1.0, support_threshold_deg=2.0,
                 minimum_support_fraction=0.70, sobol_seed=911),
            1.0,
        ),
        (
            "hexagonal_to_orthorhombic",
            EBSDPhase.from_point_group(
                1, "A", Lattice(3.2, 3.2, 5.1, 90, 90, 120), "6/mmm"
            ),
            EBSDPhase.from_point_group(
                2, "M", Lattice.orthorhombic(3.0, 4.2, 5.3), "mmm"
            ),
            dict(seed=811, n_variants=9, noise_deg=0.20),
            dict(global_samples=1024, coarse_keep=36, exact_keep=12,
                 local_radius_deg=30.0, local_max_iterations=80,
                 acceptance_median_deg=1.2, support_threshold_deg=2.5,
                 minimum_support_fraction=0.70, sobol_seed=191),
            1.2,
        ),
        (
            "cubic_to_tetragonal",
            EBSDPhase.from_point_group(
                1, "A", Lattice.cubic(3.0), "m-3m"
            ),
            EBSDPhase.from_point_group(
                2, "M", Lattice(3.0, 3.0, 4.7), "4/mmm"
            ),
            dict(seed=404, n_variants=10, noise_deg=0.20),
            dict(global_samples=1536, coarse_boundary_limit=20,
                 coarse_keep=40, exact_keep=12, local_radius_deg=25.0,
                 local_max_iterations=75, acceptance_median_deg=1.3,
                 support_threshold_deg=2.5, minimum_support_fraction=0.70,
                 sobol_seed=707),
            1.5,
        ),
        (
            "orthorhombic_to_monoclinic",
            EBSDPhase.from_point_group(
                1, "A", Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm"
            ),
            EBSDPhase.from_point_group(
                2, "M", Lattice.monoclinic_unique_b(3.1, 4.2, 5.4, 103.0),
                "2/m"
            ),
            dict(seed=909, n_variants=4, noise_deg=0.10),
            dict(global_samples=1536, coarse_keep=40, exact_keep=14,
                 local_radius_deg=30.0, local_max_iterations=90,
                 minimum_boundaries=4, minimum_distinct_operator_classes=2,
                 acceptance_median_deg=1.0, support_threshold_deg=2.0,
                 minimum_support_fraction=0.65, sobol_seed=313),
            1.5,
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("blind_campaign_report.json"))
    args = parser.parse_args()

    report = {"cases": [], "pass": True}

    with tempfile.TemporaryDirectory(prefix="ebsd-blind-campaign-") as tmp:
        root = Path(tmp)

        for name, parent, product, generator, settings, tolerance in scenarios():
            oracle_or, orientations, adjacency = hidden_case(
                parent, product, **generator
            )

            case_dir = root / name
            case_dir.mkdir()
            np.savez_compressed(
                case_dir / "observations.npz",
                orientations=orientations,
                adjacency=adjacency,
            )
            challenge = {
                "schema_version": 1,
                "data_file": "observations.npz",
                "parent_phase": phase_to_json(parent),
                "product_phase": phase_to_json(product),
                "settings": settings,
            }
            # Deliberately assert that no answer-bearing key is serialized.
            forbidden = {
                "orientation_relationship", "expected_or", "true_or",
                "oracle", "answer",
            }
            assert forbidden.isdisjoint(challenge)
            (case_dir / "challenge.json").write_text(
                json.dumps(challenge, indent=2) + "\n"
            )

            result_path = case_dir / "solver_result.json"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cualni_cryst.ebsd_blind_cli",
                    "solve",
                    str(case_dir / "challenge.json"),
                    "--output",
                    str(result_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            case_report = {
                "name": name,
                "solver_returncode": process.returncode,
                "solver_stderr": process.stderr,
            }
            if process.returncode != 0 or not result_path.exists():
                case_report["pass"] = False
                report["pass"] = False
                report["cases"].append(case_report)
                continue

            solver_result = json.loads(result_path.read_text())
            best = solver_result.get("best")
            if best is None:
                case_report.update(
                    {
                        "pass": False,
                        "solver_status": solver_result.get("status"),
                        "reason": "solver returned no OR candidate",
                    }
                )
                report["pass"] = False
                report["cases"].append(case_report)
                continue

            recovered = np.asarray(
                best["R_parent_from_product"], dtype=float
            )
            physical_distance = orientation_relationship_distance_deg(
                recovered,
                oracle_or,
                parent.proper_symmetry_cartesian,
                product.proper_symmetry_cartesian,
            )
            observable_distance = blind_observable_embedding_distance_deg(
                recovered,
                oracle_or,
                parent.proper_symmetry_cartesian,
                product.proper_symmetry_cartesian,
            )
            passed = (
                observable_distance < tolerance
                and solver_result.get("status") != "inconsistent"
            )
            case_report.update(
                {
                    "pass": passed,
                    "solver_status": solver_result.get("status"),
                    "physical_or_quotient_distance_deg": float(
                        physical_distance
                    ),
                    "blind_observable_embedding_distance_deg": float(
                        observable_distance
                    ),
                    "required_observable_less_than_deg": float(tolerance),
                    "median_boundary_residual_deg": float(
                        best["median_residual_deg"]
                    ),
                    "support_fraction": float(best["support_fraction"]),
                    "distinct_operator_classes": int(
                        best["n_distinct_accepted_operator_classes"]
                    ),
                }
            )
            report["pass"] = report["pass"] and passed
            report["cases"].append(case_report)

    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
