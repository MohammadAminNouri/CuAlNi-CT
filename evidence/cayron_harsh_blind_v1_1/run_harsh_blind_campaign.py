#!/usr/bin/env python3
from __future__ import annotations

"""
Harsh blind Cayron campaign controller.

This file is validation infrastructure, not a crystallographic theory.

Design:
  * only anonymous solver-visible inputs are accepted;
  * strict allowlists reject any extra field that could carry an answer;
  * correspondence conventions are normalized exactly with SymPy;
  * the two 2006 orientation cases are compiled from independent FCC/BCC
    plane+direction parallelisms (NW and KS), with no published variant count;
  * one Cu-Al-Ni case deliberately enters the correspondence as A<-M and must
    be inverted to the repository's M<-A convention;
  * literature/experimental oracles are not accepted by this program;
  * each transformation case is run twice from byte-different JSON
    serializations and must produce the same sealed prediction;
  * internal identities (Lagrange/Burnside/seals/finite residuals) are checked;
  * no expected variant/operator/twin/OR/habit result is asserted;
  * no files are committed, pushed or tagged.

The program tests what the frozen engine predicts.  It does not compare those
predictions with the papers.  Reveal/comparison is a separate later step.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import sympy as sp


FROZEN_CORE_TAG = "ct-ebsd-reliability-gate-v1"
FROZEN_CORE_COMMIT = "f7c616379108c0ec06d804e4692cd437c35ec4e0"
BACKBONE_TAG = "cayron-first-backbone-v1"

CASE_FILES = (
    "case_001.json",
    "case_002.json",
    "case_003.json",
    "case_004.json",
    "case_005.json",
)

# Strict schema allowlists.  Unknown keys are rejected rather than ignored.
TOP_LEVEL_KEYS = {
    "schema_version", "case_id", "mode", "parent", "product",
    "correspondence", "solver_options", "parallelisms",
}
PHASE_KEYS = {"cell", "point_group"}
CELL_KEYS = {"a", "b", "c", "alpha_deg", "beta_deg", "gamma_deg"}
CORR_KEYS = {"mode", "matrix", "verification_pairs"}
PAIR_KEYS = {"A", "M"}
OPTION_KEYS = {
    "ptmc_mode", "ptmc_dilatational_factor",
    "ball_james_fraction_samples",
    "include_ct_closing_gap", "include_ct_supercompatibility",
}
PARALLEL_KEYS = {
    "reference_first", "moving_first",
    "reference_second", "moving_second",
    "unoriented_directions",
}

# These words are intentionally broader than the backbone's leak scanner.
# They may not appear as JSON keys in a solver-visible case.
FORBIDDEN_KEY_FRAGMENTS = (
    "expected", "oracle", "answer", "ground_truth", "literature",
    "habit", "twin_plane", "twin_direction", "shear_magnitude",
    "variant_count", "operator_count", "observed", "experimental_result",
    "target_result", "successful_solution", "compatibility_result",
    "orientation_relationship_result",
)


class CampaignFailure(RuntimeError):
    pass


def sh(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def git(repo: Path, *args: str) -> str:
    try:
        return sh("git", *args, cwd=repo).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise CampaignFailure(
            f"git {' '.join(args)} failed:\n{exc.stderr}"
        ) from exc


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def find_repo(explicit: str | None) -> Path:
    if explicit:
        repo = Path(explicit).expanduser().resolve()
    else:
        result = sh("git", "rev-parse", "--show-toplevel", check=False)
        if result.returncode == 0 and result.stdout.strip():
            repo = Path(result.stdout.strip()).resolve()
        else:
            candidate = Path("/workspaces/CuAlNi-CT")
            if not candidate.is_dir():
                raise CampaignFailure(
                    "Cannot locate repository. Pass --repo /workspaces/CuAlNi-CT."
                )
            repo = candidate.resolve()
    if not (repo / ".git").exists():
        # Worktrees may use a .git file; accept both file and directory.
        if not (repo / ".git").is_file():
            raise CampaignFailure(f"Not a Git checkout: {repo}")
    return repo


def validate_repo(repo: Path) -> dict[str, str]:
    frozen = git(repo, "rev-list", "-n", "1", FROZEN_CORE_TAG)
    if frozen != FROZEN_CORE_COMMIT:
        raise CampaignFailure(
            f"Frozen core mismatch: {frozen} != {FROZEN_CORE_COMMIT}"
        )

    backbone = git(repo, "rev-parse", f"{BACKBONE_TAG}^{{}}")
    head = git(repo, "rev-parse", "HEAD")
    if head != backbone:
        raise CampaignFailure(
            "Run this campaign from the exact frozen Cayron-first backbone "
            f"checkpoint. HEAD={head}, {BACKBONE_TAG}={backbone}."
        )

    changed_core = git(
        repo, "diff", "--name-only", FROZEN_CORE_TAG, "--", "src/cualni_cryst"
    ).splitlines()
    illegal = [
        p for p in changed_core
        if p and not p.startswith("src/cualni_cryst/io_adapters/")
    ]
    if illegal:
        raise CampaignFailure(
            "Scientific-core drift detected: " + ", ".join(illegal)
        )

    # Tracked modifications are never tolerated.  Untracked files are allowed
    # only if they are clearly this campaign package; they are never imported
    # by the solver.
    tracked_dirty = git(repo, "diff", "--name-only")
    staged_dirty = git(repo, "diff", "--cached", "--name-only")
    if tracked_dirty or staged_dirty:
        raise CampaignFailure(
            "Tracked/staged modifications exist. Freeze or revert them first."
        )

    return {
        "head": head,
        "frozen_core": frozen,
        "backbone_tag": BACKBONE_TAG,
    }


def reject_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            text = str(key).casefold()
            for fragment in FORBIDDEN_KEY_FRAGMENTS:
                if fragment in text:
                    raise CampaignFailure(
                        f"answer-bearing key rejected at {path}.{key}"
                    )
            reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            reject_forbidden_keys(child, f"{path}[{i}]")


def exact_scalar(value: Any) -> sp.Expr:
    if isinstance(value, bool):
        raise CampaignFailure("Boolean is not a crystallographic scalar")
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CampaignFailure("Non-finite scalar")
        return sp.Rational(str(value))
    if isinstance(value, str):
        try:
            return sp.Rational(value)
        except Exception as exc:
            raise CampaignFailure(f"Non-rational exact scalar: {value!r}") from exc
    raise CampaignFailure(f"Unsupported scalar type: {type(value).__name__}")


def exact_matrix(values: Any) -> sp.Matrix:
    if not isinstance(values, list) or len(values) != 3:
        raise CampaignFailure("correspondence matrix must be 3x3")
    rows = []
    for row in values:
        if not isinstance(row, list) or len(row) != 3:
            raise CampaignFailure("correspondence matrix must be 3x3")
        rows.append([exact_scalar(x) for x in row])
    matrix = sp.Matrix(rows)
    if sp.simplify(matrix.det()) == 0:
        raise CampaignFailure("correspondence matrix is singular")
    return matrix


def matrix_json(matrix: sp.Matrix) -> list[list[Any]]:
    out: list[list[Any]] = []
    for i in range(3):
        row: list[Any] = []
        for j in range(3):
            x = sp.simplify(matrix[i, j])
            if x.is_Integer:
                row.append(int(x))
            elif x.is_Rational:
                row.append(f"{int(x.p)}/{int(x.q)}")
            else:
                raise CampaignFailure(
                    f"Compiled correspondence is not exact rational: {x}"
                )
        out.append(row)
    return out


def validate_case_schema(case: Mapping[str, Any]) -> None:
    unknown = set(case) - TOP_LEVEL_KEYS
    if unknown:
        raise CampaignFailure(f"unknown top-level fields: {sorted(unknown)}")

    required = {"schema_version", "case_id", "mode", "parent", "product"}
    missing = required - set(case)
    if missing:
        raise CampaignFailure(f"missing fields: {sorted(missing)}")
    if case["schema_version"] != 1:
        raise CampaignFailure("unsupported case schema_version")
    if case["mode"] not in {"transformation", "orientation_topology"}:
        raise CampaignFailure(f"unsupported case mode: {case['mode']!r}")

    for phase_name in ("parent", "product"):
        phase = case[phase_name]
        if not isinstance(phase, Mapping):
            raise CampaignFailure(f"{phase_name} must be an object")
        unknown_phase = set(phase) - PHASE_KEYS
        if unknown_phase:
            raise CampaignFailure(
                f"{phase_name} unknown fields: {sorted(unknown_phase)}"
            )
        if set(phase) != PHASE_KEYS:
            raise CampaignFailure(
                f"{phase_name} requires exactly cell and point_group"
            )
        cell = phase["cell"]
        if not isinstance(cell, Mapping):
            raise CampaignFailure(f"{phase_name}.cell must be an object")
        unknown_cell = set(cell) - CELL_KEYS
        if unknown_cell:
            raise CampaignFailure(
                f"{phase_name}.cell unknown fields: {sorted(unknown_cell)}"
            )
        if set(cell) != CELL_KEYS:
            raise CampaignFailure(
                f"{phase_name}.cell requires {sorted(CELL_KEYS)}"
            )
        for field, number in cell.items():
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise CampaignFailure(
                    f"{phase_name}.cell.{field} must be numeric"
                )
            if not math.isfinite(float(number)) or float(number) <= 0.0:
                if field in {"alpha_deg", "beta_deg", "gamma_deg"}:
                    if not math.isfinite(float(number)) or not (0.0 < float(number) < 180.0):
                        raise CampaignFailure(
                            f"invalid lattice angle {phase_name}.{field}"
                        )
                else:
                    raise CampaignFailure(
                        f"invalid lattice length {phase_name}.{field}"
                    )

    if case["mode"] == "transformation":
        if "parallelisms" in case:
            raise CampaignFailure("transformation case cannot contain parallelisms")
        if "correspondence" not in case or "solver_options" not in case:
            raise CampaignFailure(
                "transformation case requires correspondence and solver_options"
            )
        corr = case["correspondence"]
        if not isinstance(corr, Mapping):
            raise CampaignFailure("correspondence must be an object")
        unknown_corr = set(corr) - CORR_KEYS
        if unknown_corr:
            raise CampaignFailure(
                f"unknown correspondence fields: {sorted(unknown_corr)}"
            )
        if corr.get("mode") not in {"M_from_A", "A_from_M"}:
            raise CampaignFailure(
                "correspondence.mode must be M_from_A or A_from_M"
            )
        exact_matrix(corr.get("matrix"))
        pairs = corr.get("verification_pairs", [])
        if not isinstance(pairs, list):
            raise CampaignFailure("verification_pairs must be a list")
        for pair in pairs:
            if not isinstance(pair, Mapping) or set(pair) != PAIR_KEYS:
                raise CampaignFailure(
                    "each verification pair must contain exactly A and M"
                )
            for name in ("A", "M"):
                vec = pair[name]
                if not isinstance(vec, list) or len(vec) != 3:
                    raise CampaignFailure(
                        f"verification pair {name} must be a 3-vector"
                    )
                [exact_scalar(x) for x in vec]

        opts = case["solver_options"]
        if not isinstance(opts, Mapping):
            raise CampaignFailure("solver_options must be an object")
        unknown_opts = set(opts) - OPTION_KEYS
        if unknown_opts:
            raise CampaignFailure(
                f"unknown solver option fields: {sorted(unknown_opts)}"
            )
        if opts.get("ptmc_mode") not in {"none", "all_twinning"}:
            raise CampaignFailure("unsupported ptmc_mode")
        n = int(opts.get("ball_james_fraction_samples", 0))
        if n < 2:
            raise CampaignFailure("ball_james_fraction_samples must be >= 2")
        d = float(opts.get("ptmc_dilatational_factor", 0.0))
        if not math.isfinite(d) or d <= 0.0:
            raise CampaignFailure("invalid PTMC dilatational factor")
    else:
        if "correspondence" in case or "solver_options" in case:
            raise CampaignFailure(
                "orientation_topology case must not contain correspondence/options"
            )
        parallels = case.get("parallelisms")
        if not isinstance(parallels, Mapping):
            raise CampaignFailure(
                "orientation_topology case requires parallelisms"
            )
        unknown_parallel = set(parallels) - PARALLEL_KEYS
        if unknown_parallel or set(parallels) != PARALLEL_KEYS:
            raise CampaignFailure(
                "parallelisms must contain exactly the supported fields"
            )


def compile_correspondence(corr: Mapping[str, Any]) -> sp.Matrix:
    given = exact_matrix(corr["matrix"])
    if corr["mode"] == "M_from_A":
        C = sp.simplify(given)
    else:
        C = sp.simplify(given.inv())

    if sp.simplify(C.det()) == 0:
        raise CampaignFailure("compiled C_M_from_A is singular")

    # Exact convention audit against every raw direction pair, if supplied.
    for idx, pair in enumerate(corr.get("verification_pairs", [])):
        uA = sp.Matrix([exact_scalar(x) for x in pair["A"]])
        uM = sp.Matrix([exact_scalar(x) for x in pair["M"]])
        residual = sp.simplify(C * uA - uM)
        if residual != sp.zeros(3, 1):
            raise CampaignFailure(
                f"correspondence convention audit failed for pair {idx}: "
                f"{list(residual)}"
            )

    # Inversion round trip must be exact, not floating.
    if sp.simplify(C.inv().inv() - C) != sp.zeros(3):
        raise CampaignFailure("exact correspondence inversion round-trip failed")
    return C


def phase_project_block(phase: Mapping[str, Any], phase_id: str) -> dict[str, Any]:
    cell = phase["cell"]
    return {
        "phase_id": phase_id,
        "label": phase_id,
        "physical_phase": phase_id,
        "cell_representation": "user_cell",
        "cell": {
            "a": float(cell["a"]),
            "b": float(cell["b"]),
            "c": float(cell["c"]),
            "alpha_deg": float(cell["alpha_deg"]),
            "beta_deg": float(cell["beta_deg"]),
            "gamma_deg": float(cell["gamma_deg"]),
            "length_unit": "angstrom",
        },
        "point_group": str(phase["point_group"]),
    }


def compile_project(case: Mapping[str, Any]) -> tuple[dict[str, Any], sp.Matrix | None]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        # deliberately opaque and identical across literature cases
        "project_id": "blind_project",
        "title": "blind_project",
        "phases": [
            phase_project_block(case["parent"], "A"),
            phase_project_block(case["product"], "M"),
        ],
        "orientations": [],
    }

    if case["mode"] == "orientation_topology":
        payload["transformations"] = []
        return payload, None

    C = compile_correspondence(case["correspondence"])
    payload["transformations"] = [
        {
            "transformation_id": "A_to_M",
            "label": "A_to_M",
            "parent_phase_id": "A",
            "product_phase_id": "M",
            "correspondence_M_from_A": matrix_json(C),
        }
    ]
    return payload, C


def shuffled(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {k: shuffled(v, rng) for k, v in items}
    if isinstance(value, list):
        return [shuffled(v, rng) for v in value]
    return value


def atomic_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload,
        indent=2,
        sort_keys=sort_keys,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def load_backbone(repo: Path):
    path = repo / "validation" / "cayron_first_backbone.py"
    if not path.is_file():
        raise CampaignFailure(f"backbone not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "_blind_cayron_backbone", path
    )
    if spec is None or spec.loader is None:
        raise CampaignFailure("cannot import Cayron backbone")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_prediction_internal(payload: Mapping[str, Any], backbone: Any) -> str:
    seal = backbone.verify_prediction_payload(payload)
    contract = payload["blind_contract"]
    required_false = (
        "oracle_argument_available_to_predictor",
        "expected_result_keys_allowed_in_project_json",
        "stored_orientation_used_as_cayron_natural_or",
        "experimental_rows_used_to_fit_theories",
        "scalar_winner_score_computed",
    )
    for key in required_false:
        if contract.get(key) is not False:
            raise CampaignFailure(f"blind contract violated: {key}")
    if contract.get("one_common_project_state_for_all_theories") is not True:
        raise CampaignFailure("theories did not share one ProjectState")
    if contract.get("theory_backends_called_independently") is not True:
        raise CampaignFailure("theories were not called independently")
    if contract.get("all_discrete_branches_preserved") is not True:
        raise CampaignFailure("discrete theory branches were not preserved")

    topology = payload["prediction"]["cayron"]["topology"]
    if topology.get("lagrange_identity") is not True:
        raise CampaignFailure("Cayron topology failed Lagrange identity")
    if topology["burnside_operator_count"] != topology["operator_count"]:
        raise CampaignFailure("Burnside count disagrees with operator count")

    diag = payload["prediction"]["shared_metric_diagnostics"]
    for key in (
        "metric_orthonormality_residual",
        "generalized_eigen_equation_residual",
        "lambda2_minus_one",
    ):
        value = float(diag[key])
        if not math.isfinite(value):
            raise CampaignFailure(f"non-finite metric diagnostic: {key}")
    return seal


def transformation_run(
    *,
    case: Mapping[str, Any],
    repo: Path,
    backbone: Any,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    project, C = compile_project(case)
    assert C is not None

    case_id = str(case["case_id"])
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    p1 = case_dir / "input_a.json"
    p2 = case_dir / "input_b.json"
    atomic_json(p1, project, sort_keys=True)
    atomic_json(p2, shuffled(project, random.Random(8675309)), sort_keys=False)

    # The two files must differ byte-for-byte, otherwise this is not a real
    # serialization-order challenge.
    if p1.read_bytes() == p2.read_bytes():
        raise CampaignFailure("metamorphic JSON serializations unexpectedly identical")

    opts = case["solver_options"]

    def calculate(project_path: Path) -> dict[str, Any]:
        return backbone.build_prediction(
            project_source=str(project_path),
            transformation_id="A_to_M",
            case_id=case_id,
            ptmc_mode=str(opts["ptmc_mode"]),
            ptmc_base_variant_index=None,
            ptmc_dilatational_factor=float(opts["ptmc_dilatational_factor"]),
            ball_james_fraction_samples=int(opts["ball_james_fraction_samples"]),
            include_ct_closing_gap=bool(opts["include_ct_closing_gap"]),
            include_ct_supercompatibility=bool(opts["include_ct_supercompatibility"]),
            repository_provenance=dict(provenance),
        )

    first = calculate(p1)
    seal_a = validate_prediction_internal(first, backbone)
    second = calculate(p2)
    seal_b = validate_prediction_internal(second, backbone)

    if seal_a != seal_b:
        raise CampaignFailure(
            f"{case_id}: prediction changed under JSON key-order/path perturbation"
        )

    # Scientific-input fingerprints must also be identical.
    input_a = first["seals"]["scientific_input_sha256"]
    input_b = second["seals"]["scientific_input_sha256"]
    if input_a != input_b:
        raise CampaignFailure(
            f"{case_id}: scientific input changed under serialization perturbation"
        )

    # We preserve the first sealed prediction; the second is only a
    # metamorphic determinism challenge.
    prediction_path = case_dir / "prediction.json"
    atomic_json(prediction_path, first)
    p1.unlink()
    p2.unlink()

    return {
        "case_id": case_id,
        "mode": "transformation",
        "scientific_input_sha256": input_a,
        "prediction_sha256": seal_a,
        "prediction_file_sha256": file_sha256(prediction_path),
        "metamorphic_serialization_replay": "PASS",
        "correspondence_M_from_A_exact": matrix_json(C),
        "prediction_path": str(prediction_path),
        # These are solver outputs, not expected values.
        "solver_output_summary": {
            "cayron_topology": {
                "parent_group_order": first["prediction"]["cayron"]["topology"]["parent_group_order"],
                "product_group_order": first["prediction"]["cayron"]["topology"]["product_group_order"],
                "subgroup_order": first["prediction"]["cayron"]["topology"]["correspondence_subgroup_order"],
                "variant_count": first["prediction"]["cayron"]["topology"]["variant_count"],
                "operator_count": first["prediction"]["cayron"]["topology"]["operator_count"],
            },
            "row_counts": first["prediction"]["row_counts"],
            "warnings": first["prediction"]["warnings"],
        },
    }


def orientation_signature(result: Mapping[str, Any]) -> str:
    # Enumeration indices may be arbitrary; hash the canonical physical/audit
    # content after removing orientation IDs that can differ between reruns.
    copy = json.loads(json.dumps(result))
    for candidate in copy.get("candidates", []):
        state = candidate.get("state")
        if isinstance(state, dict):
            state.pop("orientation_id", None)
            state.pop("label", None)
    return canonical_sha256(copy)


def _parse_integer_triplet(text: str, opener: str, closer: str) -> tuple[int, int, int]:
    s = str(text).strip()
    if not (s.startswith(opener) and s.endswith(closer)):
        raise CampaignFailure(f"expected {opener}h k l{closer}, got {text!r}")
    tokens = s[1:-1].split()
    if len(tokens) != 3:
        raise CampaignFailure(f"expected three integer indices, got {text!r}")
    try:
        values = tuple(int(x) for x in tokens)
    except ValueError as exc:
        raise CampaignFailure(f"non-integer crystallographic indices: {text!r}") from exc
    if values == (0, 0, 0):
        raise CampaignFailure(f"zero crystallographic object: {text!r}")
    return values


def _orientation_incidence_preflight(case: Mapping[str, Any]) -> None:
    """Reject an impossible plane+direction OR before invoking the solver.

    For a crystallographic plane (h k l) and direct direction [u v w],
    incidence is the exact reciprocal/direct pairing h*u+k*v+l*w = 0.
    This is a coordinate identity, not a benchmark result.
    """
    p = case["parallelisms"]
    hp = _parse_integer_triplet(str(p["reference_first"]), "(", ")")
    up = _parse_integer_triplet(str(p["reference_second"]), "[", "]")
    hm = _parse_integer_triplet(str(p["moving_first"]), "(", ")")
    um = _parse_integer_triplet(str(p["moving_second"]), "[", "]")
    parent_incidence = sum(h*u for h, u in zip(hp, up))
    product_incidence = sum(h*u for h, u in zip(hm, um))
    if parent_incidence != 0 or product_incidence != 0:
        raise CampaignFailure(
            "orientation input preflight failed: the defining direction must lie "
            "in its stated plane on both sides; "
            f"parent pairing={parent_incidence}, product pairing={product_incidence}"
        )


def orientation_run(
    *,
    case: Mapping[str, Any],
    repo: Path,
    output_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _orientation_incidence_preflight(case)

    # Import from the frozen repository only after the repository audit.
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from cualni_cryst.project_io import project_from_dict
    from cualni_cryst.orientation import OrientationService

    project_payload, _ = compile_project(case)
    loaded = project_from_dict(project_payload)
    project = loaded.project
    project.validate().assert_passed()

    service = OrientationService(project)
    p = case["parallelisms"]

    def solve(moving_second: str):
        report = service.state_from_parallelisms(
            "A", "M",
            str(p["reference_first"]),
            str(p["moving_first"]),
            str(p["reference_second"]),
            moving_second,
            orientation_id="blind_or",
            label="blind_or",
            unoriented_directions=bool(p["unoriented_directions"]),
        )
        candidates = []
        for candidate in report.candidates:
            topology = service.topology(candidate.state)
            candidates.append(
                {
                    "state": candidate.state.to_dict(),
                    "first_parallelism_residual_deg": candidate.first_parallelism_residual_deg,
                    "second_parallelism_residual_deg": candidate.second_parallelism_residual_deg,
                    "topology": topology.to_dict(),
                }
            )
        return {
            "parallelism_report": report.to_dict(),
            "candidates": candidates,
        }

    baseline = solve(str(p["moving_second"]))

    # Projective sign challenge: the same crystallographic axis with the
    # opposite sign must not change physical topology when directions are
    # explicitly treated as unoriented axes.
    tokens = str(p["moving_second"]).strip()[1:-1].split()
    flipped = "[" + " ".join(str(-int(x)) for x in tokens) + "]"
    replay = solve(flipped)

    def topology_multiset(payload: Mapping[str, Any]) -> list[str]:
        signatures = []
        for item in payload["candidates"]:
            topo = item["topology"]
            audit = topo["audit"]
            stable = {
                "full_reference_group_order": audit["full_reference_group_order"],
                "full_moving_group_order": audit["full_moving_group_order"],
                "full_orientation_intersection_order": audit["full_orientation_intersection_order"],
                "full_orientation_variant_count": audit["full_orientation_variant_count"],
                "full_orientation_operator_count": audit["full_orientation_operator_count"],
                "operators": [
                    {
                        "size": op["size"],
                        "ambivalent": op["ambivalent"],
                        "cayron_class": op["cayron_class"],
                        "minimum_crystallographic_disorientation_deg":
                            round(float(op["minimum_crystallographic_disorientation_deg"]), 10),
                    }
                    for op in topo["operators"]
                ],
            }
            signatures.append(canonical_sha256(stable))
        return sorted(signatures)

    if topology_multiset(baseline) != topology_multiset(replay):
        raise CampaignFailure(
            f"{case['case_id']}: orientation topology changed under axis sign reversal"
        )

    case_dir = output_dir / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "mode": "orientation_topology_blind_prediction",
        "case_id": case["case_id"],
        "repository": dict(provenance),
        "blind_contract": {
            "expected_result_supplied": False,
            "oracle_supplied": False,
            "literature_output_supplied": False,
            "input_is_only_phase_symmetry_plus_two_parallelisms": True,
        },
        "prediction": baseline,
    }
    result["prediction_sha256"] = canonical_sha256(
        {
            "case_id": result["case_id"],
            "prediction": result["prediction"],
        }
    )
    path = case_dir / "prediction.json"
    atomic_json(path, result)

    audits = [
        item["topology"]["audit"]
        for item in baseline["candidates"]
    ]
    return {
        "case_id": str(case["case_id"]),
        "mode": "orientation_topology",
        "prediction_sha256": result["prediction_sha256"],
        "prediction_file_sha256": file_sha256(path),
        "axis_sign_metamorphic_replay": "PASS",
        "prediction_path": str(path),
        "solver_output_summary": {
            "candidate_count": len(baseline["candidates"]),
            "topology_audits": audits,
        },
    }


def load_cases(case_dir: Path) -> list[dict[str, Any]]:
    found = sorted(p.name for p in case_dir.glob("case_*.json"))
    if tuple(found) != CASE_FILES:
        raise CampaignFailure(
            f"case file set mismatch. expected={CASE_FILES}, found={tuple(found)}"
        )
    loaded = []
    for name in CASE_FILES:
        path = case_dir / name
        data = json.loads(path.read_text(encoding="utf-8"))
        reject_forbidden_keys(data)
        validate_case_schema(data)
        if data["case_id"] != Path(name).stem:
            raise CampaignFailure(
                f"{name}: case_id does not equal filename stem"
            )
        loaded.append(data)
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).resolve().parent / "cases"),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="default: /tmp/cayron_harsh_blind_<UTC-like epoch>",
    )
    args = parser.parse_args()

    repo = find_repo(args.repo)
    provenance = validate_repo(repo)

    # Deterministic low-level math environment where supported.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    case_dir = Path(args.cases).expanduser().resolve()
    cases = load_cases(case_dir)

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = Path("/tmp") / f"cayron_harsh_blind_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=False)

    # Import only the already frozen backbone.
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))
    backbone = load_backbone(repo)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "campaign": "cayron_harsh_blind_v1",
        "repository": provenance,
        "oracle_loaded": False,
        "literature_outputs_loaded": False,
        "cases": [],
    }

    print("=== HARSH BLIND CAYRON CAMPAIGN ===")
    print("HEAD:", provenance["head"])
    print("core:", provenance["frozen_core"])
    print("oracle supplied: NO")
    print("expected results supplied: NO")
    print("literature result comparison: NOT PERFORMED")
    print()

    for case in cases:
        cid = str(case["case_id"])
        print(f"--- {cid} / {case['mode']} ---", flush=True)
        try:
            if case["mode"] == "transformation":
                result = transformation_run(
                    case=case,
                    repo=repo,
                    backbone=backbone,
                    output_dir=output_dir,
                    provenance=provenance,
                )
            else:
                result = orientation_run(
                    case=case,
                    repo=repo,
                    output_dir=output_dir,
                    provenance=provenance,
                )
        except Exception as exc:
            manifest["cases"].append(
                {
                    "case_id": cid,
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            atomic_json(output_dir / "campaign_manifest.json", manifest)
            print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("Nothing was compared to literature and no expected value was changed.")
            print("Partial manifest:", output_dir / "campaign_manifest.json")
            return 1

        result["status"] = "PASS"
        manifest["cases"].append(result)
        print("prediction:", result["prediction_sha256"])
        if result["mode"] == "transformation":
            topo = result["solver_output_summary"]["cayron_topology"]
            print(
                "solver topology:",
                f"H={topo['subgroup_order']}",
                f"variants={topo['variant_count']}",
                f"operators={topo['operator_count']}",
            )
            print(
                "theory rows:",
                result["solver_output_summary"]["row_counts"]["by_theory"],
            )
        else:
            print(
                "orientation candidates:",
                result["solver_output_summary"]["candidate_count"],
            )
        print()

    manifest["all_cases_passed_internal_blind_checks"] = True
    manifest["campaign_manifest_sha256"] = canonical_sha256(
        {
            "repository": manifest["repository"],
            "cases": manifest["cases"],
            "oracle_loaded": False,
            "literature_outputs_loaded": False,
        }
    )
    manifest_path = output_dir / "campaign_manifest.json"
    atomic_json(manifest_path, manifest)

    print("=== CAMPAIGN FROZEN ===")
    print("output:", output_dir)
    print("manifest:", manifest_path)
    print("manifest sha256:", file_sha256(manifest_path))
    print("scientific comparison with papers: NOT YET PERFORMED")
    print("Next step is to copy/freeze these prediction hashes before opening any oracle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
