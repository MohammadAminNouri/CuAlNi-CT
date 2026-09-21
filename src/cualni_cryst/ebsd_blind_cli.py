from __future__ import annotations

"""Subprocess-safe CLI for blind OR discovery.

The challenge file contains phase crystallography, measured child orientations
and adjacency, but deliberately contains no expected OR.
"""

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from .ebsd_blind_discovery import (
    BlindORSettings,
    discover_orientation_relationship_blind,
)
from .ebsd_map import EBSDPhase
from .lattice import Lattice


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _phase(spec: Mapping[str, Any]) -> EBSDPhase:
    lattice_spec = spec["lattice"]
    lattice = Lattice(
        float(lattice_spec["a"]),
        float(lattice_spec["b"]),
        float(lattice_spec["c"]),
        float(lattice_spec.get("alpha_deg", 90.0)),
        float(lattice_spec.get("beta_deg", 90.0)),
        float(lattice_spec.get("gamma_deg", 90.0)),
        str(lattice_spec.get("label", "")),
        str(lattice_spec.get("length_unit", "")),
    )
    return EBSDPhase.from_point_group(
        int(spec["id"]),
        str(spec["name"]),
        lattice,
        str(spec["point_group"]),
    )


def solve(challenge_path: Path, output_path: Path) -> None:
    challenge = json.loads(challenge_path.read_text())
    forbidden = {
        "orientation_relationship",
        "expected_or",
        "true_or",
        "oracle",
        "answer",
    }
    overlap = sorted(forbidden.intersection(challenge))
    if overlap:
        raise ValueError(
            f"Blind challenge contains forbidden answer-bearing keys: {overlap}"
        )

    data_path = (challenge_path.parent / challenge["data_file"]).resolve()
    payload = np.load(data_path)
    orientations = np.asarray(payload["orientations"], dtype=float)
    adjacency = np.asarray(payload["adjacency"], dtype=int)

    parent = _phase(challenge["parent_phase"])
    product = _phase(challenge["product_phase"])
    settings = BlindORSettings(**challenge.get("settings", {}))

    result = discover_orientation_relationship_blind(
        orientations,
        [tuple(int(x) for x in row) for row in adjacency],
        parent,
        product,
        settings=settings,
    )
    output_path.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cualni_cryst.ebsd_blind_cli"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    solve_parser = sub.add_parser("solve")
    solve_parser.add_argument("challenge", type=Path)
    solve_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        solve(args.challenge.resolve(), args.output.resolve())
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
