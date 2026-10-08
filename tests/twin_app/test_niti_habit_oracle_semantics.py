from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "benchmarks"
PHYSICAL = BENCH / "otsuka_ren_2005_niti_typeii_habit_v1.json"
LEGACY = BENCH / "bhattacharya_niti_twin_habit_v1.json"

EXPECTED_INPUT_SHA256 = (
    "d9eda9604583115ca7707b89391a0287549de1270005a605eb569171dddc1bd1"
)
EXPECTED_EXPECTED_SHA256 = (
    "ebfd0b19384749d9bb7b1b298dabcf385dbd1aa1181c6126ed6de0ae42124864"
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _norm3(values: list[float]) -> float:
    return sum(float(value) ** 2 for value in values) ** 0.5


def test_physical_niti_oracle_is_locked_and_keeps_direction_separate_from_b():
    data = json.loads(PHYSICAL.read_text(encoding="utf-8"))
    legacy = json.loads(LEGACY.read_text(encoding="utf-8"))
    expected = data["expected"]

    assert _canonical_hash(data["input"]) == EXPECTED_INPUT_SHA256
    assert _canonical_hash(data["expected"]) == EXPECTED_EXPECTED_SHA256
    assert data["lock"]["input_sha256"] == EXPECTED_INPUT_SHA256
    assert data["lock"]["expected_sha256"] == EXPECTED_EXPECTED_SHA256

    # Exact regression against the CI failure mode:
    # a unit direction must never be treated as physical b.
    assert (
        abs(_norm3(expected["shape_direction_parent_cartesian"]) - 1.0)
        < 5.0e-5
    )
    assert (
        abs(_norm3(expected["habit_normal_parent_cartesian"]) - 1.0)
        < 5.0e-5
    )
    assert 0.0 < float(expected["shape_magnitude"]) < 0.5
    assert "unit direction" in data["source"]["output_semantics"]
    assert "separate shape-strain magnitude" in data["source"]["output_semantics"]

    # No input tuning was used to repair the benchmark.
    assert data["input"] == legacy["input"]
