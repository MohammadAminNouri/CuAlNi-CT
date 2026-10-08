from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "benchmarks"

EXPECTED = {
    "bhattacharya_cualni_orthorhombic_habit_v1.json": {
        "input": "a27ba9c16024217b18919ec239eaa41684131db78377e7a003a30990bc4d98d9",
        "expected": "ab2193bcd84de9cb967f9c9a6c70a9a0bb0cd9329f42466d4d952c989aeaf7a1",
    },
    "bhattacharya_niti_twin_habit_v1.json": {
        "input": "d9eda9604583115ca7707b89391a0287549de1270005a605eb569171dddc1bd1",
        "expected": "1c39e7a9f906f026e849355404e12abf648f935d53b558272846d16f53c173d9",
    },
    "niti_higher_order_weak_v1.json": {
        "input": "086fedc141173fcfee9a31279cbba94fad3c67c704c73da21ace4cb670a0bddb",
        "expected": "84a6e573255d4a1a92591abef5241fd0ee1652394676c320b737d183c32d1a7d",
    },
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_validation_manifests_are_immutable():
    for name, hashes in EXPECTED.items():
        data = json.loads((BENCH / name).read_text(encoding="utf-8"))
        assert _canonical_hash(data["input"]) == hashes["input"]
        assert _canonical_hash(data["expected"]) == hashes["expected"]
        assert data["lock"]["input_sha256"] == hashes["input"]
        assert data["lock"]["expected_sha256"] == hashes["expected"]
        assert data["source"]["citation"].strip()
