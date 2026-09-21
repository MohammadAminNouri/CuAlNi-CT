#!/usr/bin/env bash
set -euo pipefail

REPO="/workspaces/CuAlNi-CT"
BRANCH="fix/compound-twin-classification"
TAG="ct-chen2000-blind-freeze-2026-09-20"
BASE="144583ee00839ec518314b3a063d13f11aac95d2"

cd "$REPO"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$BRANCH" ]]; then
  echo "ERROR: expected branch $BRANCH, found $current_branch" >&2
  exit 1
fi

# Preserve the actual blind input as validation data; discard the temporary installer.
mkdir -p data/validation docs/validation docs/generated
if [[ -f chen_2000_blind_cayron.json ]]; then
  mv chen_2000_blind_cayron.json data/validation/chen_2000_blind_cayron.json
fi
rm -f apply_compound_twin_classification.py

if [[ ! -f data/validation/chen_2000_blind_cayron.json ]]; then
  echo "ERROR: data/validation/chen_2000_blind_cayron.json not found" >&2
  exit 1
fi

# First: repository-wide scientific regression.
git diff --check
python -m pytest -q

# Second: rerun the exact original Chen blind input through the patched CT backend.
python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cualni_cryst.cualni_models import do3_to_2h_branch
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.lattice import Lattice
from cualni_cryst.twinning_ct import twins_from_operator

src = Path("data/validation/chen_2000_blind_cayron.json")
out = Path("docs/generated/CHEN_2000_BLIND_CT_FINAL.json")
d = json.loads(src.read_text())
i = d["input"]

branch = do3_to_2h_branch()
expected_C = np.asarray(i["C_M_from_A"], dtype=float)
actual_C = np.asarray(branch.correspondence.C_m_from_a, dtype=float)
if not np.array_equal(expected_C, actual_C):
    raise AssertionError(
        "Blind JSON correspondence differs from the frozen DO3->2H branch"
    )

A = Lattice.cubic(float(i["parent"]["a0_A"]), length_unit="angstrom")
M = Lattice.orthorhombic(
    float(i["martensite"]["a_A"]),
    float(i["martensite"]["b_A"]),
    float(i["martensite"]["c_A"]),
    length_unit="angstrom",
)

g = correspondence_groupoid(
    list(branch.parent_point_group),
    list(branch.product_point_group),
    branch.correspondence,
)

assert len(g.subgroup) == 8, len(g.subgroup)
assert g.n_variants == 6, g.n_variants
assert g.n_operators == 3, g.n_operators

by_operator = []
all_twins = []
for op_index, operator in enumerate(g.operators):
    twins = twins_from_operator(operator, A.metric(), M.metric(), branch.correspondence)
    by_operator.append((op_index, twins))
    all_twins.extend((op_index, t) for t in twins)


def parallel(v, target, tol=1.0e-7):
    v = np.asarray(v, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    return np.linalg.norm(np.cross(v, target)) <= tol * np.linalg.norm(v) * np.linalg.norm(target)


def canonical_ratio(v):
    v = np.asarray(v, dtype=float).reshape(3)
    nz = np.flatnonzero(np.abs(v) > 1.0e-12)
    if not len(nz):
        raise ValueError("zero vector")
    r = v / v[nz[0]]
    if r[nz[0]] < 0:
        r = -r
    return r

family_101 = [(op, t) for op, t in all_twins if parallel(t.plane_m, [1, 0, 1])]
assert family_101, "{101} family not recovered"
assert {t.kind for _, t in family_101} == {"I", "II"}
assert all(t.compound for _, t in family_101)
assert all(t.representations == ("I", "II") for _, t in family_101)
assert all(parallel(t.direction_m, [1, 0, -1]) for _, t in family_101)

family_121 = [(op, t) for op, t in all_twins if parallel(t.plane_m, [1, 2, 1])]
assert family_121, "{121} family not recovered"
type_i_121 = next(t for _, t in family_121 if t.kind == "I" and not t.compound)
assert np.isclose(type_i_121.shear, 0.26066690926341635, rtol=0.0, atol=5.0e-12)

conjugates = [
    (op, t)
    for op, t in all_twins
    if t.kind == "II"
    and not t.compound
    and parallel(t.direction_m, [1, -1, 1])
]
assert conjugates, "ordinary Type-II [1 -1 1] conjugate not recovered"

# Select the one whose plane ratio is closest to the classical reported element.
target = np.array([1.0, 1.5036, 0.5036])
op_ii, type_ii = min(
    conjugates,
    key=lambda x: float(np.linalg.norm(canonical_ratio(x[1].plane_m) - target)),
)
ratio_ii = canonical_ratio(type_ii.plane_m)
assert np.allclose(ratio_ii, target, atol=1.0e-4, rtol=0.0), ratio_ii

result = {
    "freeze": {
        "date": "2026-09-20",
        "purpose": "Blind Cayron CT validation against Chen 2000 EBSD relations; experiment not used as solver input",
        "classification_policy": "CTTwin.kind preserves Type-I/II construction route; physical classification is separate",
    },
    "input": i,
    "topology": {
        "H_C_order": len(g.subgroup),
        "variants": g.n_variants,
        "operators": g.n_operators,
    },
    "blind_predictions": {
        "101": {
            "classification": "compound",
            "construction_routes": sorted({t.kind for _, t in family_101}),
            "representations": ["I", "II"],
            "plane_family_2H": [1, 0, 1],
            "direction_family_2H": [1, 0, -1],
            "shears": sorted({round(float(t.shear), 15) for _, t in family_101}),
        },
        "121": {
            "classification": "type_I",
            "plane_family_2H": [1, 2, 1],
            "shear": float(type_i_121.shear),
        },
        "type_II_conjugate": {
            "classification": type_ii.classification,
            "plane_ratio_2H": [float(x) for x in ratio_ii],
            "direction_family_2H": [1, -1, 1],
            "shear": float(type_ii.shear),
            "operator_index": op_ii,
        },
    },
    "verdict": {
        "CT_topology": "PASS",
        "101_geometry": "PASS",
        "101_compound_classification": "PASS",
        "121_geometry": "PASS",
        "121_shear": "PASS",
        "type_II_conjugate": "PASS",
        "AM_orientation_relationship": "NOT_BLINDLY_VALIDATED_IN_THIS_TEST",
    },
}

out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["topology"], indent=2))
print(json.dumps(result["blind_predictions"], indent=2))
print("BLIND CT FREEZE: PASS")
PY

cat > docs/validation/CHEN_2000_BLIND_CT_FREEZE.md <<'MD'
# Chen 2000 blind Cayron CT validation freeze

Date: 2026-09-20

## Scope

This checkpoint freezes a blind martensite/martensite validation of the Cayron CT backend for the DO3 -> 2H Cu-Al-Ni case.

The solver input is restricted to the parent/product lattice state, exact phase symmetries, and correspondence stored in `data/validation/chen_2000_blind_cayron.json`. Chen's EBSD twin labels are not supplied to the CT calculation.

## Frozen architectural rule

`CTTwin.kind` remains the Cayron construction route (`I` or `II`). Physical twin classification is stored separately as `type_I`, `type_II`, or `compound`. A compound relation therefore retains both construction provenances instead of overwriting them.

Compound recognition is CT-native: the same complete twin geometry and shear must be independently generated through both Type-I and Type-II constructions. Ball-James/Mallard is retained only as an independent cross-lock elsewhere in the project.

## Blind result

For the exact stored input:

- correspondence subgroup order: 8
- correspondence variants: 6
- operator classes: 3
- `{101}_2H` is recovered as a compound twin, with Type-I and Type-II representations and direction family `[10-1]_2H`
- `{121}_2H` is recovered as an ordinary Type-I family
- `{121}_2H` shear: `0.26066690926341635`
- ordinary Type-II conjugate is recovered with direction `[1-11]_2H` and plane ratio approximately `(1, 1.5036, 0.5036)`

The exact machine-readable rerun is frozen in `docs/generated/CHEN_2000_BLIND_CT_FINAL.json`.

## Interpretation boundary

This test supports independent recovery of the martensite/martensite twin relationships. It does **not** constitute a blind validation of the parent/martensite orientation relationship because the correspondence is supplied as an input and already carries crystallographic ancestry.

## Regression requirement

The freeze is valid only when the full repository test suite passes. No CT equation, correspondence topology, CMC mathematics, shear equation, or orientation equation was altered to obtain the compound label; the change is a classification/provenance layer on top of independently generated CT Type-I/II solutions.
MD

# Verify the generated freeze files and all tests one last time.
python -m json.tool docs/generated/CHEN_2000_BLIND_CT_FINAL.json >/dev/null
git diff --check
python -m pytest -q

# Commit only the scientific patch, tests, original blind input, and freeze records.
git add \
  src/cualni_cryst/twinning_ct.py \
  src/cualni_cryst/theory_unified.py \
  tests/test_ct_twinning.py \
  tests/test_do3_6m_twin_atlas.py \
  data/validation/chen_2000_blind_cayron.json \
  docs/validation/CHEN_2000_BLIND_CT_FREEZE.md \
  docs/generated/CHEN_2000_BLIND_CT_FINAL.json

git status --short

git commit -m "fix: classify compound CT twins and freeze Chen blind validation"

# Create an immutable annotated checkpoint.
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "ERROR: tag $TAG already exists" >&2
  exit 1
fi

git tag -a "$TAG" -m "Validated Chen 2000 blind CT checkpoint: {101} compound, {121} Type-I, full tests green"

FREEZE_SHA="$(git rev-parse HEAD)"
echo "FREEZE COMMIT: $FREEZE_SHA"
echo "FREEZE TAG:    $TAG"

# Save remotely. If network/authentication fails, the local commit + tag remain intact.
git push -u origin "$BRANCH"
git push origin "$TAG"

echo
echo "=== FROZEN AND SAVED ==="
git status --short
git log -1 --oneline
git tag --points-at HEAD
