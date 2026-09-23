# Professor-readiness hardening v1

This layer is intentionally additive. It starts from the validated
`next/experimental-validation` checkpoint and does not modify CT, Ball-James,
PTMC, weak-twin, orientation, group-theory, or EBSD equations.

## Why

The blind campaign exposed a recurrent failure class: a scientifically correct
backend can still be fed the wrong *convention* by a harness or future GUI.

This layer moves those checks to the human-input boundary.

## New adversarial coverage

1. correspondence direction reversal;
2. three-vector correspondence reconstruction;
3. direct/reciprocal duality and incidence;
4. cubic/tetragonal/orthorhombic/hexagonal/monoclinic/triclinic cells;
5. every supported Cartesian convention pair;
6. seeded random valid triclinic cells;
7. seeded random exact unimodular correspondences;
8. common length-scale invariance;
9. exact crystallographic operator orders 1/2/3/4/6;
10. rejection of false symmetry operations under the supplied metric;
11. generalized twin-index invariance under unimodular basis changes;
12. refusal to assign `q_g` to non-equal-volume maps;
13. malformed, singular, non-finite and convention-ambiguous inputs.

## Profiles

Fast:

```bash
python tools/run_professor_hardening.py --quick
```

Core:

```bash
python tools/run_professor_hardening.py --core
```

Final pre-freeze:

```bash
python tools/run_professor_hardening.py --full
```

A passing randomized/property sweep is not a proof for literally every possible
3-D input. The goal is exact algebraic contracts + metamorphic invariance +
broad deterministic adversarial input + permanent literature benchmarks.
