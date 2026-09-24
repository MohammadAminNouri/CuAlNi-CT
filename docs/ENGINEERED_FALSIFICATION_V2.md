# Engineered Falsification V2

This campaign is designed to find scientific and numerical defects, not to
accumulate green literature reproductions.

## Separation of trust

**Independent oracle modules** do not import `cualni_cryst`.  Physical truth is
created in Cartesian space first.  Only afterward is it encoded into arbitrary
crystallographic bases through

    F B_A = B_M C.

Thus the expected principal stretches and rank-one compatibility do not come
from production CMC/SMC formulas.

A second exact oracle uses rational rotations, rational stretches, exact SymPy
metrics, and the exact generalized characteristic polynomial.  This catches
common-mode floating-point and convention errors.

## Falsification architecture

1. physical-first hidden truth;
2. independent exact/high-precision oracle;
3. production adapters;
4. theorem-level contracts;
5. metamorphic representation families;
6. mutation score;
7. conditioning-aware budgets;
8. automatic failure classification;
9. greedy counterexample shrinking;
10. persistent JSON failure ledger;
11. PTMC differential oracle using scan/bracketing rather than production's
    polynomial reconstruction;
12. end-to-end JSON project loader and CalculationService boundary.

## Mutation criterion

The framework deliberately evaluates wrong scientific implementations:
wrong pullback order, wrong direct/reciprocal plane map, wrong SMC
correspondence direction, wrong basis-change law, wrong CC3 determinant sign,
and wrong PTMC singular-value selection.  Every mutant must be killed.

A surviving mutant means the validation framework has a blind spot.

## Failure handling

Do not change a production tolerance merely to make a generated case pass.

The runner stores failures in:

    /tmp/cualni_engineered_failures/

Each ledger records the original input, condition numbers, failing contract,
residual/budget, and a greedily minimized reproduction when possible.

## Profiles

- `quick`: installation/preflight.
- `full`: hundreds to thousands of physical-first unseen problems plus all
  structural differential tests.
- `torture`: tens of thousands of unseen physical problems and thousands of
  metamorphic families.

Run:

    python tools/run_engineered_falsification_v2.py --profile quick
    python tools/run_engineered_falsification_v2.py --profile full
    python tools/run_engineered_falsification_v2.py --profile torture

The existing production solver is intentionally not modified by the installer.
A red result is the desired starting point for defect isolation.
