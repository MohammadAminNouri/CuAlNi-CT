# Research methodology

## Central principle

Use the same independent crystallographic input for all theories, then compare predictions against experiment.

```text
verified structure + measured metrics + verified correspondence
              │
              ├── Cayron CT
              ├── classical PTMC
              └── Ball–James + cofactor

experiment (EBSD / interface traces / mechanical data)
              │
              └── quantitative prediction residuals
```

The project is a test of CT, not a defense of CT.

## Phase-resolved workflow

1. Record composition, temperature, stress state, processing/ordering state.
2. Identify the martensite branch. Long-period 18R/M18R data and a 6M computational cell must be treated as a cell/correspondence problem, not as arbitrary normalization.
3. Store the original reported lattice parameters and cell convention.
4. Build dimensional `M_A`, `M_M`.
5. Build normalized metric objects in parallel.
6. Recover the exact correspondence from a primary source or paired parent/product experiment.
7. Generate full parent/product symmetry groups.
8. CT: calculate `H_C`, variants, double-coset operators and twin candidates.
9. CT: calculate CMC habit planes, SMC shear and `epsilon_CT`.
10. Ball–James: calculate `U`, stretch variants and rank-one twins.
11. Cofactor: calculate CC1/CC2/CC3 for each relevant twin system.
12. PTMC: independently solve the twinned-laminate habit-plane problem.
13. EBSD: assign variants and operator classes; compare measured boundary/habit traces.
14. Report raw residuals separately; do not hide them in one arbitrary score.

## Recommended residuals

- variant/orientation error [deg]
- intervariant misorientation-angle error [deg]
- misorientation-axis error [deg]
- boundary-trace error [deg]
- A/M habit-plane trace error [deg]
- twin shear relative error
- `|lambda_2-1|`
- normalized CMC distance-to-degeneracy
- `epsilon_CT`
- CC2 residual and CC3 margin

## Composition atlas

Only build an atlas after a literature database exists with source-level provenance and phase labels. Never interpolate through a phase boundary without explicitly modelling the branch change.
