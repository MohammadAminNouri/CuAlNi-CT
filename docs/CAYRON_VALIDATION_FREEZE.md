# Cayron CT validation freeze

This document freezes the numerical/scientific validation performed after
hardening the Cayron Correspondence-Theory backend.

## 1. Exact/floating crystallographic boundary

Original failing randomized state:

- seed: `5251794611090490145`
- failing case: `270`
- issue: exact rational relation `p2 = C^{-T} p1` was rejected after an
  ill-conditioned floating-point solve.
- resolution: exact crystallographic identities are now checked exactly before
  entering the floating metric/physical layer.

Permanent regression coverage includes:

- exact `C u = u`;
- exact/projective `p2 || C^{-T}p1`;
- exact direct/reciprocal cell transforms;
- exact `|det C| = 1` weak-twin requirement;
- Cayron Eq. `C = T F`;
- generalized shear/strain audits;
- Type-I / Type-II exact crystallographic elements.

---

## 2. Realistic random axial weak-twin campaign

Seed:

`285265694414084447`

Cases:

`200`

Coverage:

- primitive representations: `167`
- C-centered representations: `33`
- cubic: `21`
- hexagonal: `25`
- monoclinic: `33`
- orthorhombic: `31`
- tetragonal: `29`
- triclinic: `28`
- C-centered monoclinic: `33`

Numerical statistics:

- median `cond(C)`: `63.8302`
- p95 `cond(C)`: `14282.7`
- maximum `cond(C)`: `571802`
- median `qg`: `4`
- maximum `qg`: `48`
- worst numerical audit: `2.484486e-14`

Result:

`PASS`

---

## 3. Harsh Cayron CT campaign

Seed:

`232877971231010711`

### Axial weak-twin part

Cases:

`100`

Balanced determinant coverage:

- `det(C)=+1`: `50`
- `det(C)=-1`: `50`

Representation coverage:

- primitive: `72`
- C-centered: `28`

Selected reticular orientation:

- proper `T`: `50`
- improper `T`: `50`

Ranges:

- generalized strain `epsilon_g`: `0.106147 -> 0.44764`
- generalized shear `s_g`: `0.101361 -> 0.444385`

### Full CT part

Cases:

`40`

Transformation coverage:

- product monoclinic: `23`
- product orthorhombic: `17`
- positive determinant correspondence: `20`
- negative determinant correspondence: `20`

Topology / twin coverage:

- correspondence variants: `918`
- correspondence operators: `910`
- Type-I twins: `355`
- Type-II twins: `355`
- closing-gap candidates: `1920`
- approximate-CMC states: `39`

Principal-stretch coverage:

- lambda1: `0.72524 -> 0.95826`
- lambda2: `0.81920 -> 1.12846`
- lambda3: `1.03595 -> 1.34531`

Worst residuals:

- `C_int = C g C^-1`: `0.000e+00`
- closing-gap: `1.171e-14`
- CMC formula: `7.648e-17`
- SMC formula: `6.939e-18`

Result:

`PASS`

---

## 4. Exact CMC / habit-plane campaign

Coverage:

- first-order exact CMC: `40/40 PASS`
- second-order exact CMC: `20/20 PASS`
- third-order exact CMC: `10/10 PASS`
- near-compatible/non-exact CMC: `30/30 PASS`

Numerical results:

- worst normalized-CMC reconstruction: `6.421e-14`
- worst dimensional CMC formula residual: `2.902e-13`
- worst SMC formula residual: `1.089e-15`
- minimum separation of first-order habit planes: `0.508189`
- near-zero residual / planted perturbation:
  - median: `1.000000000`
  - minimum: `1.000000000`

Scientific behavior verified:

### First-order degeneracy
Random normalized CMC eigenstructure:

`q- < 0, q0 = 0, q+ > 0`

Expected theory behavior:

- exact compatibility detected;
- degeneracy order `1`;
- two distinct habit planes recovered;
- returned planes independently satisfy the dimensional CMC cone.

### Second-order degeneracy

Expected theory behavior:

- exact compatibility detected;
- degeneracy order `2`;
- one habit plane recovered.

### Third-order degeneracy

Expected theory behavior:

- normalized CMC vanishes;
- degeneracy order `3`;
- no artificial finite list of unique habit planes returned.

### Near-compatible states

Expected theory behavior:

- NOT classified as exact;
- degeneracy order remains `0`;
- nearest-zero CMC residual recovered;
- approximate candidate planes solve the independently projected nearest-zero cone.

Result:

`PASS`

---

## 5. Scientifically validated Cayron components

The following have now been exercised by unit/regression tests and/or
independent randomized numerical campaigns:

- exact rational correspondence;
- direct / reciprocal duality;
- primitive / conventional representation bridge;
- C-centered representation;
- generalized twin index `qg`;
- axial weak twins;
- `det(C)=+1` and `det(C)=-1`;
- proper and improper reticular orientations;
- Cayron reticular orientation `T`;
- distortion `F`;
- `C = T F`;
- generalized shear;
- generalized strain;
- full cubic `m-3m` 48-operation topology;
- correspondence subgroup `H_C`;
- correspondence variants;
- exact double cosets;
- inverse operators;
- groupoid topology;
- Type-I transformation twins;
- Type-II transformation twins;
- exact intercorrespondence `C_int = C g C^-1`;
- closing-gap proper rotations;
- dimensional CMC;
- normalized CMC;
- SMC;
- first-order exact compatibility;
- second-order exact compatibility;
- third-order exact compatibility;
- exact CT habit planes;
- near-compatible CMC discrimination;
- approximate CMC habit-plane projection.

## 6. Explicitly not claimed

The software does NOT claim to reproduce undocumented internal GenOVa
A/B/C/D candidate ordering.

Correspondence `C`, orientation `T`, deformation `F`, stretch `U`, polar
rotation, natural OR, experimental OR, Ball-James OR, and PTMC OR remain
distinct scientific objects.

This freeze is intended as the validated Cayron backend checkpoint before
the final supercompatibility / natural-OR-ranking work and before theory
unification.
