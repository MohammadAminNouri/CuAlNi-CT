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


## 7. Final supercompatibility and natural-OR validation

### CT supercompatibility

Random campaign seed:

`1789465365311682050`

Cases:

`200`

Lattice-family coverage:

- cubic: 51
- hexagonal: 35
- monoclinic: 25
- orthorhombic: 40
- triclinic: 49

The Cayron supercompatibility relation

`2 (m_A^T n) d_A = a`

was tested using independently constructed exact-compatible states and
independently perturbed incompatible states.

Results:

- maximum exact residual: `7.099e-16`
- maximum exact vector error: `7.152e-17`
- maximum projective-scaling error: `9.387e-16`
- incompatible residual range:
  `1.35427e-05 -> 13.4303`
- zero-shear rejection: PASS

No incompatible residual was prescribed in advance.

Result:

`PASS`

### Natural-OR numerical defect discovered and corrected

The original natural-OR scalar distance used

`theta = acos((trace(R)-1)/2)`.

A randomized validation exposed loss of very small rotations:

- requested perturbation: approximately `1e-7 deg`
- production result before correction: `0 deg`
- independent stable result: approximately `1e-7 deg`

The production implementation was changed to a stable SO(3) principal-angle
evaluation using both sine and cosine:

`sin(theta) = ||R-R^T||_F / (2 sqrt(2))`

`cos(theta) = (trace(R)-1)/2`

`theta = atan2(sin(theta), cos(theta))`

Permanent regression coverage includes angles from `1e-12 deg` through
`180 deg`.

### Deterministic natural-OR regression

Seed:

`8138108684858986079`

All four CT operators passed independent proper-symmetry enumeration.

At the original `1e-7 deg` failure scale:

- operator 0:
  production `1.000000024621e-07`
  independent `9.999994569712e-08`
  error `5.676e-14 deg`
- operator 1:
  error `7.094e-14 deg`
- operator 2:
  error `0`
- operator 3:
  error `8.178e-16 deg`

Maximum production/independent ranking error:

`1.998e-13 deg`

Result:

`PASS`

### Fresh random natural-OR campaign

Seed:

`1559166451777332904`

Coverage:

- CT operators: `4`
- independent ranking comparisons: `28`
- exact candidate anchors
- random closing-gap perturbations
- sub-microdegree perturbations
- fully random natural ORs
- independent proper parent/product symmetry enumeration

Results:

- maximum production/independent error:
  `2.326e-13 deg`
- smallest nonzero production distance:
  `9.984560813833e-11 deg`

Critical `1e-7 deg` examples:

- operator 0:
  production `1.000000003845e-07`
  independent `9.999999805047e-08`
  error `2.334e-15 deg`
- operator 1:
  production `9.999994728626e-08`
  independent `9.999989070150e-08`
  error `5.658e-14 deg`
- operator 2:
  error `3.782e-15 deg`
- operator 3:
  error `4.337e-16 deg`

No expected candidate, sign branch, natural OR, or numerical distance was
hard-coded.

Result:

`PASS`

## 8. Cayron CT final validated scope

The frozen Cayron backend now includes validated coverage of:

- rational correspondence;
- direct/reciprocal duality;
- primitive and centered cell representations;
- full cubic 48-operation correspondence symmetry;
- correspondence subgroup H_C;
- variants and double cosets;
- inverse/groupoid operator topology;
- axial weak twins;
- generalized twin index q_g;
- proper and improper reticular orientations;
- generalized shear;
- generalized strain;
- Type-I twins;
- Type-II twins;
- exact intercorrespondence;
- closing-gap orientation construction;
- natural-OR symmetry-reduced ranking;
- sub-microdegree and near-identity SO(3) ranking;
- CMC;
- normalized CMC;
- SMC;
- first-order exact habit planes;
- second-order exact habit planes;
- third-order compatibility degeneracy;
- near-compatible habit-plane projection;
- CT supercompatibility;
- deliberately incompatible supercompatibility residuals.

The following distinction remains mandatory:

`C != T != F != U`

and the polar rotation, Cayron natural OR, experimental OR, Ball-James OR,
and PTMC OR remain separate physical/theoretical objects.

Undocumented GenOVa internal A/B/C/D candidate ordering is still not claimed.
