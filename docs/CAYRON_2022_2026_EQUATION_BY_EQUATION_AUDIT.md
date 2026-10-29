# Cayron 2022 + 2026: independent scientific acceptance and evidence hierarchy

Scope: `feature/twin-family-tree-app-v1` only. Scientific changes preserve the
existing CT, Ball–James and PTMC engines. No output below is a measured
Cu–Al–Ni variant population or experimentally achieved alloy.

## Primary sources reviewed

- C. Cayron (2022), *The Correspondence Theory and Its Application to NiTi
  Shape Memory Alloys*, **Crystals** 12, 130,
  https://doi.org/10.3390/cryst12020130. Sections 3.1–3.3, 4.1–4.4,
  5.1–5.4, Tables 2–5, Appendix A. This governs crystallographic
  correspondence, cosets, operators, generic twins, closing-gap ORs and weak
  planes.
- C. Cayron (2026), *Compatibilities and supercompatibility conditions in
  shape memory alloys determined from correspondence, metrics and
  symmetries*, **Acta Materialia** 316, 122399,
  https://doi.org/10.1016/j.actamat.2026.122399. Sections 2–5,
  Figure 6, Tables 2–3 and Appendix C. This governs IPS compatibility,
  CMC/SMC, Eq.42/43 shear/shear closure, the distinction between CT and
  cofactor conditions, and ideal phase metrics.

## Input conventions that must not be interchanged

2022 paper's printed basis matrix is
`C_A_to_M = [[0,1,-1],[0,1,1],[1,0,0]]`.
In the Python engine the input maps **parent direct crystal coordinates into
product direct crystal coordinates**:
`u_M = C_M_from_A u_A`, with
`C_M_from_A = inverse(C_A_to_M)
= [[0,0,1],[1/2,1/2,0],[-1/2,1/2,0]]`.

Planes are reciprocal covectors and transform by inverse transpose.
Direct shear directions are not interchangeable with plane covectors,
especially in monoclinic 18R/B19′. A correspondence does not uniquely
specify the experimental parent/product OR.

## 2022 correspondence-group theory checklist

1. Compute parent group G_A and correspondence subgroup H_C using exact
   rational matrices and verified symmetry settings.
2. Count correspondence variants M_i by the **left cosets** G_A/H_C.
   Lagrange: `N_M = |G_A|/|H_C|`.
3. Count specific inter-variant operators by **double cosets**
   `H_C\\G_A/H_C`; do not confuse this count with variants or observed
   twin boundaries. The author's O0–O6 numeric names are a GenOVa
   convention, NOT stable identifiers for another app's computed operators.
4. Distinguish a parent reflection's inherited rational Type-I plane K1
   from a parent twofold rotation's inherited rational Type-II direction
   eta2. Irrational conjugates depend on metric parameters.
5. Record weak-plane candidates separately: axial weak twins for higher
   order parent operations are NOT mathematically identical to exact
   Ball–James rank-one M/M connections.
6. Preserve all M-to-U mappings. Distinct stretch U variants generally
   do not equal correspondence M variants by identity or by count.
7. A predicted closing-gap OR must never be relabeled as an experimentally
   observed OR from EBSD/CTF alone.

Published NiTi benchmark (Sec.5, Table 2):
`a_B2=3.015 Å`; B19′ cell `a=2.889, b=4.120, c=4.622 Å,
beta=96.8°`; `|G_A|=48`, `|H_C|=4`,
**12 correspondence variants and 7 double-coset operators**.
The source gives compounds near shears 0.2385 and 0.2389, alongside
Type-I/II twins and higher-order polar weak families. Source-locked
inputs and experimental comparisons live in
`data/benchmarks/cayron_2022_niti_ct_v1.json` and are checked by
`tests/test_ct_permanent_benchmarks.py`.

## 2026 full CMC/SMC and supercompatibility checklist

1. Obtain the **actual supplied** parent and product metrics M_A and
   M_M, correspondence C, and verified symmetry; do not import a
   published NiTi metric into Cu–Al–Ni.
2. Compute **CMC, Eq.32**:
   `K = C^T M_M C − M_A`. The quadratic `u^T K u=0` tests
   preservation of lengths. Generalized eigenvalues of `C^T M_M C`
   relative to `M_A` give the squared principal stretches.
3. Determine the **CMC degeneracy and inertia**, not only the scalar
   middle stretch `lambda2−1`. A first-order indefinite double-plane
   yields two A/M habit plane candidates. A second-order single-plane
   yields one. A third-order identity metric has non-unique choices.
   A degenerate positive-semidefinite null *line* must NOT be
   reported as a compatible plane.
4. Compute **SMC, Eq.41**:
   `S = M_A^-1 − C^-1 M_M^-1 C^-T`;
   for normalized reciprocal habit covector m, **Eq.40**:
   `d_A = S m`. SMC is NOT CMC^-1.
5. Establish M/M transformation twins from correspondence operators
   and parent twofold generators. Preserve Type-I K1 and Type-II
   conjugate K2 semantics; no direct comparison across mixed frames.
6. For each *specific habit and twin*, evaluate **Eq.42**:
   `2(m_A^T n) d_A = s eta_A`, where n is a unit direct-space twin
   normal, eta_A is the unit direct-space twin shear direction,
   and m_A is a unit reciprocal covector. **Eq.49**:
   `epsilon = ||2(m_A^T n) d_A − s eta_A||_A / |s|`.
   Report the full difference vector, signed longitudinal magnitude
   gap, transverse directional gap and angle. A small lambda2 defect
   does not make these gaps zero.
7. Independently evaluate Ball–James cofactor conditions **on that
   same physically matched twin**:
   `CC1: lambda2=1`;
   `CC2: a·U cof(U²−I)n = 0`;
   `CC3: tr(U²)−det(U²)−|a|²|n|²/4−2 ≥ 0`.
   A CT pass and a BJ pass on different twins are never a joint pass.
8. A CT/BJ assignment is eligible only if all required comparable
   observables exist AND their physical geometry agrees within the
   declared tolerances. Cross-state continuity of a physical twin
   must be verified separately, not inferred from array indices.
9. Vary only symmetry-permitted independent product metrics.
   Recompute every proposed cell and quantify changes in entered
   length units / degrees. The target `lambda2=1` is a necessary
   A/M metric target but not a supercompatibility certificate.
   Sampled CT-only epsilon decreases on potentially different twins
   are exploratory, not a material-design prescription.
10. Preserve raw measured/entered metrics, source uncertainty,
    transformation temperature and correspondence provenance.
    Candidate cells are hypothetical and must never inherit the
    original specimen's measured status.

### Independent numerical anchors

2026 Table 2/Figure 7 dimensional-source NiTi:
`a_B2=3.01 Å`, `a_B19′=2.898 Å`,
`c_B19′=4.646 Å`, `beta=97.78°`.
For C1 set `b/a_B2=sqrt(2)`.
Predicted Table-3 C1 habit planes (h=1 normalization) are
`(1,-1,2.41966)` and `(1,-1,-0.31568)`.
Corresponding IPS shear d+ is
`(0.36938,-0.36938,-0.05378)`.
For the O2 compound Type-I member:
`s=0.27325`, `epsilon=0.215`, angle `5.9°`.

2026 Appendix C's ideal supercompatible O4 example at beta=98°:
`a/a0 ≈ 0.8825`, `b/a0 = sqrt(2)`, `c/a0 ≈ 1.6182`.
The independent paper oracle verifies this alongside the O2 Type-II
family. The O2 Type-I formula **as printed** `c=sqrt(2),
a=c/sin(beta)` fails the Eq.41/42 and equivalent Type-I CCI
check for its stated C1 state (squared CCI 1/2 rather than 1).
The **equation-derived, not author-confirmed** correction is
`a=1/sin(beta)` with `b=c=sqrt(2)`; keep the source and derived
formula separately labeled. The reported C2b inequality differs
between the author's Eq.47 and Table 2, and is likewise not silently
reconciled.

## Implementation and test evidence

- `twin_app/paper_methodology_2022_2026.py`: independent CMC/SMC
  inputs, full matrices, sign/degeneracy and Eq.42 vector decomposition.
- `twin_app/metric_design_ui.py`: user-triggered full mathematical
  audit and source-equation measurements, preserving earlier controls.
- `src/cualni_cryst/supercompatibility_atlas.py`: per-branch Eq.42
  vector checks cross-audited against CT scalar epsilon; **no**
  global BOTH label without a full same-pair CT/BJ physical match.
- `twin_app/metric_failure_diagnostics.py`: fail-by-condition analysis
  including vector mismatch and cofactor gate separation.
- `twin_app/variant_catalogue_ui.py`: 2022 symmetry census,
  separately reported M and U variants, operator and pair counts.
- `tests/twin_app/test_cayron_paper_methodology_2026.py`:
  independent Table-3 CMC/SMC values, quadratic degeneracy negative
  controls, length-unit invariance and Eq.42 decomposition.
- `tests/test_ct_permanent_benchmarks.py`: source-locked 2022 CT
  crystallography/variant/twin tests, plus earlier 2026 reference cases.
- `.github/workflows/cayron-2026-independent-oracle.yml`:
  independent paper oracle and specific integration regression gate.

**Not demonstrated:** a globally exhaustive Cu–Al–Ni 2H/modified-18R
supercompatible phase metric, actual controllability through chemistry
or heat-treatment, experimentally measured twin volume fractions, or
validated EBSD-specific reconstructed OR. These are distinct research
questions, not defaults inferred from literature NiTi.
