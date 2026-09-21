# Cayron CT blind-validation and generality campaign

## Purpose

This campaign has three simultaneous goals:

1. make the original Chen 2H blind case an immutable regression benchmark;
2. test the CT implementation against published Cayron examples without using
   reported outputs as solver inputs;
3. stress the same backend on crystal systems and coordinate representations
   unrelated to Cu-Al-Ni.

The campaign does not modify the Cayron CT correspondence, Type-I,
Type-II, CMC/SMC or shear equations. During the coordinate-covariance test it
exposed a numerical instability in the compound-twin classification comparator.
That comparator was replaced by a metric-native projective chord residual;
the theoretical CT equations themselves remain unchanged.

## Input/output firewall

Every versioned benchmark stores `input` and `expected` separately. Tests first
construct the calculation from `input`, then compare the completed result with
`expected`. Both blocks are permanently hash-locked.

This prevents an apparently successful validation that simply feeds a
published twin plane, shear or operator count into the solver.

## Permanent Chen 2000 2H benchmark

Frozen original blind input:

- parent cubic `a0 = 5.836 A`, point group `m-3m`;
- product 2H orthorhombic `a = 4.382 A`, `b = 5.356 A`,
  `c = 4.222 A`, point group `mmm`;
- package correspondence
  `C_M_from_A = [[0,1,1],[1,0,0],[0,1,-1]]`.

Locked outputs:

- correspondence subgroup order `8`;
- `6` variants;
- `3` correspondence operators;
- `{101}` / `[10-1]` is recovered through both Type-I and Type-II
  constructions and classified as **compound**;
- `{121}` is retained as an ordinary Type-I system with
  `s = 0.26066690926341635`;
- its distinct Type-II conjugate has
  `K2 ~ (1, 1.503634, 0.503634)`, `eta2 ~ [1 -1 1]`.

The Chen experimental twin observations remain comparison targets; they are
not solver inputs.

## Cayron primary-source cases

### 2006 — groupoid of orientational variants

Source: C. Cayron, *Acta Crystallographica A* 62 (2006) 21-40,
DOI `10.1107/S010876730503686X`.

The Burgers OR is constructed from the two published parallelisms only. The
code then finds the orientation intersection subgroup inside the explicit
crystallographic point groups and applies the repository's exact left-coset
and double-coset machinery.

Locked targets: intersection order `4`, `12` variants, `7` operators.

### 2019 — transformation matrices and correspondence variants

Source: C. Cayron, *Acta Crystallographica A* 75 (2019) 411-437,
DOI `10.1107/S205327331900038X`.

The fcc/bcc Bain correspondence is converted once at the package convention
boundary (`u_M = C_M_from_A u_A`). The published answer is hidden during
construction.

Locked target: `3` correspondence variants.

The campaign also follows the corrigendum-level distinction between a
crystallographic point group and an unrestricted metric-isometry set:
symmetries come from explicit crystallographic point groups.

### 2022 — Correspondence Theory applied to NiTi

Source: C. Cayron, *Crystals* 12 (2022) 130,
DOI `10.3390/cryst12020130`.

Inputs are B2/B19' lattice parameters, point groups and correspondence.
Topology and twinning elements are calculated before comparison.

Locked targets include:

- intersection order `4`, `12` variants, `7` operators;
- compound `(100)/[001]` and conjugate `(001)/[100]`, `s ~ 0.2385`;
- Type-I `(-111)`, `s ~ 0.3096`, and its Type-II conjugate;
- Type-I `(111)`, `s ~ 0.1422`, and its Type-II conjugate;
- Type-I `(011)`, `s ~ 0.2804`, and its Type-II conjugate.

Matching is by crystallographic geometry and route, never by Cayron's
operator number, because operator numbering depends on enumeration order.

The published natural OR is explicitly marked
`not_blind_from_correspondence_alone`. It is not falsely promoted to a
prediction from `C + metrics`.

### 2022 — axial weak twins

The already-existing `tests/test_weak_twins.py` remains part of this campaign.
It independently reproduces Cayron's two published Mg a-axis weak-twin
examples, including generalized twin index, generalized shear and generalized
strain. This is intentionally reused rather than duplicated.

Scientific boundary: this is a numerical/theoretical regression of the 2022
weak-twin equations/examples, not an endorsement of every earlier experimental
interpretation used to motivate them. In a 2023 reply Cayron agreed that the
specific 2018 interpretation of the reported Mg twins was not correct. The
software benchmark therefore tests the mathematics it implements, not that
superseded microstructural interpretation.

### 2026 — CMC compatibility / NiTi C1 state

Source: C. Cayron, *Acta Materialia* 316 (2026) 122399,
DOI `10.1016/j.actamat.2026.122399`.

The normalized C1 B19' lattice and correspondence are supplied. The backend
must independently obtain:

- first-order CMC compatibility;
- inertia `(negative, zero, positive) = (1,1,1)`;
- two exact habit-plane solutions;
- compound `(100)/[001]` shear `s ~ 0.27325`.

## Generality campaign

Deterministic systems:

- cubic `m-3m` -> tetragonal `4/mmm`;
- cubic `m-3m` -> orthorhombic `mmm`;
- hexagonal `6/mmm` -> orthorhombic `mmm`;
- monoclinic `2/m` -> monoclinic `2/m`;
- monoclinic `2/m` -> triclinic `-1`;
- triclinic `-1` -> triclinic `1` as a legitimate zero-twin state.

For every case the tests require:

- exact finite-group topology is self-consistent;
- point-group operations preserve their actual non-Cartesian metrics;
- subgroup order obeys Lagrange;
- cosets/operators partition the parent group;
- Burnside count agrees with explicit double-coset enumeration;
- CMC spectrum is finite;
- every generated twin has finite shear, correctly normalized reciprocal/direct
  elements, exact plane/direction incidence, and a consistent
  Type-I/Type-II/compound label.

A fixed-seed 24-state campaign randomizes lattice parameters and exact
correspondences across these crystal-system classes. It checks invariants only,
not invented "literature" answers.

## Coordinate-covariance test

A NiTi state is independently re-expressed under different exact unimodular
basis changes in parent and product phases:

`M' = P^T M P`, `G' = P^-1 G P`, `C' = P_M^-1 C P_A`.

The test requires unchanged subgroup/variant/operator counts, CMC eigenvalues,
and the complete multiset of twin route/classification/shear values. This is a
direct guard against hidden tuning to a conventional axis setting.

## Pathological / degeneracy boundary

The campaign explicitly tests:

- third-order CMC degeneracy;
- second-order CMC degeneracy;
- first-order exact compatibility;
- near-compatible but non-exact CMC;
- fully non-compatible CMC;
- singular correspondence rejection;
- invalid lattice rejection;
- symmetry/metric incompatibility rejection.

## Scope boundary

This campaign validates the current CT mathematical/software backbone. It does
not claim that every Cayron paper has the same input assumptions or that a
natural atomistic hard-sphere path can be inferred from lattice correspondence
alone. Cayron's 2024 B2 -> B19' hard-sphere model is therefore not converted
into a fake generic-CT regression; it belongs to a separate atomistic/distortion
model validation layer.
