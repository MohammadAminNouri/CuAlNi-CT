# Dependency + Calculation Service

## Purpose

`ProjectState` owns scientific input truth.

The calculation service owns derived results.

The future GUI should only:

1. construct/edit an immutable `ProjectState`;
2. ask `CalculationService` for a result;
3. render the returned typed result.

It must not implement crystallographic equations in button callbacks.

---

## Why fingerprints are scientific, not merely a performance optimization

The core Cu-Al-Ni architecture separates two kinds of information.

### Discrete topology

For a fixed phase symmetry and correspondence,

\[
G_A,\quad G_M,\quad C
\]

determine:

- correspondence subgroup;
- left-coset variant topology;
- double-coset operator topology;
- adjacency/groupoid structure.

These objects do **not** depend on lattice lengths or monoclinic beta once the
phase/basis/symmetry branch is fixed.

### Metric geometry

The lattice metrics determine:

\[
M_A,\quad M_M,\quad C^TM_MC,
\]

and therefore:

- stretch \(U\);
- principal stretches;
- CMC;
- SMC;
- A/M compatibility;
- metric-dependent stretch variants;
- Cartesian deformation gradient.

Changing beta while retaining the same symmetry/correspondence must therefore
invalidate metric geometry while preserving the discrete groupoid.

The cache is keyed according to exactly this distinction.

This encodes the project's central conceptual statement:

\[
\text{symmetry + correspondence}
\Rightarrow
\text{discrete topology},
\]

while

\[
\text{lattice parameters}
\Rightarrow
\text{metric deformation of that topology}.
\]

---

## Built-in nodes

```text
PROJECT_VALIDATION

DISCRETE_TOPOLOGY
METRIC_CORE
    ├── REPRESENTATION
    ├── AM_COMPATIBILITY
    ├── SCIENTIFIC_CONTRACTS
    └── STRETCH_VARIANTS

TRANSFORMATION_BUNDLE
    aggregates all transformation-level results
```

The dependency graph is explicitly checked for cycles.

---

## Exact group theory remains exact

`group_theory.py` correctly uses SymPy exact matrices for subgroup, coset and
double-coset decisions.

`PhaseState` currently stores symmetry matrices numerically for portable
serialization. The calculation service therefore reconstructs a rational
candidate for every symmetry operator and verifies that it agrees with the
stored matrix within the explicit algebraic tolerance **before** exact group
theory is allowed to use it.

The exact determinant must also be \(+1\) or \(-1\).

An approximate arbitrary matrix is never silently admitted into an exact
coset calculation.

For the locked DO3 -> 6M reference branch the service must recover:

```text
|G_A| = 48
|G_M| = 4
|H_C| = 4
N_C   = 12 correspondence variants
N_O   = 8 double-coset operator classes
```

and the double-coset count is independently checked by the existing Burnside
implementation.

---

## Metric core

All metric-dependent theories consume one common metric result:

\[
M_A,\quad M_M,\quad G_C=C^TM_MC,
\]

\[
CMC=G_C-M_A,
\]

\[
D=M_A^{-1/2}CMC\,M_A^{-1/2},
\]

\[
SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T},
\]

\[
U^2=M_A^{-1/2}G_CM_A^{-1/2}.
\]

The service does not maintain separate hidden CT and Ball-James copies of the
lattice state.

---

## A/M compatibility node

The A/M node independently obtains:

- Cayron CMC classification and exact/approximate habit-plane information;
- Ball-James single-variant rank-one solutions.

The exact Boolean comparison uses the explicit project eigenvalue tolerance.
The Ball-James analytical solver is still retained independently.

If exact solutions exist, CT plane covectors are converted to physical normals
in the symmetric metric Cartesian frame and compared projectively with the
Ball-James habit normals.

The source-rounded James-Hane benchmark must remain non-exact.

A diagnostic approximate CMC plane is not promoted to an exact habit plane.

---

## Stretch variants in the correct frame

`U` is defined in the parent metric-whitened orthonormal frame.

Therefore a crystallographic parent symmetry matrix \(g\) must not be applied
directly as though it were already an orthogonal Cartesian rotation.

The service converts it by

\[
Q=B_A g B_A^{-1},
\qquad
B_A=M_A^{1/2},
\]

checks

\[
Q^TQ=I,
\]

keeps only \(\det Q=+1\), and then generates

\[
U_i=Q_i UQ_i^T.
\]

For the cubic parent there must be 24 proper rotations and 12 distinct
cube-edge stretch variants.

This frame conversion is important for future non-cubic parents.

---

## Selective invalidation

`CalculationService.impact(proposed_project, transformation_id)` reports which
scientific domains change before performing any expensive calculation.

Examples:

### Change martensite beta

Expected:

```text
unchanged:
    discrete_topology

changed:
    metric_core
    representation
    am_compatibility
    scientific_contracts
    stretch_variants
    transformation_bundle
```

### Change only Cartesian display convention

Expected:

```text
changed:
    representation
    transformation_bundle

unchanged:
    discrete_topology
    metric_core
    am_compatibility
    scientific_contracts
    stretch_variants
```

This is the mechanism the future GUI will use for immediate responsiveness.

---

## Cache behavior

Cache entries are addressed by:

```text
calculation kind
transformation ID
scientific fingerprint
```

The fingerprint contains only inputs relevant to that node.

When a new immutable `ProjectState` is installed, old cache entries are not
blindly deleted. If a scientific fingerprint is unchanged, the result can be
reused safely.

Thus changing beta can reuse the groupoid calculation while recomputing CMC.

---

## What is deliberately not forced into this milestone

PTMC/cofactor calculations involving a selected M/M twin pair require a
scientifically explicit `TwinSelectionState` / variant-pair identity.

The current `ProjectState` does not yet contain such a selection.

Therefore this service does **not** manufacture an arbitrary twin simply so it
can display PTMC/cofactor numbers.

Those nodes will be added after the transformation/variant selection model is
explicit.

This is an important reliability rule:

> a theory is calculated only when all of its physically meaningful inputs are
> represented in the project state.

---

## Next milestone

With this service verified, the codebase has the correct backend boundary for
the first PTCLab-like user-facing tool:

**Crystallography Console**

The console will query the same `ProjectState` and `CalculationService` used by
the transformation workspaces, rather than implementing a second set of
geometry equations.
