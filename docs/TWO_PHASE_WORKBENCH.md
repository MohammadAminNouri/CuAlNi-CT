# Two-Phase Crystallography Workbench

This is the general PTCLab-style layer of CuAlNi-CT.

It is intentionally **not a Cayron-only interface**.

The workbench consumes a physical orientation relationship and provides
ordinary two-crystal crystallography:

- direction ↔ direction angles;
- plane ↔ plane angles;
- direction ↔ plane angles;
- direct-vector length and interplanar-spacing comparison;
- relative geometric misfit;
- direct-coordinate transformation between crystals;
- reciprocal-plane transformation between crystals;
- nearest low-index target objects with explicit angular mismatch;
- crystallographic OR-variant angle tables;
- symmetry-equivalent cross-phase angle tables;
- PTCLab-compatible Cartesian OR display (`x || a`, `c in xz`);
- JSON output for downstream analysis;
- an interactive calculator.

## Scientific boundary

For a physical orientation relationship,

\[
x_A = R_{A\leftarrow M}x_M.
\]

This is ordinary two-phase orientation crystallography.

It is **not** the crystallographic correspondence law

\[
u_M=C_{M\leftarrow A}u_A.
\]

The two-phase workbench therefore uses `OrientationState` and never substitutes
the correspondence matrix for the OR.

Cayron CT, Ball-James, PTMC, polar decomposition, literature ORs, and EBSD
measurements can all supply an `OrientationState` to the same workbench.

## Coordinate transformations

For direct crystal coordinates,

\[
u_A
=
B_A^{-1}
R_{A\leftarrow M}
B_Mu_M.
\]

For reciprocal plane coefficients,

\[
p_A
=
B_A^T
R_{A\leftarrow M}
B_M^{-T}p_M.
\]

The software derives and audits the forward and inverse matrices numerically.

## Easy one-shot usage

### Summary

```bash
cualni-two summary \
  --or polar:do3_to_6m_reference
```

### Cross-phase angle + mapping + misfit

```bash
cualni-two pair \
  --or polar:do3_to_6m_reference \
  "[1 1 0]" "(1 0 1)"
```

### Map one object between crystals

```bash
cualni-two map \
  --or polar:do3_to_6m_reference \
  --from moving \
  "[1 0 1]"
```

### Show coordinate-transformation matrices

```bash
cualni-two matrices \
  --or polar:do3_to_6m_reference
```

### Angle over every crystallographically distinct OR variant

```bash
cualni-two variants \
  --or polar:do3_to_6m_reference \
  "[1 0 0]" "[1 0 0]"
```

### Symmetry-equivalent cross-phase angle table

```bash
cualni-two equivalents \
  --or polar:do3_to_6m_reference \
  "[1 2 3]" "(1 0 1)" \
  --limit 30
```

## General OR inputs

The same commands accept:

```text
--or identity
--or "matrix:1 0 0; 0 1 0; 0 0 1"
--or "euler:10 20 30"
--or "quat:1 0 0 0"
--or "axis:0 0 1; 5"
--or "axis-crystal:[1 1 0]; 5"
--or "parallel:(0 1 0)|(0 1 0)|[1 0 0]|[1 0 0]"
--or stored:MY_ORIENTATION_ID
```

For a non-polar OR that should be associated with a particular transformation
hypothesis, the binding is explicit:

```bash
--bind do3_to_6m_reference
```

The workbench itself does not require this binding because ordinary two-phase
geometry needs only the OR. Theory/correspondence cross-locking does require it.

## Interactive mode

```bash
cualni-two shell
```

Example session:

```text
two-phase> summary
two-phase> pair [1 1 0] ; (1 0 1)
two-phase> map moving [1 0 1]
two-phase> variants [1 0 0] ; [1 0 0]
two-phase> equivalents [1 2 3] ; (1 0 1)
two-phase> matrices
two-phase> max 8
two-phase> derive on
```

To switch OR:

```text
two-phase> or identity
two-phase> or euler:10 20 30
two-phase> or axis-crystal:[1 1 0]; 5
```

## Design rule

The normal workflow is deliberately PTCLab-like:

```text
choose crystal pair
→ choose OR
→ type one or two crystallographic objects
→ one obvious calculation
→ immediate table
```

Advanced mathematics remains available, but it is not required for routine
two-crystal calculations.
