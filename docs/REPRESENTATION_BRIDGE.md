# Representation Bridge

## Purpose

CuAlNi-CT keeps the crystallographic metric as the scientific source of truth,
while also exposing a Cartesian mode comparable to the workflow used by
PTCLab.

There are **not two independent physics engines**.

There is one physical crystallographic state with multiple rigorously equivalent
representations.

## Why this is needed

For a lattice with direct metric

\[
M =
\begin{bmatrix}
a^2 & ab\cos\gamma & ac\cos\beta\\
ab\cos\gamma & b^2 & bc\cos\alpha\\
ac\cos\beta & bc\cos\alpha & c^2
\end{bmatrix},
\]

a Cartesian basis matrix \(B\) satisfies

\[
B^T B = M.
\]

A direct crystallographic vector \(u\) becomes

\[
x = Bu,
\]

whereas a reciprocal plane covector \(p=(hkl)\) becomes the physical Cartesian
normal

\[
g = B^{-T}p.
\]

Consequently,

\[
\|Bu\|^2 = u^T M u,
\]

\[
\|B^{-T}p\|^2 = p^T M^{-1}p,
\]

and the incidence pairing is representation-independent:

\[
(B^{-T}p)^T(Bu)=p^Tu.
\]

These equalities are unit-tested.

## Cartesian conventions

### `LEGACY_A_X_B_XY`

This is the Cartesian structure matrix already used historically by
`Lattice.structure_matrix()`:

- \(a \parallel x\);
- \(b\) lies in the \(xy\) plane.

It is retained for backward compatibility.

### `PTCLAB_A_X_C_XZ`

This reproduces the coordinate convention stated in PTCLab User Manual,
section 2.2:

- \(x\parallel e_1\) / \(a\);
- \(e_3\) / \(c\) lies in the \(xz\) plane.

The matrix is built from the same six lattice parameters and is tested to
satisfy \(B^TB=M\).

This matters because PTCLab explicitly notes that a Cartesian convention does
not change the physical crystallographic result but *does* change quantities
such as Euler angles and orientation-matrix entries.

### `SYMMETRIC_METRIC`

This uses the symmetric positive square root

\[
B=M^{1/2}.
\]

It is especially natural for the Cayron/metric and Ball-James bridge because it
introduces no arbitrary crystallographic axis preference.

## Transformation correspondence

The package convention remains

\[
u_M = C_{M\leftarrow A}u_A.
\]

With explicit Cartesian embeddings \(B_A\) and \(B_M\), the same physical
transformation is

\[
\boxed{
F = B_M C_{M\leftarrow A} B_A^{-1}
}
\]

and therefore

\[
\boxed{
F^TF =
B_A^{-T}
(C^T M_M C)
B_A^{-1}.
}
\]

This is the central representation-parity identity.

For planes,

\[
p_M=C^{-T}p_A
\]

is exactly equivalent to

\[
g_M=F^{-T}g_A.
\]

Both routes are calculated independently and compared numerically.

## Principal stretches

Changing Cartesian convention changes matrix coordinates but only by proper
frame rotations. Therefore the singular values of \(F\),

\[
\lambda_1,\lambda_2,\lambda_3,
\]

are invariant.

The implementation evaluates all \(3\times3=9\) parent/product convention
pairs and requires the same principal stretches. These are also compared
against the metric-whitened stretch tensor already used by
`stretch_from_metrics`.

## Why this is stronger than PTCLab's original architecture

PTCLab primarily converts crystal quantities into one chosen orthogonal frame
for calculation.

CuAlNi-CT instead keeps:

1. the metric-native crystallographic representation;
2. an explicit Cartesian representation;
3. numerical parity identities connecting them.

The future GUI will therefore offer:

- **Metric / Crystallographic**
- **Cartesian / PTCLab-compatible**
- **Side-by-side**

without duplicating or silently changing the physics.

## Future GUI implication

Every displayed object should be able to expose simultaneously:

- crystallographic coordinates;
- direct or reciprocal object type;
- selected Cartesian convention;
- Cartesian coordinates;
- source/provenance;
- units;
- parity residual.

This bridge is the foundation for the future project tree, crystallography
console, transformation workspace, stereographic views, EBSD adapters, and
theory-comparison interface.
