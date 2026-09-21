# Physical-equivalence and independent-theory validation

This layer strengthens the CT validation in two independent directions without
changing the verified CT equations.

## 1. Exact crystallographic basis covariance

A lattice basis is replaced by an exact unimodular matrix:

\[
B'_A=B_A P_A,\qquad B'_M=B_M P_M,\qquad P_A,P_M\in GL(3,\mathbb Z).
\]

The same physical state must then be represented by

\[
M'_A=P_A^T M_A P_A,\qquad
M'_M=P_M^T M_M P_M,
\]

\[
g'_A=P_A^{-1}g_A P_A,\qquad
g'_M=P_M^{-1}g_M P_M,
\]

and, for the package convention \(u_M=C_{M\leftarrow A}u_A\),

\[
C'=P_M^{-1}CP_A.
\]

Direct vectors and reciprocal plane covectors transform differently:

\[
u_{\rm old}=P\,u_{\rm new},\qquad
p_{\rm old}=P^{-T}p_{\rm new}.
\]

The validator therefore solves the transformed problem independently and pulls
every output back before comparison.

It checks substantially more than variant/operator counts:

- exact correspondence subgroup;
- exact left-coset/variant partition;
- exact double-coset/operator partition;
- the complete variant-to-variant operator adjacency after physical relabelling;
- exact parent-symmetry provenance of each CT twin;
- Type-I / Type-II / compound classification;
- parent and product twin planes;
- parent and product shear directions;
- shear magnitude;
- product-space intercorrespondence.

Projective geometry uses a metric-native chord residual. It never evaluates
`sqrt(1-cos(theta)^2)` near zero angle.

## 2. CT vs Mallard vs generic Ball-James

The comparison deliberately keeps three calculations independent.

### Cayron CT

Inputs:

- parent/product symmetry;
- correspondence;
- parent/product metrics.

Outputs:

- Type-I/Type-II twin elements;
- shear;
- compound classification.

### Mallard

The parent metric is whitened to a physical orthonormal frame. For every proper
parent twofold \(Q\),

\[
U_j=Q\,U_0\,Q^T
\]

is formed without using any CT twin output. Mallard's law then gives its
Type-I and Type-II rank-one solutions.

### Generic Ball-James

The generic analytical rank-one solver receives only \(U_0\) and \(U_j\). It
solves

\[
R U_0-U_j=a\otimes n
\]

through the middle-eigenvalue condition and the Ball-James analytical
construction. Mallard output is not used to construct the solutions; it is
used only afterward to identify the two unordered branches.

### Coordinate comparison

Ball-James/Mallard `n` is a reference-configuration interface normal. CT
`plane_a` is therefore converted from a reciprocal parent covector to the same
physical orthonormal normal.

Ball-James/Mallard `a` is a current/deformed shear direction. CT `direction_a`
is stored in parent reference coordinates and is first pushed by \(U_j\) before
the angular comparison.

The twinning shear is compared with the scale-invariant quantity

\[
s=\|a\|\,\|U_j^{-T}n\|.
\]

Compound status is cross-checked independently: a stretch pair is classified
as compound when at least two distinct physical parent twofold axes generate
the same \(U_j\). The CT classifier is compared with this result only after the
independent generator search is complete.

## Current hard cases

The tests apply the complete basis-covariance and three-theory comparison to:

- the permanent Chen 2000 Cu-Al-Ni 2H blind benchmark;
- the Cayron 2022 B2 -> B19' NiTi benchmark.

The generic three-theory validator is also run on the existing James-Hane
Cu-Al-Ni 6M truth-locked state, independently of the older 6M twin-atlas tests.

The basis campaign uses fixed-seed, exact, small-entry GL(3,Z) transformations
in both parent and product phases. This is intended to detect coordinate
convention errors, not to turn the test into an arbitrary-precision
conditioning benchmark.
