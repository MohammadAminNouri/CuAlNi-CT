# Universal metric-native orientation kernel

## Purpose

The orientation layer already contains project-aware OR services, Cayron
closing-gap adapters, orientation topology, weak-twin orientation, EBSD
boundaries and theory-specific adapters.  This kernel adds a smaller,
project-independent mathematical core that can be used to audit all of them.

The supported domain is deliberately precise:

> any finite, physically admissible three-dimensional crystallographic phase
> pair with positive-definite metrics and exact finite symmetry groups.

"Any possible input" does **not** mean silently accepting invalid matrices,
incompatible point groups, singular correspondences, left/right handedness
ambiguities, or underdetermined parallelisms.  Those cases fail with explicit
diagnostics.

## C, F, U and R are different objects

The package correspondence convention is

\[
u_M=C_{M\leftarrow A}u_A.
\]

Using symmetric physical embeddings

\[
B_A=M_A^{1/2},\qquad B_M=M_M^{1/2},
\]

the correspondence-induced Cartesian deformation bridge is

\[
F_{M\leftarrow A}=B_M C_{M\leftarrow A}B_A^{-1}.
\]

Its right polar decomposition is

\[
F_{M\leftarrow A}=R_{M\leftarrow A}U_A,
\]

with

\[
R_{M\leftarrow A}\in SO(3),\qquad U_A=U_A^T>0.
\]

The OR convention used by the orientation workbench is the inverse physical
rotation,

\[
R_{A\leftarrow M}=R_{M\leftarrow A}^T.
\]

The kernel stores and audits **all four objects separately**.  It also
cross-checks `U_A` against the metric-native generalized eigenproblem

\[
C^T M_M C\,v_i=\mu_i M_A v_i,
\qquad \lambda_i=\sqrt{\mu_i}.
\]

The polar OR remains a comparator.  It is not relabelled as Cayron's natural
OR, a Ball--James OR, PTMC, literature, or experiment.

## Orientation intersection topology

For a supplied physical OR

\[
x_A=R_{A\leftarrow M}x_M,
\]

the full orientation intersection subgroup is

\[
H_T^A=G_A\cap R_{A\leftarrow M}G_MR_{A\leftarrow M}^{-1}.
\]

Membership is discovered in the physical Cartesian metric.  After membership
has been identified, subgroup closure, left cosets, double cosets, adjacency
and Burnside counting are performed with the **exact SymPy crystallographic
matrices**.

This avoids using floating matrices to do finite-group algebra.

The kernel also computes the proper subgroup topology separately, because full
crystallographic groups and physical SO(3) orientation variants have different
jobs.

If a correspondence is present, `H_T` and `H_C` are compared as computed
results.  Equality is never assumed.

## Symmetry-reduced disorientation

Equivalent OR representatives are

\[
R'=S_A R S_M^{-1}.
\]

For two ORs, independent symmetry choices reduce to the exact minimization

\[
\Delta\theta
=
\min_{S_A\in G_A^+,\;S_M\in G_M^+}
\operatorname{angle}
\left(S_A R_1 S_M R_2^T\right).
\]

Only proper point-group operations enter this SO(3) minimization.

Rotation angles use an `atan2(sin(theta), cos(theta))` formulation rather than
an `arccos(trace)`-only formula, preserving sensitivity near zero and stability
near 180 degrees.

## General parallelism solver

Two independent crystallographic parallelisms may use any combination of:

- direct direction `[uvw]`;
- reciprocal plane normal `(hkl)`.

Direct vectors are mapped by `B u`; planes by `B^-T p`.  The solver exposes
all projective sign branches and rejects:

- incompatible internal angles;
- collinear/underdetermined pairs;
- non-SO(3) results.

No branch is selected silently.

## Cayron exact closing-gap OR

A `CTTwin` is accepted only after provenance checks:

\[
C_{\rm int}=C g_A C^{-1},
\]

\[
p_M\parallel C^{-T}p_A,
\qquad
u_M\parallel C u_A.
\]

Only then are the plane/direction parallelisms converted to proper physical
OR candidates.  If a natural OR is supplied independently, branch ranking uses
the symmetry-reduced disorientation.  Without a natural OR, every branch
remains unranked.

## Exact crystal-basis covariance

For exact unimodular basis matrices

\[
P_A,P_M\in GL(3,\mathbb Z),
\]

the crystal-coordinate state transforms as

\[
M'=P^T M P,\qquad
g'=P^{-1}gP,\qquad
C'=P_M^{-1}CP_A.
\]

The symmetric Cartesian embedding changes gauge.  The kernel therefore
computes the corresponding orthogonal gauge matrices `Q_A`, `Q_M` and audits

\[
F'=Q_MFQ_A^T,
\]

\[
U'=Q_AUQ_A^T,
\]

\[
R'_{A\leftarrow M}=Q_AR_{A\leftarrow M}Q_M^T.
\]

Right-handed (`det P=+1`) rebasing preserves SO(3) directly.  Left-handed
basis changes are still represented mathematically, but if the parent and
moving basis parities differ the canonical-frame OR becomes parity reversing.
The API refuses to disguise that O(3) matrix as an SO(3) orientation.

## Generality tests

The hardening suite covers:

- Chen 2000 Cu--Al--Ni 2H;
- Cayron 2022 NiTi B2 -> B19';
- James--Hane Cu--Al--Ni 6M;
- exact random SL(3,Z) parent/product basis changes;
- polar C/F/U/R covariance;
- closing-gap OR covariance;
- exact H_T subgroup/coset/operator covariance;
- cross-lock with the existing `OrientationService`;
- cross-lock with the existing `CayronOrientationAdapter`;
- the published NiTi natural-OR plane/direction parallelisms;
- every one of the 32 crystallographic point groups using a
  group-invariant positive-definite metric;
- invalid metrics, incompatible point groups, improper ORs,
  underdetermined parallelisms and non-unimodular bases.

## Scope boundary

This kernel does not manufacture an atomistic/natural OR from `C + metrics`.
That would be a theory claim not justified by generic correspondence data.

It also does not identify the standard Cartesian Hamilton quaternion with
Cayron's crystallographic metric-quaternion formalism.  A metric-quaternion
implementation requires its own derivation and independent validation rather
than a name substitution.
## Quotient-disorientation correction discovered by hardening

The orientation-kernel cross-lock exposed a real issue in the previous
`OrientationService.compare_orientations()` implementation.

The old comparison enumerated parent-derived orientation-variant
representatives and minimized the raw matrix angle between representatives,

\[
\operatorname{angle}(A B^T).
\]

That is not yet a complete crystallographic quotient distance because a
moving/product crystal orientation is unchanged by replacing its representative
with

\[
R \longrightarrow R S_M^{-1},
\qquad S_M\in G_M^+.
\]

Therefore the correct symmetry-reduced OR distance is obtained by the extra
proper moving-phase reduction

\[
\boxed{
\Delta\theta
=
\min_{A,B,S_M\in G_M^+}
\operatorname{angle}(A S_M B^T)
}.
\]

Equivalently, before coset compression this can be written as a minimization
over one relative parent symmetry and one relative moving symmetry,

\[
\Delta\theta
=
\min_{S_A\in G_A^+,\;S_M\in G_M^+}
\operatorname{angle}
\left(S_A R_1 S_M R_2^T\right).
\]

The hardening test found a concrete case where the incomplete representative
comparison returned approximately `41.401158 deg`, while the full quotient
minimization returned approximately `39.397354 deg`.

The code was corrected at the service level. The expected test value was
**not** changed to force a pass. The new independent metric-native kernel and
the project-aware orientation service now cross-lock on the same quotient
definition.

The comparison angle is evaluated with

\[
\theta=\operatorname{atan2}(\sin\theta,\cos\theta),
\]

using the skew and trace invariants of the relative rotation. This avoids the
small-angle conditioning problem of an `acos(trace)`-only evaluation.
