# Scientific consistency contracts

## Purpose

CuAlNi-CT continuously checks identities that must hold when the same
crystallographic state is represented through different mathematical forms.

These contracts do **not** claim universal equivalence of Cayron CT,
Ball-James, PTMC or cofactor theory, and they do not constitute experimental
validation.

## Numerical policy

Numerical tolerances are floating-point policies, not measurement
uncertainties.  Experimental uncertainty belongs in provenance.

## Contract 1 — dimensional and normalized CMC

\[
CMC=C^TM_MC-M_A.
\]

With \(W=M_A^{-1/2}\),

\[
\boxed{D=W\,CMC\,W}
\]

and independently

\[
D=M_A^{-1/2}C^TM_MCM_A^{-1/2}-I.
\]

## Contract 2 — CMC and transformation stretch

\[
\widehat G=M_A^{-1/2}C^TM_MCM_A^{-1/2},
\qquad
U^2=\widehat G.
\]

Therefore

\[
\boxed{D=U^2-I}.
\]

## Contract 3 — eigenvalue bridge

If \(\lambda_1\le\lambda_2\le\lambda_3\),

\[
\boxed{q_i=\lambda_i^2-1}.
\]

## Contract 3b — metric-native generalized eigenproblem

The same principal stretches must be obtained directly in crystallographic
coordinates from

\[
\boxed{C^TM_MC\,v_i=\mu_iM_Av_i},
\qquad
\lambda_i=\sqrt{\mu_i}.
\]

The eigenvectors must satisfy

\[
V^TM_AV=I.
\]

This extra route is intentionally independent of the whitening calculation
and is a strong regression check for arbitrary non-cubic parent metrics.

## Contract 4 — exact single-variant A/M classification

For a positive-definite three-dimensional stretch, Ball-James
single-variant compatibility requires \(\lambda_2=1\).

Because \(q_i=\lambda_i^2-1\), the exact CMC degeneracy/signature test and the
\(\lambda_2=1\) test must classify the same sourced state consistently.

This is a shared mathematical statement, not a claim that full CT and
Ball-James are identical theories.

## Contract 5 — SMC duality

Cayron's **shear by metric correspondence** is

\[
SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}.
\]

It is not \(CMC^{-1}\).

With \(S=M_A^{1/2}\),

\[
\boxed{
S\,SMC\,S=I-\widehat G^{-1}=I-U^{-2}
}.
\]

## Contract 6 — representation independence

Every supported Cartesian realization must describe the same physical map.
Changing crystallographic-to-Cartesian convention may change matrix
coordinates, but not the underlying geometry.

## Contract 7 — cubic full/proper symmetry split

For conventional cubic `m-3m`:

\[
\boxed{|G|=48,\qquad |G^+|=24}.
\]

The full group must contain exactly nine reflections and nine proper
two-fold rotations.  CT topology uses the full group; rotation-valued OR/EBSD
operations use the proper subgroup.

## Contract 8 — cofactor sign and scaling

The code locks the Chen et al. CC3 condition as

\[
\boxed{
\operatorname{tr}(U^2)-\det(U^2)
-\frac{|a|^2|n|^2}{4}-2\ge0
}.
\]

The determinant sign is **minus**.

Because \(a\otimes n\) is unchanged by
\(a\mapsto\kappa a,\ n\mapsto n/\kappa\), CC2 and CC3 must be invariant under
that reciprocal rescaling.

## What passing the contracts does not prove

Passing does not prove:

- that a chosen correspondence is physically correct;
- that a literature cell is the correct representation for another specimen;
- that CT predicts experiment;
- that CT and PTMC are universally equivalent;
- that a nearest-zero CMC projection is an actual compatible material;
- that numerical tolerance equals experimental uncertainty.

The contracts prove only that mathematically linked software representations
are internally coherent.
