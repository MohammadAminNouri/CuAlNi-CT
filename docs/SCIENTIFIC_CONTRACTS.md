# Scientific consistency contracts

## Purpose

Before building a large interactive application, CuAlNi-CT needs a layer that
continuously proves that its independently implemented mathematical
representations remain mutually consistent.

This layer does **not** claim that Cayron CT, PTMC, Ball-James, or cofactor
theory are universally equivalent theories.

It checks only identities that must hold when the same crystallographic state
is represented by the corresponding mathematical objects.

Passing these checks is therefore a **software/theory-implementation
consistency statement**, not experimental validation.

---

## Numerical policy

`NumericalPolicy` centralizes computational tolerances.

The defaults are numerical floating-point policies, not measurement
uncertainties and not materials-science acceptance criteria.

For example:

- `algebraic = 1e-10`
- `representation = 1e-9`
- `exact_eigenvalue = 1e-8`

A measured beta angle is never called physically exact merely because a
computed residual happens to be less than one of these numbers.

Experimental uncertainty belongs in provenance.

---

## Contract 1 — dimensional and normalized CMC

Cayron's dimensional correspondence-metric change is

\[
CMC = C^T M_M C - M_A.
\]

Define

\[
W = M_A^{-1/2}.
\]

Then the normalized project quantity must satisfy

\[
\boxed{
D = W\,CMC\,W
}
\]

and independently

\[
D =
M_A^{-1/2} C^T M_M C M_A^{-1/2}-I.
\]

The code evaluates both routes and reports their relative residual.

---

## Contract 2 — CMC and transformation stretch

The parent-whitened pulled-back daughter metric is

\[
\widehat G =
M_A^{-1/2} C^T M_M C M_A^{-1/2}.
\]

The stretch implementation defines

\[
U^2=\widehat G.
\]

Therefore

\[
\boxed{
D = U^2-I.
}
\]

This relation is checked directly.

---

## Contract 3 — eigenvalue bridge

If

\[
\lambda_1\le\lambda_2\le\lambda_3
\]

are the principal stretches, then because \(D=U^2-I\),

\[
\boxed{
q_i=\lambda_i^2-1
}
\]

after consistent ordering.

This is the exact bridge between normalized CMC eigenvalues and the
Ball-James stretch spectrum.  It is stronger and clearer than merely observing
that both methods produce a small middle residual.

---

## Contract 4 — exact single-variant A/M classification

For a positive-definite three-dimensional stretch tensor, the Ball-James
single-variant austenite/martensite compatibility criterion is

\[
\lambda_2=1.
\]

The CMC eigenvalues are \(q_i=\lambda_i^2-1\).  Consequently the CMC
degeneracy/signature criterion and the \(\lambda_2=1\) criterion must give the
same exact numerical classification when evaluated with the same tolerance.

The audit checks this Boolean agreement.

This does **not** assert that all of Cayron CT is identical to all of
Ball-James/cofactor theory.  It checks this specific common compatibility
statement.

---

## Contract 5 — SMC duality

Cayron's SMC is

\[
SMC =
M_A^{-1}
-
C^{-1}M_M^{-1}C^{-T}.
\]

With

\[
S=M_A^{1/2},
\qquad
\widehat G=
M_A^{-1/2} C^T M_M C M_A^{-1/2},
\]

the exact identity is

\[
\boxed{
S\,SMC\,S
=
I-\widehat G^{-1}
=
I-U^{-2}.
}
\]

The code evaluates the two sides independently.

---

## Contract 6 — Metric/Cartesian representation independence

For each supported parent and product Cartesian convention,

\[
F=B_M C B_A^{-1}
\]

must represent the same physical transformation.

The audit runs all currently supported \(3\times3=9\) convention pairs and
requires the Representation Bridge parity residual to remain below the
explicit representation tolerance.

---

## What the audit deliberately does not claim

A passing audit does not prove:

- that a chosen correspondence is the physically correct correspondence;
- that a literature lattice parameter is accurate;
- that CT predicts experiment;
- that PTMC and CT are generally equivalent;
- that a hypothetical exact-compatible projection is a measured material;
- that a numerical tolerance represents experimental uncertainty.

Those are separate scientific questions.

The purpose of this layer is narrower and foundational:

> if two implemented quantities are mathematically required to be the same,
> the program must continuously prove that they are the same.

This will later become the application's **Reliability / Theory Contracts**
panel.
