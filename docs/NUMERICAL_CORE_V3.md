# Numerical Core V3 — equation-preserving adaptive hardening

## Non-negotiable rule

No scientific equation is changed to make a test pass.  No exact-compatibility
tolerance is widened to hide floating-point error.  V3 changes the numerical
route only where an algebraically identical route exists, and escalates
arithmetic precision when binary64 cannot safely resolve the existing decision.

## Source equations retained

With the package convention `u_M = C_{M<-A} u_A`:

- **CMC**

  `A = C.T @ M_M @ C - M_A`

- **normalized pulled metric / normalized CMC**

  `Ghat = M_A^(-1/2) C.T M_M C M_A^(-1/2)`

  `D = Ghat - I`

- **SMC**

  `SMC = M_A^(-1) - C^(-1) M_M^(-1) C^(-T)`

- **metric-native stretch pencil**

  `(C.T M_M C) v_i = mu_i M_A v_i`, `lambda_i = sqrt(mu_i)`

The exact identity

`(C.T M_M C)^(-1) = C^(-1) M_M^(-1) C^(-T)`

is used only to evaluate SMC with solves/high precision rather than an explicit
chain of inverses.

## Exact relation between normalized CMC and the generalized pencil

Let `S = M_A^(1/2)` and `G = C.T M_M C`.  If

`G v_i = mu_i M_A v_i`

with `V.T M_A V = I`, then `q_i = S v_i` are orthonormal and

`(S^(-1) G S^(-1) - I) q_i = (mu_i - 1) q_i`.

Therefore the normalized CMC eigenvalues are exactly

`eta_i = mu_i - 1`.

V3 classifies CMC through this equivalent generalized pencil, using the same
fixed `exact_eigenvalue` tolerance as before.

## Habit-plane factorization

For first-order CMC degeneracy, choose the positive and negative nonzero
`eta` values.  In the generalized eigenbasis the CMC cone is

`eta_+ x_+^2 + eta_- x_-^2 = 0`.

Because `x = V.T M_A u`, its two linear factors have crystal-coordinate plane
covectors

`p_A = M_A (sqrt(eta_+) v_+ +/- sqrt(-eta_-) v_-)`.

This is algebraically identical to the old whitened construction

`p_A = M_A^(1/2) (sqrt(eta_+) q_+ +/- sqrt(-eta_-) q_-)`.

For second-order degeneracy, the unique finite plane is proportional to
`M_A v_k`, where `k` is the sole nonzero `eta`.  Third-order degeneracy has no
unique finite set of planes.

## Numerical defects identified before V3

1. Binary64 whitening/classification could cross the fixed `1e-8` decision
   boundary under hostile but valid crystallographic rebasing.
2. A tiny generalized-eigen backward residual did not guarantee forward
   eigenvalue accuracy for badly conditioned pencils.
3. The first Cholesky-factorized binary64 repair still misclassified certified
   hostile cases; it was rejected.
4. The old high-precision validator used `Rational(str(float(x)))`, which is not
   the exact IEEE-754 number passed to production.
5. The metamorphic validator mixed solver error with finite-precision
   re-encoding error.  Exact-binary root isolation showed that some hostile
   rebases genuinely move the encoded pencil across the fixed decision
   boundary.
6. The habit-plane helper compared a product of two eigenvalues with an
   eigenvalue tolerance (`qi*qj > tol`), which is dimensionally/logically
   inconsistent.  The relevant requirement is strict opposite sign after the
   individual zero tests.
7. General `sqrtm` plus explicit inversion was used for SPD metric square roots.
8. Several inverse products were formed explicitly when exact solve identities
   exist.
9. PTMC silently accepted middle-singular-value residuals up to `1e-6` even
   when its caller supplied `tol=1e-9`; V3 removes that hidden widening.
10. PTMC also used fixed absolute polynomial/duplicate thresholds.  V3 moves
    polynomial evaluation to arbitrary precision and uses the caller tolerance
    for root acceptance.
11. The James-Hane exact-compatibility angle helper allowed a fixed `1e-12`
    out-of-domain clip.  V3 narrows this to a machine-roundoff guard only.

## Precision policy

Production first evaluates the same pencil through two algebraically equivalent
binary64 routes.  Precision is escalated when any of the following holds:

- the closest `mu_i` lies within `1024 * exact_eigenvalue` of `1`;
- a metric/correspondence/pulled-metric condition number reaches `1e8`;
- the two binary64 routes disagree by at least one eighth of the unchanged
  exact-eigenvalue tolerance.

Escalation evaluates the exact IEEE-754 inputs with `mpmath` at 90 decimal
digits.  The independent validator separately converts each input float using
`float.as_integer_ratio()` and isolates the exact rational roots of

`det(C.T M_M C - mu M_A) = 0`.

The production implementation does **not** call the certified oracle.

## Validation philosophy

A basis change is not automatically required to preserve a discrete result if
the *binary64 encoded input problem itself* changed.  Each encoded
representation is first judged against its own independent exact-binary oracle.
Cross-representation invariance is required only when the oracle itself says
that the encoded quantity remained invariant.

The hardened gate includes:

- exact-binary certified eigenvalue/classification checks;
- arbitrary-precision generalized eigenvectors;
- metric-orthonormality and pencil residual diagnostics;
- habit-plane factorization and projective-family checks;
- CMC/SMC independent high-precision oracles;
- SPD square-root identities;
- Ball-James inverse-vs-solve branch preservation;
- strict PTMC root validation;
- the published cofactor-condition minus-determinant sign regression;
- mutation testing;
- all frozen Cayron/source regressions;
- whole-repository pytest.

A production commit is made by the V3 workflow only if every gate passes.


## V3.1 closure of the remaining red gates

The first gated V3 run deliberately did not commit production source because
three focused checks remained red.  V3.1 changes only those areas.

### 1. Literal CMC remains literal

After metric-input validation, `cmc()` now returns exactly

`C.T @ M_m @ C - M_a`

without a post-hoc `0.5*(A + A.T)` repair.  Metric inputs themselves are
canonicalized as symmetric tensors before the equation is evaluated; this is
the semantic input contract, not a change to CMC.  Any symmetric storage view
needed by an eigensolver is local to that eigensolver.

The validator no longer demands an arbitrary fixed forward residual for this
literal binary64 matrix product.  It uses the standard length-three dot-product
roundoff model

`gamma_3 = 3u / (1 - 3u)`

and propagates that bound through the two matrix multiplications and final
subtraction.  A CMC result fails only if its error against the arbitrary-
precision equation exceeds that derived machine-arithmetic envelope.

### 2. SPD admissibility is decided exactly near the binary64 sign boundary

A metric is first interpreted as the canonical binary64 symmetric tensor

`M_c = 0.5 * (M + M.T)`.

If the smallest binary64 eigenvalue is safely positive, the normal symmetric
eigensolver is used.  If its sign is numerically ambiguous, V3.1 applies
Sylvester's criterion to the exact rational values of the canonical IEEE-754
entries.  For a 3x3 symmetric matrix this means the three leading principal
minors are checked exactly.

No negative eigenvalue is clipped.  No nearest-SPD projection is performed.
If the exact canonical binary64 metric is not SPD, the encoded input has left
the mathematical domain and is rejected explicitly.  If it is exactly SPD
but binary64 lost the sign, the calculation proceeds and high precision is
used.

The same rule is used by the independent conditioning gate, so an explicit
production rejection is accepted only when an exact-rational oracle proves
that the encoded metric pencil itself is invalid.

### 3. SMC uses the defining equation without binary64 inverse cancellation

SMC remains

`M_A^-1 - C^-1 M_M^-1 C^-T`.

Production evaluates the algebraically identical identity

`M_A^-1 - (C.T M_M C)^-1`

at arbitrary precision on the exact canonical binary64 inputs, then rounds
once for the public float64 result.  The independent validator evaluates the
original Eq.-41 route with `C^-1 M_M^-1 C^-T`, also at high precision.  Thus
the production and oracle paths are algebraically equivalent but
implementation-independent.

### 4. Metamorphic checks use a causal error decomposition

For a linear representation transform `T`, write

`P_0 = O_0 + E_0`, `P_1 = O_1 + E_1`,

where `P` is production and `O` is the independent oracle.  Then

`P_1 - T(P_0) = [O_1 - T(O_0)] + E_1 - T(E_0)`.

Therefore the production cross-representation defect must satisfy the triangle
bound

`||P_1 - T(P_0)|| <= ||O_1 - T(O_0)|| + ||E_1|| + ||T(E_0)||`

plus only the machine-roundoff needed to evaluate the diagnostic transform.

The first term is **encoded-input drift**.  The other two are **local solver
errors**.  V3.1 records them separately for CMC, SMC, common length scaling,
and generalized-spectrum comparisons.  It no longer uses a single loose
metamorphic number that can either blame the solver for representation drift
or hide a solver failure.

Discrete compatibility is still judged against the unchanged
`exact_eigenvalue = 1e-8` policy for each encoded problem independently.
