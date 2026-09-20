# CuAlNi-CT scientific hardening patch

Base checkpoint:

```text
branch: fortify/do3-6m-truth-lock
commit: 77b65d0af96667b7c940b4a79fe596f3a36798aa
```

This patch deliberately avoids a large architecture rewrite.  It strengthens
the mathematical core and adds opt-in reference data while leaving the
ProjectState / calculation-service / future PTCLab-like direction intact.

## Corrected errors

1. `docs/THEORY.md` had the wrong sign in CC3.  It is now locked as

   `tr(U^2) - det(U^2) - |a|^2 |n|^2/4 - 2 >= 0`.

2. SMC is documented and implemented strictly as **shear by metric
   correspondence**, not a reciprocal CMC and not `CMC^-1`.

3. The CMC first-order test no longer accepts two same-sign small eigenvalues
   merely because their product falls below a loose product tolerance.
   With exactly one zero eigenvalue, the two remaining eigenvalues must have
   opposite signs.

4. PTMC documentation now keeps the relative martensite rotation in the twin
   equation before forming the common-frame laminate.

5. Stale documentation saying the DO3->6M correspondence still had to be
   derived has been removed.  The branch is already truth-locked; the new text
   distinguishes that verified internal reference from universal Cu-Al-Ni
   validity.

## Strengthened mathematics

- added the direct crystallographic generalized eigenproblem

  `C^T M_M C v = mu M_A v`, `lambda=sqrt(mu)`;

- cross-checks generalized stretches against the metric-whitened `U`;

- CMC analysis reports inertia `(negative, zero, positive)` and exact
  first/second/third-order degeneracy;

- SPD/finite/invertibility input validation added to CT metric routines;

- cubic `m-3m` has an explicit 48-operation inventory and regression lock.

## Why 48 matters

The full cubic crystallographic group is:

```text
48 total
24 proper
24 improper
9 mirrors
9 proper two-fold rotations
```

CT correspondence variants/operators use all 48.  Physical OR/EBSD rotations
use the 24 proper operations.  These are deliberately separate calculations.

## Cu-Al-Ni scope expansion

A new opt-in literature-state registry prevents the project from becoming a
James-Hane preset.  It includes independent examples/observations from:

- James-Hane 2000 reduced 6M benchmark;
- Landa et al. 2007 2H single crystal;
- Chen et al. 2000 2H EBSD OR/twin observations;
- Otsuka-Nakamura-Shimizu 1974 stress-induced 18R;
- Ibarra et al. 2006 conventional C2/m beta3' and Pmmn gamma3' cells.

Missing parameters remain missing.  The 1974 18R entry, for example, refuses
to build a metric because the accessible source statement does not supply the
needed monoclinic angle.

## Added regression tests

The patch contains 14 focused tests covering:

- exact 48/24 cubic symmetry inventory;
- full-group metric preservation;
- generalized metric-native stretch eigenproblem on a non-cubic parent;
- CMC degeneracy orders 1/2/3 and an invalid same-sign case;
- source-specific Cu-Al-Ni reference states;
- fail-loudly behavior for incomplete literature metrics;
- CC2/CC3 invariance under reciprocal rescaling of the rank-one factors.

The focused patch tests passed locally:

```text
14 passed
```

Full repository CI should still be run after applying the patch.
