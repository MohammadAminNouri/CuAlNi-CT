# DO3 -> 6M truth lock

## Purpose

This document freezes the crystallographic meaning of the repository's
DO3 -> 6M reference branch before any further work on transformation twins,
PTMC, cofactor conditions, EBSD, or composition sweeps.

The central rule is:

> A single 3x3 correspondence matrix is not "the whole transformation".
> It is one reference correspondence variant written in one declared pair of
> crystallographic bases. Symmetry generates the rest of the family.

## 1. Primary-source facts retained

James & Hane (Acta Materialia 48, 2000, 197-222) summarize the revision of the
old 18R/M18R description to a 6M cell. Their discussion makes the following
points relevant here:

- the 6M cell is consistent with the accepted monoclinic symmetry;
- changing the cell changes the correspondence;
- the DO3/B2 long-period structure is built from parent close-packed
  {011}-type geometry;
- the resulting cubic-to-monoclinic transformation has twelve cube-edge
  stretch variants (their Eq. 10);
- for the 6M cell the dimensionless stretches used in the cube-edge family are
  `sqrt(2)a/a0`, `b/a0`, and `sqrt(2)c/(3a0)`;
- the non-right monoclinic angle is the angle between the martensite a and c
  edges.

The source does not print the Python matrix used by this repository verbatim.
Accordingly the stored matrix is classified as a **source-constrained
reconstruction**, not as a literal quotation.

Cayron's Correspondence Theory convention used by the package is

    u_M = C_M_from_A u_A

for direct-space crystallographic coordinates. Reciprocal plane covectors obey

    p_M = C_M_from_A^{-T} p_A.

These two actions are deliberately kept separate.

## 2. Selected reference basis

The reference 6M daughter basis is represented in cubic parent coordinates as

    a_6M = 1/2 [0  1  1]_A
    b_6M =     [1  0  0]_A
    c_6M = 3/2 [0  1 -1]_A

Therefore

    C_A_from_M =
        [[0,   1,    0],
         [1/2, 0,  3/2],
         [1/2, 0, -3/2]]

and, by exact inversion,

    C_M_from_A =
        [[0,   1,    1],
         [1,   0,    0],
         [0, 1/3, -1/3]].

This is the repository's **selected reference correspondence variant**.

The geometry is independently consistent with the James-Hane stretch
normalizations. The parent lengths of the three selected translations are

    a0/sqrt(2),  a0,  3 a0/sqrt(2),

so the daughter/parent length ratios are exactly

    sqrt(2) a/a0,
    b/a0,
    sqrt(2) c/(3 a0).

## 3. Monoclinic angle convention

The package uses a conventional unique-b metric

    M_6M =
        [[a^2, 0, a c cos(beta)],
         [0, b^2, 0],
         [a c cos(beta), 0, c^2]].

Here `beta` is the package's internal symbol for the physical non-right angle
between `a_6M` and `c_6M`. Source notation and package notation are recorded
separately.

## 4. Exact pulled-back metric

For the selected reference variant,

    G_C = C_M_from_A^T M_6M C_M_from_A

is derived symbolically as

    G_C =
        [[b^2, 0, 0],
         [0,
          a^2 + 2 a c cos(beta)/3 + c^2/9,
          a^2 - c^2/9],
         [0,
          a^2 - c^2/9,
          a^2 - 2 a c cos(beta)/3 + c^2/9]].

For a cubic parent `M_A = a0^2 I`,

    U^2 = G_C/a0^2

and

    CMC = G_C - a0^2 I = a0^2 (U^2 - I).

## 5. Independent James-Hane verification

The code contains a second implementation of the twelve James-Hane cube-edge
matrices, written directly from their Eq. (10) formulas. It does not call the
correspondence-derived stretch code.

The truth-lock test therefore performs two independent routes:

1. `C + M_6M -> G_C -> U -> cubic symmetry -> 12 U_i`;
2. `James-Hane Eq. (10) -> 12 U_i`.

The two unordered sets are compared with an optimal one-to-one assignment
(Hungarian algorithm), not with a nearest-neighbor test that could reuse the
same reference matrix twice.

For the rounded James-Hane Cu-14 wt% Al-4 wt% Ni benchmark, the expected
maximum Frobenius mismatch is at floating-point roundoff scale (about 1e-16 in
the independently checked calculation).

## 6. Reference-matrix non-uniqueness

An earlier exploratory matrix was

    C_alt =
        [[1, 0, 1],
         [0, 1, 0],
         [-1/3, 0, 1/3]].

Exact algebra gives

    C_alt = C_ref P

with

    P =
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]],

and `P^T P = I`, `det(P)=+1`.

Thus the displayed matrices differ by a proper cubic parent-axis permutation
in this coordinate statement.

## 7. Exact discrete CT consequences

Using full cubic `m-3m` (48 operations), monoclinic `2/m` (4 operations), and
the selected reference correspondence, the repository derives:

    |H_C| = 4
    N_C = 12 correspondence variants
    N_operators = 8
    double-coset sizes = 4,4,4,4,8,8,8,8.

The operator count is independently checked by Burnside's lemma.

## 8. What this truth lock does NOT claim

This pass does not claim that:

- every mathematically allowed variant occurs experimentally;
- every double-coset operator is a physical twin;
- Type-I/Type-II/weak junction predictions have already been verified for
  Cu-Al-Ni;
- PTMC, CT and Ball-James have already been quantitatively ranked;
- the rounded James-Hane Table 4 lattice constants satisfy exact compatibility;
- the reference benchmark is a default for an unknown specimen.

Those are later validation layers.

## 9. Next scientific gate

Only after this reference layer passes all tests should the repository proceed
to operator-by-operator CT Type-I/Type-II calculations, independent
Ball-James/Mallard rank-one solutions, CMC/SMC compatibility,
cofactor/supercompatibility, PTMC comparison, and finally EBSD/interface
validation.

The purpose of the later work remains to **test** Correspondence Theory, not to
assume it is correct.
