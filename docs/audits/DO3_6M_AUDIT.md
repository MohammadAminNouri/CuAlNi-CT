# DO3 -> 6M audit — corrected status

## Status

This audit supersedes the earlier wording that described the stored 3x3 matrix
as simply "the exact/source-derived correspondence".

The repository now treats it correctly as:

> one selected, symmetry-equivalent **reference correspondence variant** in a
> declared DO3 parent basis and 6M daughter basis.

The full derivation and verification are in
`docs/audits/DO3_6M_TRUTH_LOCK.md`.

## What is source-supported

From James & Hane's review of the Otsuka/Hane 6M revision:

- 6M is a monoclinic cell description of the long-period martensite;
- changing from the old 18R/M18R cell changes the correspondence;
- the long-period geometry is tied to parent {011}-type close-packed geometry;
- the 6M branch produces twelve cube-edge stretch variants;
- the relevant stretch ratios are
  `sqrt(2)a/a0`, `b/a0`, and `sqrt(2)c/(3a0)`;
- the non-right monoclinic angle is between martensite `a` and `c`.

Cayron's convention used by this package is
`u_M = C_M_from_A @ u_A`, with planes mapped by inverse transpose.

## What is a source-constrained interpretation

The explicit repository reference basis

    a_6M = 1/2 [0  1  1]_A
    b_6M =     [1  0  0]_A
    c_6M = 3/2 [0  1 -1]_A

and its inverse correspondence matrix are a reconstruction consistent with the
published geometry and stretch ratios. The matrix is not represented as a
verbatim quotation from James & Hane.

## Independent checks

The repository requires:

1. exact inversion of the declared basis matrix;
2. exact symbolic pulled-back metric;
3. exact 48-operation cubic group and 4-operation monoclinic group;
4. exact computation of `H_C`, cosets, and double cosets;
5. Burnside agreement with explicit double-coset enumeration;
6. independent one-to-one matching of the 12 correspondence-derived stretches
   against the 12 James-Hane Eq. (10) matrices.

## Computation-derived results for the declared reference

For this convention the exact finite-group machinery gives:

- `|G_A| = 48`;
- `|G_M| = 4`;
- `|H_C| = 4`;
- `N_C = 12`;
- `N_operators = 8`;
- double-coset sizes `4,4,4,4,8,8,8,8`.

These are consequences of the declared correspondence and symmetry groups; they
are not entered as physical observations.

## Not yet validated experimentally

This truth-lock pass does not claim experimental validation of variant
frequencies, operator frequencies, Type-I/Type-II/weak junction planes, habit
planes, supercompatibility, PTMC-vs-CT predictive performance, or EBSD
assignments.
