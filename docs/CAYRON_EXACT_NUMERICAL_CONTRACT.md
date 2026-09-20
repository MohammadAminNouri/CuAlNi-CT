# Cayron CT exact/numerical contract

The Cayron implementation now enforces a hard boundary between exact
crystallography and floating metric calculations.

Exact rational checks include correspondence, reciprocal duality,
primitive/conventional duality, axial invariance `C u = u`, projective
`p2 || C^{-T} p1`, equal-volume `|det C| = 1`, order-two parent symmetry,
Type-I rational K1, Type-II rational eta2, and exact intercorrespondence
`C_int = C g_A C^{-1}`.

Floating arithmetic is reserved for lattice metrics, reticular orientation
T, distortion F = T^{-1} C, generalized shear/strain, CMC/SMC eigensystems,
closing-gap rotations, and angular or metric residuals.

Protected Cayron relations:
- axial weak twins, Acta Materialia 236 (2022) 118128: Eqs. (1), (3)-(7);
- transformation twins: Type-I Eqs. (17)-(20), Type-II Eqs. (21)-(24);
- direct/reciprocal transforms are never conflated.

Regression case:
strict randomized campaign seed 5251794611090490145, case 270. The previous
implementation falsely rejected an exact C^{-T} plane relation because a
floating solve accumulated ~3.8e-10 projective error for an ill-conditioned
but exact rational correspondence.
