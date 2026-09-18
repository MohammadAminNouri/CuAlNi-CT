# Cayron-parallel scientific audit for CuAlNi-CT

This document records which steps in the CuAlNi-CT architecture are deliberately
parallel to Cyril Cayron's correspondence-theory workflow, which are already
implemented, and which are still separate future theory adapters.

## 1. Non-negotiable distinction: F, T/R and C are different objects

Cayron distinguishes distortion, orientation and correspondence matrices.  In
our code the same separation is enforced:

- `C`: crystallographic correspondence, acting on direct lattice indices;
- `R`: a proper physical Cartesian orientation relationship;
- `F`: deformation/stretch path quantities handled by the transformation and
  compatibility layers.

The current `polar` OR command produces a **polar-rotation candidate** derived
from correspondence + metrics.  It is intentionally not promoted to Cayron's
natural orientation matrix `T`, a Ball-James rotation, a PTMC OR, or an
experimental EBSD OR.

## 2. Metric-native crystallography remains the scientific source of truth

Cayron's direct/reciprocal distinction and metric tensor are respected:

- direct-vector norm: `sqrt(u^T M u)`;
- reciprocal-plane norm: `sqrt(p^T M^-1 p)`;
- physical plane normal is metric-derived, never identified numerically with
  Miller indices in a non-cubic cell;
- Cartesian frames are representation layers, not alternative material laws.

This allows PTCLab-compatible Cartesian input/output without abandoning the
metric formulation needed for monoclinic Cu-Al-Ni.

## 3. Correspondence variants: Cayron H_C, left cosets and double cosets

The existing exact group-theory layer implements the correspondence subgroup

    H_C^A = G_A ∩ C_A->M G_M C_M->A

under the package's explicit coordinate convention, then:

- correspondence variants = left cosets `G_A / H_C`;
- intercorrespondence operators = double cosets `H_C g H_C`;
- groupoid/operator composition remains a separate exact structure.

For the truth-locked DO3 -> 6M reference branch this gives 12 correspondence
variants and 8 intercorrespondence operators.

## 4. Orientation variants: fixed in this milestone

For an OR stored as

    x_A = R_A<-M x_M,

the orientation intersection subgroup is calculated as

    H_T^A = G_A ∩ R_A<-M G_M R_A<-M^-1.

This milestone fixes the previous over-counting error.  Distinct orientation
variants are **left cosets**, not every raw matrix from `S_A R S_M^-1`.

Two parallel calculations are kept:

1. **full crystallographic groups** for Cayron topology;
2. **proper SO(3) subgroups** for physical rotation-matrix representatives.

For the current DO3 -> 6M polar candidate:

    |G_A| = 48, |G_M| = 4, |H_T| = 4  -> N_T = 12

and in the proper SO(3) parts:

    |G_A+| = 24, |G_M+| = 2, |H_T+| = 2 -> N_T+ = 12.

The old list of 24 raw proper matrices therefore contained two
symmetry-equivalent matrix representatives for each of 12 crystallographic
orientation classes.

## 5. Orientation operators: Cayron double cosets

Orientation operators are now calculated from

    O_k^T = H_T g_k H_T.

The code reports:

- the number and sizes of full double cosets;
- Cayron-style `ambivalent` vs `polar` classification;
- whether an operator contains a parent reflection (Type-I candidate);
- whether it contains a parent 180-degree rotation (Type-II candidate);
- a physical symmetry-reduced disorientation representative.

For the present DO3 -> 6M polar candidate the orientation topology contains 8
full operators.

## 6. H_T must never be assumed equal to H_C

This is a central correction.

Cayron explicitly distinguishes orientation, correspondence and distortion
intersection groups and notes that their equality in important SMA examples is
not a general rule.

The code now computes both independently.

For the current correspondence-derived polar candidate:

    H_T = H_C,
    N_T = N_C = 12,
    O_T = O_C = 8,

so a one-to-one orientation/correspondence topology is reported **as a computed
result for this state**.

A generic user-defined DO3/6M OR is included in the tests/examples.  It gives a
smaller H_T and 24 orientation variants while H_C remains unchanged.  This
proves in software that the equality is not hard-coded.

## 7. M/M twins: already parallel to Cayron's CT path

The existing correspondence/groupoid/twin layer follows Cayron's logic:

- parent reflection -> Type-I possibility;
- parent 180-degree rotation -> Type-II possibility;
- other operators can remain polar/weak candidates;
- rational Type-I `K1` and rational Type-II `eta2` are correspondence/symmetry
  objects;
- the complementary irrational twin elements depend on the martensite metric.

The orientation-topology layer now gives the matching H_T/operator structure
without collapsing it into H_C.

## 8. A/M compatibility: CMC path already separate

The compatibility engine uses Cayron's metric-correspondence construction

    CMC = C^T M_M C - M_A

and checks the degeneracy condition

    q_i = 0,  q_j q_k <= 0.

Habit-plane solutions remain tied to that degeneracy, while the normalized
metric bridge permits independent comparison with Ball-James principal
stretches.

## 9. Shear extraction and A/M/M supercompatibility

The next layers already distinguish:

- SMC from CMC;
- A/M IPS shear from the habit plane;
- M/M twin shear;
- the CT shear/shear residual;
- Ball-James/cofactor checks as an independent theory path.

The code must continue to state only the implication supported by Cayron 2026:
CT A/M + M/M + shear/shear conditions imply the supercompatibility/cofactor
conditions.  Full reciprocal equivalence is not assumed.

## 10. Quaternion path

The OR CLI currently prints a standard Cartesian Hamilton quaternion for user
convenience.  This is explicitly labelled as such.

Cayron's 2026 crystallographic quaternion/crossmetric formalism is a separate
future metric-native adapter and must not be silently identified with the
Cartesian Hamilton output.

## 11. Exact research workflow going forward

The intended comparison pipeline is now:

```text
composition / state
 -> phase + cell representation
 -> metrics + full/proper symmetry
 -> correspondence C
 -> H_C -> C variants -> C operators/groupoid
 -> candidate orientation T/R
 -> H_T -> orientation variants -> orientation operators
 -> compare H_T vs H_C (never assume equality)
 -> M/M CT twins
 -> A/M CMC habit compatibility
 -> SMC / IPS shear
 -> A/M/M shear-shear supercompatibility
 -> Ball-James / cofactor / PTMC independent solutions
 -> experimental OR / EBSD / twin trace / habit-plane comparison
```

The next scientific milestone after this correction should therefore be a
**theory-adapter layer** that emits independently labelled candidates:

- `T_CT` from Cayron's natural/closing-gap OR construction;
- `R_BJ` from Ball-James rank-one solutions;
- `R_PTMC` from classical PTMC;
- measured `R_EBSD`;
- the existing correspondence-derived `R_polar` comparison candidate.

All of them should enter the same comparison schema without one theory being
used to generate another theory's answer.
