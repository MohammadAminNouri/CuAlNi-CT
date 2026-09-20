# Cayron axial weak twins: production contract

## Scope

This layer implements the reticular axial weak-twin mathematics described by
C. Cayron, *Acta Materialia* **236** (2022) 118128, and a CT-constrained weak
plane search for an already-known exact inter-correspondence.

It is deliberately **not** presented as a byte-for-byte reconstruction of
GenOVa's internal candidate ordering. The generic GenOVa search constructs
supercells by enumerating close-length direction pairs and 16 third-vector
pairs. In the CT use case the inter-correspondence is already supplied by the
correspondence operator, so the scientifically necessary problem is narrower:
find low-index axial weak-plane pairs compatible with that exact
inter-correspondence, calculate the reticular orientation/distortion, and then
construct proper parent/product closing-gap OR candidates.

## Objects kept separate

The implementation never identifies the following objects:

- `C`: crystallographic correspondence;
- `T`: reticular orientation/isometry between weak-twin lattices;
- `F`: active distortion, `F = T^{-1} C`;
- physical parent/product OR: a proper Cartesian rotation in `SO(3)`.

For weak twins Cayron explicitly allows `T` to be a rotation or a
rotoinversion in the reticular theory. Therefore an improper `T` is retained
as such and is **not** silently stored as a physical `OrientationState`.

## Bravais lattice

A weak-twin calculation is a lattice-node calculation. Conventional cell
indices are not automatically a primitive lattice. `BravaisNodeBasis` is
therefore mandatory for the CT weak adapter.

The convention is

```text
u_conventional = P u_primitive
p_primitive    = P^T p_conventional
M_primitive    = P^T M_conventional P
C_primitive    = P^-1 C_conventional P
```

A C-centered unique-b primitive node basis is supplied explicitly. No
centering is inferred from `2/m`, `C2/m`, a material name, or a cell label.

## Cayron 2022 equations

For a specified weak-plane pair and supercell correspondence,

```text
C_2<-1 = B_supC,2 B_supC,1^-1
q_g    = |det(B_supC) / det(B_c)|
T_2<-1 = B_supT,2 B_supT,1^-1
F_1    = T_2<-1^-1 C_2<-1
```

The generalized shear is

```text
s_g^2 = tr[M (F-I) M^-1 (F-I)^T]
```

and the generalized strain is

```text
epsilon_g^2 = tr[M C M^-1 C^T] - 3.
```

The implementation evaluates every distinct sign branch in Cayron's Eq. (3)
and audits metric isometry, invariant-axis preservation, plane mapping, and
`C = T F` factorization.

## Exact generalized twin index

For a rational correspondence `C = A/d`, with integer `A`, the common-domain
sublattice is

```text
{x in Z^3 | A x == 0 (mod d)}.
```

If `s_i` are the Smith invariants of `A`, the exact index is

```text
q_g = product_i d / gcd(d, |s_i|).
```

This avoids floating determinant/index decisions.

## CT-constrained weak-plane search

For an exact CT inter-correspondence

```text
C_int = C_M<-A G_A C_M<-A^-1
```

and its inherited invariant product axis `u_M`, the search enumerates
primitive reciprocal planes `p1` with

```text
p1^T u_M = 0
```

and constructs

```text
p2 = C_int^-T p1.
```

The rational `p2` is reduced projectively to primitive integer indices. Each
pair is evaluated with Cayron's reticular equations. This is an exact search
for weak planes **conditional on the already-known CT correspondence**.

## Physical closing-gap OR

The dedicated `CTWeakOrientationAdapter` keeps the reticular weak-twin result
separate from the physical parent/product orientation. For the base product
variant, its weak plane `p_M` is mapped back to the parent by

```text
p_A = C_M<-A^T p_M,
u_M = C_M<-A u_A.
```

Proper Cartesian rotations satisfying the projective parallelisms

```text
u_A || u_M
p_A || p_M
```

are constructed and stored as `OrientationState` objects with theory origin
`CAYRON_CT`. If a natural OR is explicitly supplied, candidates are compared
using the existing symmetry-reduced OR comparison service.

## Published Mg regressions

Using the lattice parameters quoted by Cayron (`a = 3.21 A`, `c = 5.21 A`),
the regression tests reproduce the first two published a-axis weak twins.

Basal-prismatic sister:

```text
published: q_g = 2, s_g = 0.092, epsilon_g = 0.130
computed : q_g = 2, s_g = 0.0920332215, epsilon_g = 0.1300860007
```

Basal-pyramidal sister:

```text
published: q_g = 4, s_g = 0.107, epsilon_g = 0.137
computed : q_g = 4, s_g = 0.1068698448, epsilon_g = 0.1367030152
```

The tests also verify exact Smith-normal-form index calculation, direct/
reciprocal duality of a C-centered primitive basis, CT-constrained recovery of
the basal-prismatic pair, and loud rejection of a non-invariant axis.

## Scientific boundary

The existing `exact_genova_weak_plane_search()` refusal remains valid. This
milestone does not fabricate undocumented GenOVa implementation details. It
implements the published weak-twin equations and the narrower CT-constrained
problem needed by CuAlNi-CT.
