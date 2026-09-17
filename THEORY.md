# Theory map

## 1. Cayron: variants and operators

The correspondence subgroup in the parent is

`H_C^A = G_A ∩ C_{A<-M} G_M C_{M<-A}`.

With the package convention `C = C_{M<-A}`, this is

`H_C^A = G_A ∩ C^{-1} G_M C`.

Correspondence variants are left cosets `g_i H_C`.

Intervariant correspondence classes are double cosets `H_C g_k H_C`, called operators by Cayron. The indices are arbitrary; the sets, not their labels, are the invariant objects.

Do not assume correspondence, orientation, distortion and stretch variants are the same.

## 2. Cayron Type-I / Type-II transformation twins

For a parent reflection on `p_A`:

- `p_M = C^{-T} p_A` gives rational Type-I `K1`;
- `C_int = C m_A C^{-1}`;
- `s² = tr(C_int^T M_M C_int M_M^{-1}) - 3`;
- the non-generic shear direction depends on `M_M`.

For a parent 180° rotation around `a_A`:

- `a_M = C a_A` gives rational Type-II `eta2`;
- `C_int = C R_pi C^{-1}`;
- `s*² = tr(C_int M_M^{-1} C_int^T M_M) - 3`;
- the Type-II `K2` plane depends on the product metric.

Parent rotations of order other than two lead in Cayron CT to nonconventional/weak-junction candidates rather than perfectly compatible twin planes.

## 3. CMC and A/M compatibility

`CMC = C^T M_M C - M_A`.

A parent direction `u` keeps its length if

`u^T CMC u = 0`.

A coherent single-variant A/M IPS corresponds to a degeneracy of this quadratic cone. In the normalized metric-whitened representation, compatibility is detected by a zero eigenvalue with the correct signature of the remaining eigenvalues.

## 4. SMC

With `u_M=C u_A`, this package writes

`SMC = M_A^{-1} - C^{-1} M_M^{-1} C^{-T}`.

For a normalized habit-plane covector `m_A`:

`d_A = SMC m_A`.

## 5. Cayron full supercompatibility

For an A/M IPS and an M/M twin to be mutually compatible, Cayron uses

`2 (m_A^T n) d_A = a`,

with `n` the twin-plane unit normal and `a` the twin-shear vector in the parent basis.

The package reports the normalized residual

`epsilon_CT = ||2(m_A^T n)d_A - a|| / s`.

## 6. Ball–James bridge

The same correspondence and metrics give

`U^T M_A U = C^T M_M C`.

In the parent metric-whitened orthonormal basis:

`U² = M_A^{-1/2} C^T M_M C M_A^{-1/2}`.

Thus normalized CMC eigenvalues are `lambda_i² - 1`.

M/M rank-one compatibility is

`R U_j - U_i = a ⊗ n`.

## 7. Cofactor conditions

For a specified twin system `(a,n)`, Chen et al. give:

- CC1: `lambda_2 = 1`
- CC2: `a · U cof(U²-I) n = 0`
- CC3: `tr(U²) + det(U²) - |a|²|n|²/4 - 2 >= 0`

Together they make the crystallographic theory solvable for every twin volume fraction `f in [0,1]` for that twin system.

## 8. PTMC branch in this repository

For a twin relation written in a common reference frame as

`U_2 = U_1 + a⊗n`,

a laminate has

`F(f) = U_1 + f a⊗n`.

The classical crystallographic-theory condition is detected numerically by requiring the middle singular value of `F(f)` to equal 1, followed by a rank-one A/M connection

`R F(f) - I = b⊗m`.

The numerical implementation is intentionally separated from CT so that the theories can be compared using identical input data.
