# Conventions — read this first

Convention errors are more dangerous than numerical errors in this project.

## Direct and reciprocal coordinates

A direct crystallographic direction is a column vector `u`.

A crystallographic plane is a reciprocal covector `p` and the incidence relation is

`p.T @ u = 0`.

With direct metric `M`:

- `|u|^2 = u.T M u`
- `|p|^2 = p.T M^{-1} p`
- the direct-space normal corresponding to plane `p` is proportional to `M^{-1} p`.

Never treat a monoclinic `(hkl)` as if it were automatically the Cartesian vector `[h,k,l]`.

## Correspondence convention

Internally the package uses

`u_M = C_m_from_a @ u_A`.

This is the action that Cayron describes for the matrix he denotes `C^{M→A}` in his 2026 paper. Because the superscript notation can be counter-intuitive, the code uses explicit names:

- `C_m_from_a`
- `C_a_from_m = inverse(C_m_from_a)`

Plane transformation follows from duality:

`p_M = C_m_from_a^{-T} p_A`.

## Full vs proper cubic symmetry

- full `m-3m`: 48 operations, including reflections/inversion;
- proper cubic rotations: 24 operations in `SO(3)`.

CT needs the full crystallographic group because parent reflections are physically meaningful in its Type-I construction. Ball–James stretch-variant generation uses proper rotations.

## Dimensional vs normalized metrics

Physical calculations keep measured lengths and therefore `M` has length² units.

For cross-composition comparison we additionally form

`G_hat = M_A^{-1/2} C^T M_M C M_A^{-1/2}`

and `CMC_hat = G_hat - I`.

This normalization is a project comparison device. It does not replace Cayron's dimensional metric formulation.
