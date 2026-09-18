# A/M and A/M/M compatibility engine

## Why this layer exists

The previous milestones established the truth-locked DO3 -> 6M reference
correspondence, its twelve stretch variants, eight double-coset operators, and
the agreement of Cayron CT M/M twins with the independent Ball-James/Mallard
rank-one construction.

This layer connects those M/M results to the austenite/martensite interface.

The physical flow is

    lattice metrics + correspondence
        -> CMC
        -> exact A/M compatibility and habit planes
        -> SMC
        -> individual-variant IPS shear d_A
        -> CT shear/shear A/M/M residual
        -> independent cofactor test
        -> independent PTMC laminate solutions.

## Exact versus approximate is deliberately strict

For a parent direction u_A,

    u_A^T CMC u_A = 0

defines invariant-length directions. An exact coherent A/M habit plane requires
the CMC quadratic form to degenerate into plane factors. In eigenvalue form,

    q_i = 0,
    q_j q_k <= 0.

The source-rounded James-Hane Cu-Al-Ni benchmark is close to, but not exactly
on, that manifold. Therefore the software does NOT compute or report exact
A/M/M supercompatibility for the measured/rounded state.

Approximate CMC planes remain diagnostics only.

## Exact-compatible control state

For validation, the engine constructs a clearly labelled HYPOTHETICAL_TEST
state by changing only the monoclinic beta angle to the nearest exact
James-Hane Eq.(25) solution while keeping a0, a, b and c unchanged.

This is not a fit and not a measurement. It is a mathematical control that
allows the full exact chain to be tested.

With the rounded Table-4 lengths the nearest obtuse solution is approximately
95.24 degrees, whereas the rounded benchmark beta is 95.68 degrees. The exact
number is calculated by the code rather than hard-coded.

## Independent A/M cross-check

When the projected state satisfies exact CMC degeneracy, two habit planes are
obtained from CMC.

Independently, the Ball-James theorem solves

    R U - I = b ⊗ m

for a single martensite variant when lambda_2 = 1.

The code requires the two unordered CMC habit normals to match the two
Ball-James rank-one normals to numerical precision. This is a genuine
cross-theory test.

For cubic parent metric M_A = a0^2 I, the implementation also verifies

    CMC = a0^2 (U^2 - I)

and

    SMC = (1/a0^2) (I - U^-2).

## SMC and Cayron shear/shear compatibility

For each exact habit-plane covector m_A,

    SMC = M_A^-1
          - C_A_from_M M_M^-1 C_A_from_M^T

and

    d_A = SMC m_A.

For each exact M/M twin relation the software evaluates both CMC habit-plane
branches and retains the smaller Cayron incompatibility

    epsilon = || 2 (m_A^T n) d_A - a || / s,

where m_A and n are correctly normalized in reciprocal/direct parent metric
spaces and ||a|| = s.

An epsilon of zero is exact A/M/M supercompatibility. A finite epsilon is a
distance-like incompatibility amplitude, not an energy.

## Independent cofactor conditions

For the same M/M relation, but using the independent Mallard/Ball-James
rank-one data, the software evaluates Chen et al.:

    CC1: lambda_2 = 1

    CC2: a . U cof(U^2-I) n = 0

    CC3: tr(U^2) - det(U^2) - |a|^2 |n|^2/4 - 2 >= 0.

The projected beta enforces single-variant A/M compatibility (CC1), but the
code does not assume CC2 or CC3. Their raw residual/margin values are returned.

Chen et al. show that the cofactor conditions give crystallographic-theory
solutions for every twin fraction f in [0,1]. This is why the code treats them
as a stronger condition than lambda_2 = 1 alone.

## Independent PTMC branch

The PTMC module starts from the independent Mallard twin relation

    R_hat U_hat = U + a ⊗ n

and forms the twinned-martensite average

    Fbar(f) = U + f a ⊗ n.

It then solves for volume fractions for which the middle singular value of
Fbar is one and independently obtains A/M rank-one connections.

The PTMC roots are reported next to Cayron epsilon and cofactor residuals.
One theory is not allowed to feed its habit plane into another theory.

## Interactive-software direction

The core returns dataclasses that can be converted to a plain JSON-compatible
dictionary. This is intentional.

The final software should behave like a modern phase-transformation
crystallography laboratory: the user supplies phases, cells, correspondence,
orientation/experimental data and theory options; the backend recomputes
variants, operators, interfaces and compatibility immediately; the frontend
can then render stereographic projections, variant tables, interfaces,
diffraction/pole-figure views and theory-comparison diagnostics.

The numerical physics engine must stay independent from the eventual GUI so
that every visual result remains reproducible from tests and scripts.
