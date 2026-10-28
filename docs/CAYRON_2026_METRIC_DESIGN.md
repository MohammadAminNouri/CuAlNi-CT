# Lattice-parameter suggestions for Cayron 2026 theory comparison

Primary source: C. Cayron, *Compatibilities and supercompatibility conditions
in shape memory alloys determined from correspondence, metrics and
symmetries*, Acta Materialia 316 (2026) 122399,
https://doi.org/10.1016/j.actamat.2026.122399.

The proposal in §5.3 is to select a valid A/M invariant-plane family
first, then find lattice parameters that close the M/M twin and A/M
IPS shear/shear equation (Eq. 43). A lambda2=1 target is only the
single-variant A/M part. The cofactor conditions apply to a specified,
valid M/M twin branch, not all twins simultaneously.

## Universal metric calculation (not paper-specific C1)
For the active direct lattice correspondence C_(M<-A), with positive
definite M_A and M_M:

    G = C.T M_M C
    G v_i = mu_i M_A v_i
    lambda_i = sqrt(mu_i), ordered

The middle principal stretch lambda2=1 is equivalent to the
non-strict generalized metric degeneracy. The app varies only free
product-cell parameters allowed by its registered point group.
Uniform product-cell scaling is a generic exact A/M scaling direction:
all lambda_i scale by the same factor, but this does NOT guarantee
M/M or shear/shear compatibility.

The search is finite and bounded. All sign-changing roots in the
sampled intervals are refined by Brent, with extra local-minimum
checks for tangent candidates. No numerical root outside the user
window is asserted; no finite sample is exhaustive. Near eigenvalue
crossings, finite-difference sensitivities may be ill-conditioned.

## Cayron §5, specifically B2→B19' (source correspondence only)

Dimensionless a,b,c are the B19' conventional cell lengths divided by
cubic B2 cell parameter a0, with angle beta in degrees. This
normalization and the source-selected correspondence are essential.

C1 A/M IPS:
    b = sqrt(2)
    2 a^2 + c^2 - a^2 c^2 sin^2(beta) >= 2

Other analytic C2/C3 branches are implemented in the pre-existing
twin_app/cayron_2026_reference.py audit. Do not silently treat them as
Cu–Al–Ni design laws. Do not use the paper's internally inconsistent
printed O2 Type-I family as a benchmark; the source-derived correction
is explicitly unconfirmed. C2b Table-2 and Eq.-47 inequality signs
also disagree and are displayed unresolved.

## Two-stage candidate audit

1. Display deterministic parameter targets with actual metric/point-group
   validation, derivatives, exact lambda2 residual and provenance. None
   changes the user's primary input editor.
2. On explicit request, evaluate the selected metric target using the
   already existing CT exact A/M CMC/SMC and A/M/M shear/shear machinery,
   Ball--James M/M and cofactor checks, preserving matched/unmatched
   branch conditions and reporting unevaluable cases.
3. Optionally compute the full independently existing CT/BJ/PTMC
   comparison on that **same** candidate state. The PTMC laminate
   fraction problem must not be confused with CT's individual-variant
   CMC solutions.

This is a target in crystal metric space, not a prescription for how
alloy composition, annealing, defect density or processing changes
individual cell parameters. Chemical/thermodynamic feasibility and
measured lattice uncertainty require separate experiments or data.
