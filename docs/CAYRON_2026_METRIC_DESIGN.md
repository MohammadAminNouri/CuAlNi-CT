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

## Conditional two-parameter A/M/M continuation (implemented)

The interactive **Design toward full A/M/M shear/shear compatibility**
control follows the paper's *sequence*, rather than assuming CC1 alone
proves supercompatibility:

1. Hold the entered correspondence, parent metric and phase symmetries fixed.
2. Select one independent product-cell parameter to target lambda2=1.
3. Vary a **different** allowed product-cell parameter over 3, 5 or 7
   explicit grid states. At each state retarget the first parameter by
   generalized-metric root solving, preserving lambda2=1 at the
   configured numerical tolerance.
4. For each candidate independently run the existing CT A/M exact
   CMC/SMC and M/M + A/M/M shear/shear evaluation, Ball--James M/M
   rank-one geometry and **cofactor CC1, CC2, CC3**. Do not equate
   a small shear/shear residual with CC2 or CC3.
5. Match CT and Ball--James physical twin branches. Display one
   *matched pair's* shear/shear epsilon and cofactor residuals, never
   a false combined score made from minima of unrelated twin branches.
   A pair whose required observable matching is partial is explicitly
   excluded from recommendations.
6. Rank only those **sampled**, fully matched, evaluable candidates
   by the established Cayron shear/shear epsilon, and report whether
   the second parameter should increase, decrease, stay unchanged,
   or whether no defensible suggestion exists.
7. Preserve actual a, b, c, beta, the retuned primary parameter,
   lambda2 residuals, all sampled states, all missing/evaluation statuses,
   matching completeness and source DOI in a JSON export.

No branch can be promoted to physical twin evidence merely because its
operator/twin integer IDs appear alike at several lattice states.
The optimizer does **not** prove monotonicity, global completeness,
experimental attainability, or an exact epsilon=0 root. It is a numerical
lattice-metric *hypothesis generator*. The measured correspondence,
preferred twin mode, physical composition->metric relationship and EBSD
variant populations remain independent questions.

Do not apply NiTi's B2->B19' **C1 b=sqrt(2)** source branch to the
orthorhombic Cu-Al-Ni 2H or modified monoclinic 18R input without
rederiving it for that structure and its exact correspondence.
