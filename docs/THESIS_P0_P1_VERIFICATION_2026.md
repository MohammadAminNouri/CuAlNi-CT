# Thesis P0/P1 verification — mathematical scope and limitations

Status: computational certification only, not specimen identification.
The report uses the existing parent and product metrics, correspondence,
symmetry groups and CT/Ball–James/PTMC outputs. It introduces no new
lattice or experimental data, no orientation angles and no solver changes.

## P0 — independent passive crystal-basis covariance

The explicit direct-space correspondence is u_M = C u_A, while a
reciprocal plane covector transforms as p_M = C^{-T} p_A. Therefore
p_M^T u_M = p_A^T u_A.

For independent parent and product passive changes P_A and P_M in
GL(3,Z) with determinant ±1:

    M'_A = P_A^T M_A P_A
    M'_M = P_M^T M_M P_M
    C'   = P_M^{-1} C P_A
    g'_A = P_A^{-1} g_A P_A
    g'_M = P_M^{-1} g_M P_M
    u'_A = P_A^{-1} u_A
    p'_A = P_A^T p_A

The following must hold without adjusting physical inputs:

    G = C^T M_M C
    CMC = G - M_A                   (Cayron 2026)
    SMC = M_A^{-1} - G^{-1}         (Cayron 2026)
    CMC' = P_A^T CMC P_A
    SMC' = P_A^{-1} SMC P_A^{-T}
    generalized eigenvalues of (G, M_A) unchanged
    principal stretches lambda_i = sqrt(mu_i) unchanged

Independent checks cover parent-only, product-only and simultaneous
parent/product basis changes. These are coordinate relabelings,
not new physical shears, OR choices or alloy parameters.

**Important:** Cayron's Acta Crystallographica A75 (2019) corrigendum,
DOI 10.1107/S2053273319009276, states that preserving a metric
is necessary but not sufficient to be an actual crystal symmetry.
The P0 audit checks integer lattice operations, determinants,
metric preservation, identity and finite group closure; it never
certifies atomic structural symmetry or a specimen's correspondence.

## P1 — equation-to-implementation audit

| Mathematical claim | Academic basis | Implemented and independently checked |
| --- | --- | --- |
| M/M Type I/II twins | Cayron 2026 Eqs. 19–25, double cosets | twinning_ct.py, CT branch identities and P0 group audit |
| A/M CMC | Cayron 2026 Eqs. 32 and 37 | ct.py, generalized eigenproblem + passive covariance |
| IPS shear SMC | Cayron 2026 Eq. 41 | ct.py, contravariant SMC covariance |
| A/M/M intercompatibility | Cayron 2026 Eqs. 43 and 49 | independently reconstructed epsilon, unchanged by parent rebase |
| BJ M/M rank-one | Ball–James 1987, James–Hane 2000 | RU_j-U_i=a outer n and proper SO(3) rotation |
| Cofactor CC1–CC3 | Chen et al. 2013 Theorem 2 | independent tensor and numerical residual reconstruction |
| PTMC laminate habit | WLR 1953; Bowles–Mackenzie 1954 | fraction, middle stretch and rank-one habit reconstruction |
| CT implies cofactor | Cayron 2026 Sections 2.3, 6 | one-way implication on physically matched CT/BJ twins only |

For parent crystal metric M_A, normalized reciprocal habit
covector m_A, normalized direct twin-plane normal n, IPS vector
d_A and metric-normalized twin shear a, independently check

    epsilon = ||2(m_A^T n) d_A - a||_(M_A) / |s|

against the software's stored numerical epsilon. Only exact
epsilon=0 satisfies the geometric equality. A near-zero value
within an explicitly reported tolerance is a numerical judgment.

**Cofactor source discrepancy:** use the MINUS sign in Chen
et al. 2013 Theorem 2 and proof:

    CC1: lambda_2(U) = 1
    CC2: a dot U cof(U^2-I) n = 0
    CC3: tr(U^2) - det(U^2) - |a|^2 |n|^2/4 - 2 >= 0

The abstract has an inconsistent PLUS det(U^2). These three
conditions require an admissible M/M rank-one twin, and their
all-volume-fraction conclusion is not proved by only 101
sampled fractions.

PTMC delta=1 represents the strict invariant-plane condition.
The delta!=1 extension must not be labeled strictly invariant
without further hypotheses.

## What the programmatic certificate actually establishes

- Three passive basis covariances, dual-vector incidence,
  point-group computational consistency and metric eigenspectrum.
- Up to 256 CT shear/shear branches independently reconstructed;
  COMPLETE or PARTIAL coverage is explicitly reported.
- Independently reconstructed Ball–James rank-one/cofactor
  tensors and PTMC laminate habits, with REVIEW_REQUIRED
  failures rather than silent acceptance.
- An audit tolerance of 1e-7, distinct from native solver tolerances.

What it CANNOT establish:

- That the declared group is the actual experimental crystal symmetry.
- That Cu–Al–Ni 2H and modified 18R share the same correspondence.
- That atomic stacking, phase identity, EBSD variant selection or
  measured twin/habit planes have been established.
- That the mathematical solvers have globally found every
  admissible branch or continuum root in all materials.

## Remaining P2–P4

P2: reproducible independent 2H and modified-18R benchmarks,
including negative and degenerate cases and independently checked
root completeness.

P3: actual phase-specific diffraction and EBSD evidence,
orientation uncertainties, physically matched variant interfaces.

P4: cite this commit, input lattice settings, C and its source,
group membership provenance, exact source equations, numerical
tolerances, branch-level residuals, uncertainty and experimental
status in each thesis table.

Thesis citations:
- Cayron, Acta Materialia 316 (2026) 122399,
  DOI 10.1016/j.actamat.2026.122399.
- Cayron, Acta Crystallographica A75 (2019) 411–437
  and corrigendum A75 (2019) 777.
- Ball and James, Archive for Rational Mechanics and Analysis
  100 (1987) 13–52.
- Chen et al., Journal of the Mechanics and Physics of Solids
  61 (2013) 2566–2587.
- Wechsler, Lieberman and Read (1953);
  Bowles and Mackenzie (1954).
