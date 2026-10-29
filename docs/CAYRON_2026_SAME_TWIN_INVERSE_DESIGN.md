# Same-twin inverse supercompatibility — scientific implementation (29 Oct 2026)

**Scope:** `feature/twin-family-tree-app-v1`, Section 04. This is a new
additive workflow. The original CT, Ball–James, PTMC, CTF/EBSD, variant
catalogue, and Cayron metric-design panels are retained. No main merge.

## Exact physical twin identity (Cayron 2022)

A source-published Type-I or Type-II *transformation twin* is generated
from a specific exact order-two parent symmetry operation, mapped to the
martensite through the declared correspondence. CT generic rational
elements are inherited from that parent generator. In the present
algorithm a twin identifier is an SHA256 shorthand for the canonical
**exact 3×3 rational parent generator plus construction route**. This
identifier does **not** depend on arbitrary double-coset list ordering,
shear-based sorting, or the number of computed variants.

The input parent metric, correspondence, parent/product point groups,
and phase IDs must be unchanged. A candidate must reconstruct that
same generator and remain physically continuous in all of:
- parent-reciprocal twin plane (metric-aware projective angle);
- parent-direct twin shear direction (metric-aware projective angle);
- shear magnitude (bounded adjacent-state ratio);
- the *selected* CMC A/M habit plane (separate metric-aware projective angle);
- a nonzero, finite physical twin shear.

These are necessary conservative continuity checks for one CT
construction family. A generator alone is insufficient to certify
continuity at a degeneracy, and it does not identify an EBSD-measured
physical twin or variant population. Compound CT twins can admit two
generator routes; these are preserved as distinct construction
histories, not merged by an output-array index.

Relevant code: `twin_app/same_twin_continuity_2026.py`.

## Bounded exact A/M manifold and inverse optimization (Cayron 2026)

Let `M_A`, `M_M(q)` be parent/product metric matrices with a fixed
correspondence `C` and a vector `q` of crystal-family-independent
product-cell parameters. For *each* proposed `q`, calculate the
ordered square roots of the generalized eigenvalues

    C.T M_M(q) C v = lambda² M_A v.

Eliminate the single A/M compatibility equality **exactly** by applying
a uniform product scale `u(q)=1/lambda_2(q)`; this scales each
principal stretch by `u(q)` and is independently checked at each
candidate. It is not an unconstrained perturbation of the parent
metric or correspondence. Reject any final `a,b,c,alpha,beta,gamma`
outside the explicitly selected cell bounds. Invalid unit cells or
changed crystallographic symmetry fail.

For each admissible cell, calculate the two (when they exist) exact
CMC habit-plane candidates, choose *one* anchored habit, derive its IPS
shear through SMC, regenerate the **fixed CT twin**, and calculate

    r_42 = 2 (m_A.T n) d_A - s eta_A
    epsilon = ||r_42||_(M_A) / |s|.

Report both vector sides, longitudinal amplitude gap and transverse
direction mismatch. Numerical epsilon may only be evaluated when an
exact A/M IPS exists. Tangencies, collapsed twins, ambiguous habits
and discontinuities remain **not evaluable**. The physical signed
shear is never replaced by an arbitrary "better" twin.

Numerical method: bounded, derivative-free Powell on the remaining
symmetry-permitted independent `q` coordinates; max evaluations is
a hard cap. A local objective (not a global solution theorem):

    epsilon² + w_CC2 * (||U^(±1)e||² - 1)²
             + w_change * || normalized(q-q0) ||².

The standard numerical weights are `w_CC2=0.25` and
`w_change=1e-5`, exposed/recorded. The source's equivalent CCI/CCII
test determines the sign: Type-I uses `U^-1` on the rational mirror
normal; Type-II uses `U` on the rational twofold axis. These are
the Cayron/Zhang equivalent **CC2** equalities only, not CC3.

The best numerical candidate is not accepted until 5–11 intermediate
cells, *including the original lambda2-corrected anchor and final
candidate*, all preserve the exact generator and one continuous
CM C habit. The unchanged, exact-lambda2 reference is evaluated
explicitly as a fallback. An infeasible candidate never becomes
evidence of a supercompatible alloy.

A leave-one-factor-out analysis repeats a proposed state with just
one independent modification removed, re-eliminates lambda2, and
recalculates Eq.42 and equivalent CC2. The difference identifies a
**conditional contribution** at the proposed cell, not an additive
chemical or mechanical causal effect. The solver reports changes in
the entered length units, percentage length change, angular degrees,
all intermediate metric values and residuals. No prior experimental
cell is edited.

Relevant code: `twin_app/same_twin_inverse_design_2026.py`,
`twin_app/same_twin_cofactor_metric_2026.py`.

## Ball–James CC1/CC2/CC3 verification: strict second stage

The source's full cofactor conditions are

    CC1: lambda2 = 1
    CC2: a dot U cof(U²-I) n = 0
    CC3: tr(U²) - det(U²) - |a|²|n|²/4 - 2 >= 0

with the actual Ball–James twinning vectors `a,n` in a coherent
configuration. We do **not** substitute CT's physical shear magnitude
for `|a||n|` without independently verifying the reference/current
configuration mapping. The Type-I/II eigenvector metric condition is
a *separate source-derived CC2 equivalent*, not a fictitious full
Ball–James assignment.

At the final proposed cell only, run the existing full CT and BJ
backends. The selected A/M habit must be independently identified by
**geometrical plane angle**, not a local habit array index. The selected
CT generator must match the atlas CT/BJ M/M twin assignment with all
required physical observables present and agreeing within tolerance.
One same-pair BJ cofactor result must satisfy CC1–CC3, *and* the selected
habit must independently close Eq.42. Selecting a BJ result from another
generator or a CT habit from another state is forbidden. If the CT/BJ
matcher intentionally marks reference/current shear directions as
non-comparable, report **UNRESOLVED** rather than an invented
certificate.

A report may state `CT_BJ_SAME_PAIR_NUMERICALLY_SATISFIED` only for
that fully assigned numerical condition. It never states achieved
fatigue resistance, reduced hysteresis, new alloy composition or
experimental EBSD-confirmed twins.

## Paper-derived independent acceptance tests

2022 Crystals, 12, 130, NiTi B2→B19′:
- parent cubic group order 48, correspondence subgroup order 4,
  12 M variants and 7 double-coset operators;
- Type-I mirror plane and Type-II twofold axis remain rational
  generator invariants;
- compound/weak twin families are not conflated with a unique
  ordinary M/M rank-one shear.

2026 Acta Materialia, 316, 122399:
- independently reproduce source Tables 2–3, habit planes, C1
  nonzero Eq.42 epsilon, O4 Appendix-C exact-compatible target;
- reject O2 Type-I **printed** `a=sqrt(2)/sin(beta), b=c=sqrt(2)`
  as a full CCI pass: squared `||U^-1 e||²=0.5`, not 1;
- the paper-equation-derived (not author-confirmed) value
  `a=1/sin(beta)` restores squared CCI=1;
- O2 Type-II `a=1,b=sqrt(2),c=sqrt(2)/sin(beta)` gives
  squared `||U e||²=1`;
- negative controls reject invalid metrics, guessed generator
  IDs, changed parent identity, metric-only false SC certificates,
  and local twin-/habit-branch swapping;
- repeat both 2022 and 2026 references under an independent
  focused GitHub Actions check.

## Unresolved experimental work for Cu–Al–Ni

The app has **not** inferred a validated parent→2H or
parent→modified-18R correspondence from CTF alone, a unique
experimental OR, a stress/temperature-conditioned martensitic
transformation, experimentally measured variant frequency, or
composition→lattice-parameter response. These require physical
observations and/or independent thermodynamic/mechanical inputs.

The optimization bounds are user-defined *hypothetical crystal
metrics*. Optimization convergence does not prove global
supercompatibility, manufacture feasibility, high fatigue resistance,
or a defensible physical composition change. Uncertainty in measured
lattice parameters and the parent reconstruction should be supplied
before turning a nominal numerical equality into an experimental
claim.

Primary sources:
- Cayron (2022), DOI 10.3390/cryst12020130;
- Cayron (2026), DOI 10.1016/j.actamat.2026.122399.
