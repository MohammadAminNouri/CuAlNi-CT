# Science Novel Hardening V1

Base frozen milestone:

`professor-ready-backend-v1`
`0f516e879d663f8abd0a290e6293080814ca8ba3`

## Why this batch exists

This batch was designed only after reading the frozen repository. It deliberately
does **not** repeat already-covered work:

- direct/reciprocal correspondence convention;
- ordinary user input facade;
- metric/Cartesian representation parity;
- 32-point-group finite-order route classification;
- existing weak-twin Cayron-2022 benchmarks;
- existing weak-operator NiTi blind benchmark;
- existing cross-crystal-system CT generality campaign;
- existing permanent Cayron/Chen benchmarks;
- existing CMC first-order C1 benchmark;
- existing Ball-James/PTMC adapter regression.

The new attack surface is:

1. **Coupled CMC -> habit -> SMC -> IPS -> shear/shear covariance**
   under *independent* parent/product exact basis changes and extreme common
   length-unit rescaling.

2. **Exact double-coset hypergroup laws**
   with an independent set-product oracle, identity/inverse laws and support
   associativity. Quick profile uses six representative crystal families; full
   profile covers all 32 crystallographic point groups and multiple subgroups.

3. **Cayron 2026 beyond the existing C1 lock**
   - Table-3 SMC shear vector;
   - Table-3 incompatibility epsilon and angle;
   - analytic O2 Type-I supercompatible family at several beta values;
   - analytic O2 Type-II supercompatible family at several beta values;
   - cofactor-condition cross-check on those CT-derived states;
   - negative controls that separate A/M compatibility from true A/M/M
     supercompatibility.

4. **Blindness firewall**
   Core solver modules are forbidden from embedding permanent benchmark IDs,
   and a representative CT solve is executed while runtime access to
   `data/benchmarks` is denied.

## Scientific restraint

The 2026 Cayron paper explicitly reports numerical agreement of its
supercompatible B19' solutions with the cofactor conditions but also states
that a complete formal CT/PTMC equivalence was not established. These tests
therefore cross-check only the published analytic families; they do **not**
assert a universal theorem.

The repository's existing `weak_planes.py` intentionally refuses to fabricate
the unpublished GenOVa generic weak-plane search. This batch preserves that
boundary.

## Run order

Do not run the full repository suite first.

1. `python tools/run_science_novel_hardening.py --profile quick`
2. If PASS: `python tools/run_science_novel_hardening.py --profile full`
3. If either fails, preserve the full failure output. Do not loosen tolerances
   or change expected values to make a test green.
4. Only if a genuine core fix is required: patch the root cause, rerun the new
   tests, then run the relevant old regression files, and only once at the end
   run the expensive full repository suite.

A failure in the 2026 quantitative tests may indicate a convention mismatch,
a genuine implementation gap, or a mistaken test derivation. Diagnose it
before modifying scientific code.

## Cayron 2026 O2 Type-I source-consistency audit

The temporary strict-XFAIL used during diagnosis has been retired. The discrepancy
is now represented as a positive source-consistency regression rather than as an
expected solver failure.

The independent audit does not import `cualni_cryst`. It directly transcribes the
paper's correspondence matrices, Eq. 41 SMC construction, the Type-I cofactor
criterion, the O2 Type-II control, and the O4 Appendix-C control.

Permanent provenance states are kept distinct:

- `VERIFIED_PUBLISHED_RESULT`: an independently reproduced published result;
- `SOURCE_INTERNAL_INCONSISTENCY`: a printed source claim that does not satisfy the
  source's own equations under the published inputs/conventions;
- `PROJECT_DERIVED_RESULT`: an equation-derived result from this project which must
  not be attributed to the source author without confirmation.

For the printed O2 Type-I relation `c=sqrt(2), a=c/sin(beta)`, direct symbolic
substitution gives both Eq. 43 normal-component residual `d_z = 1/2` and the
independent Type-I CCI quantity `||U^-1 e||^2 = 1/2`. The equation-derived closure
`a=1/sin(beta)=c/(sqrt(2) sin(beta))` gives `d_z=0` and CCI squared `=1`.

The repository must never modify the frozen CT/SMC backend merely to force the
printed source relation to pass. The source relation and the project-derived closure
remain recorded separately.
