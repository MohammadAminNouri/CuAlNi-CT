# Immutable scientific benchmarks

This directory contains **versioned input/output contracts** for validation cases.

## Rule

A benchmark has two scientifically distinct blocks:

- `input`: the only information passed to a solver;
- `expected`: literature or previously frozen blind-result targets used only after
  the calculation has finished.

Each block is SHA-256 locked in two places:

1. inside its manifest;
2. inside `tests/test_ct_permanent_benchmarks.py`.

Changing a value and updating only the JSON lock therefore fails. If a primary
source correction, convention correction, or better specimen state is needed,
create a new benchmark ID/version and keep the old one for provenance.

## Blindness

"Blind" here means that a reported answer (variant count, twin plane, shear,
etc.) is not used to construct the calculation that is then compared with it.
Legitimate theory inputs such as lattice parameters, point groups,
correspondence, or an explicitly published OR are allowed when that is the
quantity the paper assumes.

A benchmark can still have a declared scientific boundary. For example,
Cayron's NiTi natural OR is an additional theory hypothesis and is therefore
recorded but **not** claimed as a blind prediction from correspondence and
metrics alone.

## Synthetic generality tests

Cross-crystal-system random/adversarial tests live in
`tests/test_ct_generality_campaign.py`. They test algebraic and coordinate
invariants only. Their synthetic numbers are not material predictions.
