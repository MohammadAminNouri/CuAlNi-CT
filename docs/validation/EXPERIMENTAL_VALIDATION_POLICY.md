# Experimental validation branch policy

## Frozen scientific core

The scientific backend is frozen at:

```text
tag:    ct-ebsd-reliability-gate-v1
commit: f7c616379108c0ec06d804e4692cd437c35ec4e0
```

`next/experimental-validation` is **not** a continuation of core mathematical
development.

The purpose of this branch is to test the frozen backend against new,
independent evidence.

## Allowed work

Changes on this branch are restricted to:

- `validation/` — case definitions, manifests, run metadata, evaluation tools;
- `evidence/` — compact, reviewable evidence and provenance;
- `docs/validation/` — validation notes and protocols;
- `examples/validation/` — validation examples;
- `tests/validation/` — tests of validation infrastructure/adapters;
- `src/cualni_cryst/io_adapters/` — input adapters only;
- `.github/workflows/experimental-validation-guard.yml`;
- `tools/check_experimental_validation_scope.py`.

Raw EBSD datasets and large binary outputs are not committed. Store them
outside git and preserve cryptographic hashes and provenance instead.

## Frozen work

Do not modify on this branch:

- CT equations;
- variant/operator mathematics;
- orientation quotient mathematics;
- twinning equations;
- weak-twin mathematics;
- Ball--James / Mallard comparison code;
- EBSD inference equations;
- reliability gates;
- production scientific acceptance thresholds;
- existing scientific expected values.

The scope guard rejects such changes.

## Genuine counterexample protocol

A real counterexample is not "a dataset did not give the expected answer".

Before changing the core:

1. Freeze the failing input and record its SHA-256.
2. Record the exact frozen backend tag and complete configuration.
3. Produce the smallest reproducible case that still fails.
4. Demonstrate that the failure is in the mathematical/scientific core rather
   than parsing, metadata, convention, insufficient evidence, or data quality.
5. Cross-check independently against theory, a second implementation, or an
   analytical calculation where possible.
6. Do **not** tune tolerances or expected values to make the case pass.
7. Create a separate branch from `ct-ebsd-reliability-gate-v1`, named for
   example:

   ```text
   investigate/core-counterexample-<short-name>
   ```

8. Add the failing regression first.
9. Only then modify the core.
10. Re-run the full repository and every frozen scientific benchmark before
    considering a new core milestone.

Until those conditions are met, an experimental failure is evidence to study,
not permission to rewrite the mathematics.

## Validation evidence standard

Each real-data validation case should preserve:

- input SHA-256 and original filename outside the solver-visible challenge;
- acquisition/export metadata that are legitimately known;
- adapter name/version;
- exact backend tag/commit;
- exact configuration;
- runtime environment;
- gate decisions (`allowed`, `ambiguous`, `insufficient_evidence`, etc.);
- compact human-readable result;
- hashes of large numerical sidecars;
- independent comparison performed **after** the blind prediction is frozen.

The preferred outcome is not always an inferred OR. A scientifically correct
`insufficient_evidence`, `ambiguous`, or reliability-gate rejection is a valid
result.
