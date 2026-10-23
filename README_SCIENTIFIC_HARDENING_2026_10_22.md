# CuAlNi-CT — scientific hardening gate before Streamlit

Date: 2026-10-22. Target branch: `feature/twin-family-tree-app-v1`.

## Why this is safer than replacing the solver again

The working tree **already includes** independently tested fixes for:
- distinct stretch variant identity, correct symmetry generator provenance, and binary64 round-off;
- near-degenerate parent symmetry transport, Type I/II cross-lock and mode permutation uniqueness;
- near-tangent PTMC habit-fraction polynomial roots, with high-precision escalation;
- preservation of all reconstructed habit geometries in two benchmark inventories.

Do **not** reapply older repair ZIPs. The supplied installer checks **all seven important working-source SHA-256 digests against the successful classification-retention audit** and REFUSES any unexpected change. It does not replace the PTMC or Ball–James engines or manipulate their numerical tolerances.

### One substantiated interpretation problem

The data model's old UI-compatible `by_type('Unresolved')` class contains both genuinely unresolved classical classifications **and** branches for which the supplied parent mirror/twofold conventional route was excluded. These scientifically different states should not be conflated. The installer adds:

- `ModeEntry.evidence_category` — provenance claim, not a mathematical proof;
- `PhysicalTwinMode.by_evidence_category(category)`;
- `PhysicalTwinMode.evidence_counts()`.

Evidence categories are:
- `direct_classical_claim`
- `symmetry_transported_classical_claim`
- `parent_mirror_twofold_excluded_under_supplied_group`
- `unresolved_classical_provenance`

A false or malformed category triggers an exception. **No label is inferred from a geometric witness.** The original `by_type()`, `types`, existing mode IDs, shear, dyads, rotations, and habit solutions are unchanged. The browser still needs a separate audit and UI update to show this distinction to users.

## Commands (Codespace root only)

First upload `cualni_scientific_hardening_2026_10_22.zip` into `/workspaces/CuAlNi-CT`. Paste these commands one at a time, without leading quotation marks:

```bash
unzip -o cualni_scientific_hardening_2026_10_22.zip
python -u scientific_hardening_2026_10_22.py --preflight
python -u scientific_hardening_2026_10_22.py --install
python -u scientific_hardening_2026_10_22.py --verify
```

**Stop on the first nonzero exit or traceback.** Never weaken tolerances to make a check pass. If `--preflight` sees an unexpected hash, the source has changed since the last audit; this needs review before applying anything.

`--install` creates:

`audit_findings/scientific_hardening_2026_10_22/mode_catalogue_before_hardening.py`

plus `numerical_source_snapshot_before_hardening.zip` (the exact seven audited source files and uncommitted PTMC root-isolation helper), a tracked-source diff, and a machine-readable installation manifest. It edits only `twin_app/mode_catalogue.py`, and adds one separate evidence-status module and one permanent regression-test file. No commit/push/merge/deploy.

`--verify` runs and checks the results of:
1. Evidence-type regression and the **entire** pytest suite (includes published Cu–Al–Ni tests and the previous repairs' regressions).
2. Baseline and hostile variant-identity forensic campaigns, asserting generator and retained-variant counts, not just subprocess exit codes.
3. The seven-case near-tangency numerical falsification, requiring zero findings.
4. Both-benchmark independent solution inventory.
5. Both-benchmark independent twin/habit tensor geometry.
6. Both-benchmark classification and mode catalogue retention.
7. Fresh, source-fingerprinted JSON evidence consistency for all these audits.

The run stops on any failure. Detailed stage logs and a consolidated JSON report are recorded under `audit_findings/scientific_hardening_2026_10_22/`.

### What to send back

Download and upload:

`audit_findings/scientific_hardening_2026_10_22/hardening_report.json`

If verification stops before producing the report, send the terminal traceback and the latest `stage_NN.log` instead. Do not rerun or repair blindly.

### Safe rollback

```bash
python -u scientific_hardening_2026_10_22.py --rollback
```

Rollback restores the original mode source **only if it is still byte-for-byte the installed patch**, and does not touch the prior numeric repairs or the historical scientific audit ledgers.

## Remaining scientific constraints

Even a `PASS_CONDITIONAL` does not certify arbitrary lattice geometries, exact experimental parent/product correspondence, near-degenerate variants outside the tested domain, every published discrete CT classification, actual Streamlit visibility, physical twin occurrence, or uniqueness of all 528 directional habit records. It is a **pre-UI scientific regression and provenance gate**, not release approval.

The audit is deliberately read-only for all crystallographic algorithms. In particular, no one should interpret the 24 directional 6M branches lacking a conventional twofold witness as automatically Type I, Type II, or Compound, or equate those 24 with the 12 unoriented report constructions.
