# CuAlNi-CT — 23 October 2026: complete pending-file scientific curation

## Crucial preservation guarantee

- **Every original**: 168/168 pending files verified byte-for-byte against snapshot inventory SHA-256.
- Git bundle includes the local **de88380** scientific-source commit; its checksum also matches the inventory.
- **No source files were deleted, renamed, overwritten or moved in Codespaces by this review.**
- Full user backup: `CUALNI_PENDING_20261023T122637Z_BACKUP.zip` (retain OFF Codespaces on user's own computer).
- The present curation verifies every original and studies all primary independent auditors and results; archive-only installer/log content is classified from its function and provenance, not claimed to have undergone line-by-line scientific validation. The CSV includes one row for each original with exact hash, original path, preservation status, and a disposition rationale.
- No mandatory full pytest reruns. Original test-suite crash and incomplete end-to-end EBSD verification are recorded as open evidence gaps.

## IMPORTANT: previous project README accidentally overwritten

The uncommitted root `README.md` (SHA-256 `ffcf14ac25bb914f8ee3df08b30a5c1ee4faf60403d3e0db9d70e91e0524b5c6`) contains only the boundary-classification investigation notes, **not the project introduction**. Git commit `de88380` still has the proper main README. Preserve the audit note under `docs/validation/2026-10-23/BOUNDARY_CLASSIFICATION_INVESTIGATION.md`. Do not commit that changed root README as-is. Never use a blind reset; the explicit README recovery must verify SHA and have an independent archive.

## Disposition: 168 originals, no omissions

| Classification | Count |
|---|---:|
| `ARCHIVE_BACKUP_OR_TRANSIENT_LOG` | 46 |
| `ARCHIVE_IDENTICAL_TO_COMMITTED` | 17 |
| `ARCHIVE_INSTALLER_OR_ROLLBACK` | 9 |
| `ARCHIVE_INSTALLER_OR_WORKING_COPY` | 8 |
| `ARCHIVE_OTHER` | 2 |
| `ARCHIVE_PACKAGE_OR_LOG` | 27 |
| `ARCHIVE_REVIEW_HISTORICAL_TEST` | 1 |
| `ARCHIVE_SECONDARY_PROVENANCE` | 8 |
| `ARCHIVE_SUPPLEMENTARY_DOCUMENT` | 9 |
| `CURATED_COMMIT_AUDIT_SOURCE` | 9 |
| `CURATED_COMMIT_AUDIT_TEST` | 4 |
| `CURATED_COMMIT_EVIDENCE` | 17 |
| `CURATED_COMMIT_METHOD_NOTES` | 10 |
| `RELOCATE_AUDIT_README_KEEP_PROJECT_README` | 1 |
| **TOTAL** | **168** |

### Curated Git candidates

- 9 independent audit scripts: mathematically distinct adversarial, inventory, geometry, variant-identity and classification audits.
- 4 unique auditor self-test files: added as **auditor tests**, not incorrectly represented as experiments or current CI certification.
- 17 selected scientific results/provenance records: original JSON/JSONL and relevant numeric witness records.
- 10 original methodology/repair notes: keep historical status (e.g. *provisional*) and date; do not rewrite old claims to suggest formal certification.
- **40 original files selected** for an additional research-evidence Git commit; plus 3 new documentation/index files prepared by the curation tool. All other originals are preserved in the independent archive and indexed in Git's ledger.

### Scientific results and limits (reviewed from uploaded JSON)

- `geometry_independent_2026_10_22.json`: 2H 192/192; 6M 336/336 habit solutions independently matched; reported mismatches and inconclusive counts both zero for these finite benchmark configurations.
- `independent_inventory_2026_10_22.json`: 85-digit oracle checks, mismatches=0/inconclusive=0 for these two cases; explicitly **PASS_CONDITIONAL**, not proof of global completeness.
- `classification_retention_2026_10_22.json`: mode/report pair retention 2H and 6M, mismatch=0 and reviews=0; explicitly **not a full correspondence-theory classification proof**.
- `upstream_2026_10_21/all/audit.json`: `PASS_CONDITIONAL` across its encoded test cases with correspondence assumptions.
- `variant_identity_2026_10_21/baseline.json` and `hostile.json`: original before/after forensic runs, not live experimental results.
- `or_cache_ab_2026_10_23.json`: 42 recorded A/B checks, bit-exact reported matching; synthetic orientation probes, not measured-map certification.
- MATLAB/simulator proof of experimental **18R** twin modes and 3D habit planes **not established** by this archive. The 6M benchmark cannot silently certify an 18R indexed specimen.

### Local tests actually performed by this review

- 168/168 SHA-256 exact matches and ZIP structure checks; bundled Git history retrieved successfully.
- Syntax parsed for all snapshot Python files (no syntax errors).
- Auditor self-test scripts: **18 tests passed** against a clone of the archived local commit, not against an experimental specimen.
- No live Codespaces files or Git references changed during the review.

### Privacy and reproducibility

- The source repository is **public**. The full backup contains intermediate logs, historic source copies and local-path details. **Do not add or push the complete backup ZIP to GitHub.**
- A narrow scan found no obvious API-key/private-key signature in text files, but this is **not a comprehensive secret/PII clearance**. A final human check is necessary before pushing research records publicly.
- The lone non-identical legacy `test_symmetry_resolution_guard_2026_10_21.py` was compared with its committed replacement. The difference is only the benchmark-fixture path and an explicit skip when the fixture is unavailable; its scientific assertions are already retained in the committed test. The older exact bytes remain in the backup.
- Some auditors write output files when executed and others rely on legacy backup sources. The committed scientific record should identify those dependencies, not claim every script is immediately rerunnable standalone.

### Suggested next actions (same existing feature branch)

1. Download and separately retain the user's **full original backup ZIP**.
2. Use `reviewed_evidence_stager.py --verify` in Codespaces (no changes).
3. Inspect reviewed staged file set with `--stage` only if the verified current-state hashes and HEAD match. This stages existing original files, creates 3 explicit documentation files, and does **not** commit/push/reset/overwrite the project README.
4. Confirm the published evidence contains no material that should remain private before committing/pushing to the SAME feature branch.
5. Recover the main root README separately, from commit `de88380`, after archiving the current audit README copy (stager already does so in `docs/validation/2026-10-23/`).

### Why not stage every ZIP, installer and diagnostic log?

All those bytes are already indexed with SHA-256 and preserved inside the recoverable backup. Publishing a second copy of every outdated installer and binary backup in a public Git repository would increase risk and confusion, not improve scientific reproducibility. This policy preserves **all work** without silently activating historic patches or making contradictory scripts part of the production runtime.
