# Independent classification and mode-retention audit (22 October 2026)

**Read-only.** No installer, no application patches, no commits or deploys.

## Scientific scope

Reconstructs the published-input **Cu–Al–Ni 2H** and **James–Hane 6M** states from their public application inputs. Computes the scientific-family report, its mode catalogue, and the raw PTMC relations using the existing source tree. It then checks the following **independent invariants**, without using a material-specific table of answers:

1. **Completeness across topology and modes.** Each `(family, correspondence pair, construction ID, twin branch)` from `TwinFamilyReport` occurs **exactly once** in `ModeCatalogue`. The full dataclass content, habit fractions, plane normals, and shape vectors must be unchanged; fabricated, duplicated, altered, and lost entries are errors. Every advertised `mode.types` and `mode.by_type` subset must agree with its underlying entries. Unresolved constructions and their habits must remain visible as *unresolved*, not be relabeled.
2. **No missing compatible stretch-pair coverage.** Every PTMC-compatible *unordered* stretch pair must be represented by at least one correspondence-pair construction if all stretch variants have a correspondence representative. Direction-reversed ordered PTMC branches **are not** double-counted as extra modes. A missing correspondence mapping produces `REVIEW_REQUIRED`, not an invented correspondence.
3. **Type I/II/Compound provenance.** A classical label must have documented direct CT cross-lock or *properly marked* parent-symmetry-transport evidence. Each **direct** claim must refer to real discrete classical system IDs with consistent physical shear. Compound requires both `I` and `II` representations. CT source generators are checked for parent-group membership, positive shear, and `(plane)·[direction]=0` in direct/reciprocal coordinates.
4. **Independent necessary geometric screening.** The audit reconstructs symmetry-axis checks using the physical rank-one dyad and all parent proper twofolds: geometric Type I/II/Compound candidates are **witnesses only**, not independent certificates of discrete CT theory. Missing or extra geometric witnesses are recorded as review items where appropriate, **never used to force a label**. The 24 previously reported 6M no-twofold-witness branches must not be guessed into Type I/II/Compound.
5. **Reproducibility.** Record `git HEAD` **and SHA-256 hashes of the actual working source files**, since earlier numerical repairs are uncommitted. All findings keep exact case and construction IDs.

## Run

Unzip the package **into the existing** `CuAlNi-CT` Codespace repository root:

```bash
unzip -o classification_retention_audit_2026_10_22.zip
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u classification_retention_audit_2026_10_22.py --selftest
python -m pytest -q test_classification_retention_audit_2026_10_22.py
python -u classification_retention_audit_2026_10_22.py --case both
```

Output: `audit_findings/classification_retention_2026_10_22.json`. Upload **this one file** to ChatGPT, even if it reports an error or review requirement. Exit code 0: no errors/reviews; 3: review required; 2: discrepancy or execution error. Do **not** change tolerances or code to turn a status green.

## Limits that matter

- The test checks report → mode catalogue retention, **not actual Streamlit browser rendering, selection, pagination, or responsive layout**. That requires a later interactive UI audit.
- A CT generator's presence in the parent group and geometric axis alignment are *necessary* constraints, not a complete independent reproduction of Correspondence Theory. Even a `PASS_CONDITIONAL` is **not an independent proof of every classical subtype**.
- Raw PTMC ordered branches and correspondence-topology mode records have different equivalence conventions; comparing their raw counts as if they were identical generates false alarms.
- The known 2H/6M input geometries are literature-based encoded test inputs, not measured EBSD parent/martensite correspondence or experimental validation of physical twin formation.
- `REVIEW_REQUIRED` can be scientifically correct. In particular, absent a complete independent discrete proof, **keep a rank-one solution labeled unresolved** and preserve its shear, normal, fraction, and habit solutions.
- The guard makes no statement about arbitrary lattice ratios, near degeneracies, numerical interval certification, the experimental frame, variant selection, or observed twins.

Do not merge/deploy until scientific and UI retention issues have been addressed and final CI has been run on reviewed files.
