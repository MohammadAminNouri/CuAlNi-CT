INDEPENDENT UPSTREAM TWIN + HABIT COMPLETENESS AUDIT (2026-10-21)
===================================================================

Purpose: find wrong or missing crystallographic predictions, not merely
reproduce existing green tests. Does NOT edit scientific source files.

Inputs:
- 2h_reference: Bhattacharya's Cu-Al-Ni orthorhombic reference parameters.
- 2h_landa: Landa 2007 reported 2H martensite metrics already preserved in
  data/reference/cualni_literature_states.json, using the standard 2H
  correspondence as an explicit assumption, NOT a measured specimen C.
- niti: Bhattacharya NiTi B2/B19prime reference lattice/point-group/C.
All required input states are loaded from the current repository's
'input' or 'states' sections. NO 'expected' block, published habit fraction,
shear, precomputed twin normal, or classification is fed to the oracle.

What is independently computed:
1. Parent/product metric tensors from cell lengths and angles.
2. Physical stretch U via orthonormal metric pullback and eigen-decomposition.
3. All cubic parent rotations from signed permutation matrices.
4. The full independent stretch orbit and all unordered variant pairs.
5. Relative compatibility using eigenvalues of (Uj Ui^-1)^T(Uj Ui^-1).
6. BOTH twin dyads using geometric null-cone planes, not the production
   two-branch closed-form formula.
7. Twin types by checking physical planes/shear against matching parent
   twofold-generator axes; Type I/II/Compound are not supplied as targets.
8. Laminate fraction roots from the middle singular value using bracketing
   plus special near-tangency/flatness searches, not production's quadratic.
9. BOTH habit shape dyads from an independent null-cone geometric solve.
10. Compares all results against the actual build_twin_family_report() output.

Honest limitations:
* The mathematical theories are shared. An independent implementation is not
  independent experimental verification of a specimen.
* True specimen correspondence, chemistry, temperature, and morphology are
  NOT established by lattice lengths alone.
* No bounded numerical fraction scan proves global completeness for arbitrarily
  close roots; near-flat and near-degenerate cases go to the review ledger.
* Reproducible P0/P1 issues must be triaged. A finding can indicate an oracle
  or interface-mapping defect, not necessarily a production-code defect.
* This verifies only cubic-parent branch variants. For other parent symmetries,
  add a separately validated symmetry-group oracle first.

CODESPACES (already installed Python packages)
----------------------------------------------
1. Drag upstream_twin_falsification_2026-10-21.zip into the repo root.
2. In terminal:
   unzip -o upstream_twin_falsification_2026-10-21.zip
   export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   python -u upstream_twin_falsification.py --selftest
   python -u upstream_twin_falsification.py --case 2h_reference
3. Read the status before running further cases:
   python -u upstream_twin_falsification.py --case 2h_landa
   python -u upstream_twin_falsification.py --case niti
4. To run the 3 together (optional):
   python -u upstream_twin_falsification.py --case all

OUTPUT PERSISTENCE (safe across separate runs)
----------------------------------------------
Each case writes separately under:
   audit_findings/upstream_2026_10_21/2h_reference/audit.json
   audit_findings/upstream_2026_10_21/2h_reference/repair_ledger.md
   audit_findings/upstream_2026_10_21/2h_landa/...
   audit_findings/upstream_2026_10_21/niti/...
And every run updates an append-only (deduplicated) master:
   audit_findings/upstream_2026_10_21/findings_history.jsonl
   audit_findings/upstream_2026_10_21/runs_history.jsonl

Do not commit, run repair scripts or change solver tolerances yet. For each
P0/P1/P2 finding, preserve exact input and context, reproduce it, verify the
independent oracle, identify the actual wrong formula/setting, THEN make
minimal changes in an isolated fix with regression tests. Re-run every
published benchmark and both oracles before accepting a correction.

P0: wrong/missing mathematical branch, false positive or invalid habit.
P1: branch mathematically valid but crystallographic label differs.
P2: not certifiable at double precision or ambiguous geometry.

The exit code 1 denotes one or more P0 review findings, 2 denotes P1/P2
findings needing further analysis, and 0 means PASS_CONDITIONAL only.

Sources/provenance already included within repository:
- data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json
- data/benchmarks/bhattacharya_niti_twin_habit_v1.json
- data/reference/cualni_literature_states.json
- docs/SOURCES.md; docs/VERIFICATION_MATRIX.md
- Ball & James (1987) and Bhattacharya, Microstructure of Martensite
- Landa et al., Mater Sci Eng A 462 (2007), 320-324,
  DOI 10.1016/j.msea.2006.02.472 (published 2H elastic-context source)
