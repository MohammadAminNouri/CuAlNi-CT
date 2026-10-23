NUMERICAL BOUNDARY SCIENTIFIC FALSIFICATION | 2026-10-21

Purpose
-------
Detect missing or merged physical martensite variants and false/missing habit
fraction roots near computational degeneracies. Also challenge Type I/II/
Compound classifications near increased metric symmetry.

Input provenance
----------------
Only the INPUT section of
  data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json
is used. Never the expected results. Its parent/product lattice constants and
rational 2H correspondence are a published-worked input; the additional
near-symmetry and tangent-critical lattice states are *synthetic probes*, not
real measurements nor externally validated materials.

Oracle methods
--------------
- Arbitrary precision (85 decimal digits) tensor eigenproblem for the full
  cubic 24-operation orbit with exact input decimal parameter perturbation.
  This can mathematically distinguish tensors separated by <1e-9 even when
  production floating-point variant deduplication merges them.
- Independent, literature-derived geometric rank-one interface reconstruction
  via the earlier upstream_twin_falsification.py oracle.
- 85-digit determinant polynomial and high-precision middle eigenvalue check
  of EXACTLY the binary64-encoded input matrices for every tangent probe.
  Production polynomial roots are NOT used as oracle inputs.
- Type I/II/Compound rechecked using proper cubic twofold generators and the
  independent interface normal/shear direction, not production label rules.

Scientific boundaries
---------------------
A 85-digit calculation on an originally binary64-encoded tensor is a precise
oracle for THAT encoded tensor, not for infinite-precision specimen geometry.
Near numerical tolerance thresholds, a difference is an UNREVIEWED finding,
not automatically a physical/implementation mistake. Accidental metric
symmetry does not prove new atomic symmetry. No experimental EBSD claim here.

Prerequisites
-------------
Run INSIDE root of CuAlNi-CT Codespace, branch feature/twin-family-tree-app-v1.
The prior upstream_twin_falsification.py must be present beside this file.
The package .[app,dev] (numpy, scipy, sympy, mpmath) must be installed.

Recommended low-compute sequence:
  unzip -o numerical_boundary_falsification_2026-10-21.zip
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  python -u numerical_boundary_falsification.py --selftest
  python -u numerical_boundary_falsification.py --phase variants

After inspecting findings, run:
  python -u numerical_boundary_falsification.py --phase tangency
  python -u numerical_boundary_falsification.py --phase classification

You may combine into --phase all but three separate stages are easier to debug.
No solver files / CI / pull requests / main branch are changed.

Outputs (persistent untracked forensic data):
  audit_findings/numerical_boundary_2026_10_21/audit.json
  audit_findings/numerical_boundary_2026_10_21/repair_ledger.md
  audit_findings/numerical_boundary_2026_10_21/repair_ledger.jsonl
  audit_findings/SCIENTIFIC_REPAIR_MASTER.jsonl
  audit_findings/REPAIR_WORKLIST.md

A priority of P1/P2 is a request to INVESTIGATE, not a confirmed bug. If a
failure occurs, preserve the input, suspected code path, certified result, and
minimal reproduction. After reproducing, patch ONLY the responsible source on
the feature branch and add a permanent regression, then CI.
