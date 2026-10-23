# PTMC near-tangency root repair — 22 October 2026

## Scope and provenance

This is a **local, uncommitted repair candidate** for the existing
`feature/twin-family-tree-app-v1` branch of `MohammadAminNouri/CuAlNi-CT`.
It changes only `src/cualni_cryst/ptmc_adapter.py` and adds the independent
`src/cualni_cryst/ptmc_root_isolation.py` module and a root-level pytest file.
The current variant-identity, generator-provenance, symmetry-completion,
mode-permutation, and historical literature benchmark code are not changed.

The source of the defect is the pre-existing quadratic root utility:

1. It floors the discriminant scale at `1.0` although the determinant
   polynomial is scaled around `1e-3` in the recorded tangency case.
2. It uses a broad tolerance to clamp genuinely negative discriminants to 0.
3. It merges nearby positive-discriminant roots before establishing whether
   the roots are actually identical.

The audit JSON received on 22 October 2026 records three false-positive
fractions (epsilon `0`, `+1e-10`, `+1e-8`) in a **synthetically rescaled**
Cu–Al–Ni 2H-based metric. This is numerical falsification, not a claim
about measured specimens.

## Repair design

* **Fast path:** quadratic roots with a strictly negative discriminant are
  rejected; positive and zero discriminants are not rounded into one another.
* **Near-critical bounded path:** for finite-domain PTMC problems, when the
  binary64 discriminant is close to zero compared with its own natural scale,
  the solver recomputes a degree-≤6 determinant polynomial from the **original
  encoded binary64 deformation matrices**, using mpmath at 100 decimal digits.
  Higher-degree terms introduced by binary64 representation of numerically
  rank-one dyads are retained, not assumed away.
* **Independent isolation:** monotonic subintervals are derived recursively
  from polynomial derivative roots. The solver detects genuine critical-point
  roots as well as sign-crossing roots. It separately checks the **middle**
  singular value, not merely det(FᵀF-I).
* **Fail closed:** indistinguishable critical-point cases, a continuum needing
  independent proof, multiple roots rounding to one float, or more than two
  habit fractions for a numerically rank-one dyad raise `UnresolvedPTMCRoot`.
* **No changes** to the PTMC equations or to published expected outputs.

The 100-digit search is a **high-precision numerical procedure, not a formal
interval-arithmetic proof**. The numerical values are only statements about
the exact binary64-encoded tensors, not about measured lattice certainty.
The original `continuum` guard remains in place and is not reclassified by
this patch.

## Install in Codespaces

Upload this ZIP to the `CuAlNi-CT` repository root on the existing feature
branch; do not overwrite other uploaded scripts or use Copilot to change files.

```bash
unzip -o ptmc_tangency_repair_2026_10_22.zip
python -u apply_ptmc_tangency_repair.py
python -m pytest -q test_ptmc_tangency_repair_2026_10_22.py
```

The installer verifies the branch, source structure, and old failure pattern,
and backs up the original adapter to:

`audit_findings/ptmc_tangency_repair_2026_10_22/original_ptmc_adapter.py`

**Stop** and send the exact traceback if anything fails; do not lower
validation thresholds or force a green result.

## Evidence required BEFORE commit or merge

First, rerun the **original seven synthetic tangency cases** (not just the new
regressions):

```bash
python -u numerical_boundary_falsification.py --phase tangency
```

Expected: same two compatible-side root pairs as independent
oracle and zero false-positive fractions. Its JSON is written to:

`audit_findings/numerical_boundary_2026_10_21/audit.json`

Then run the unaffected scientific benchmarks and full test suite:

```bash
python -m pytest -q tests/test_ball_james_adapter.py tests/twin_app/test_published_cualni_habit.py tests/twin_app/test_cualni_blind_roots_2026.py
python -m pytest -q
```

If these pass, only then can the false-positive habit-root **regression** be
marked resolved. Global PTMC twin-branch and habit-plane completeness is not
proven by these tests.

## Rollback

```bash
python -u rollback_ptmc_tangency_repair.py
```

This restores only the adapter from the local backup and refuses to overwrite
files if the repair marker is absent. It does **not** undo unrelated changes.
The new helper/test files remain visible for inspection.

No branch creation, commit, push, CI dispatch, or deployment is performed.
