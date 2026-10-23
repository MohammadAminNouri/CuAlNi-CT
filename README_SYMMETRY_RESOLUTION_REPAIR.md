# CuAlNi-CT: Near-degenerate symmetry classification resolution guard

**Status:** PROVISIONAL repair candidate; not an independently certified physical twin classification.

**Why:** With a synthetic orthorhombic perturbation epsilon=1e-9, the diagnostic recorded six distinct stretch tensors separated by only 8.444381648e-10 relative, while the symmetry completion was attempting to transport Type I/II classifications at a fixed 1e-7 tolerance. Its opposite-type witness dyads differ by ~9.4927e-9 relative, yet were accepted as the same interface. A diagnostic rerun at min gap / 8 reported 20 direct + 4 transported branches without classification conflict.

**Remedy:** In `twin_app/symmetry_completion.py` and `twin_app/mode_catalogue.py`, cap the tensor-matching tolerance by one eighth of the minimum **relative separation between distinct recorded stretches**, and fail closed if the separation is too small for binary64. This is a numerical uniqueness requirement (less than half the pairwise separation), NOT a physical tolerance or a change in Type I/II equations. It retains direct cross-lock anchors and raises when truly identical opposite-type constructions conflict. The mode catalogue also replaces its hardcoded closest-match distance-difference `1e-9` rule with a uniqueness margin relative to the actual orbit separation.

**Potential limitations:** This check addresses the demonstrated ambiguity; it is not a complete global certificate of all possible near-coincident rank-one dyad branches and cannot validate source correspondence. Future separate tests should check relative dyad separation, alternate parent operations, duplicate correspondence mappings, and convergence under high precision.

**Install:** Upload the ZIP to the root of `CuAlNi-CT` Codespace on `feature/twin-family-tree-app-v1` and run:

```bash
unzip -o symmetry_resolution_repair_2026_10_21.zip
python -u apply_symmetry_resolution_guard.py
python -m pytest -q test_symmetry_resolution_guard_2026_10_21.py
python -m pytest -q tests/twin_app/test_symmetry_completion_2026.py tests/test_ball_james_adapter.py tests/twin_app/test_published_cualni_habit.py
```

**Stop** and report the first exception if red. If green, run `python -m pytest -q` and rerun `python -u diagnose_symmetry_conflict.py`; it should report `APPLICATION_COMPLETED`, rather than only successful isolated classification. No push/merge/deploy yet.

**Backup:** The installer backs up both modules in `audit_findings/` using `_before_resolution_guard_2026_10_21.py` filenames. To undo after reviewing local changes: `cp audit_findings/symmetry_completion_before_resolution_guard_2026_10_21.py twin_app/symmetry_completion.py`.
