This second repair changes ONLY src/cualni_cryst/variant_identity.py.

The previous v1 patch was too strict: exact rationalization of IEEE-754 values
made trigonometric roundoff in 90-degree lattice angles appear to be genuine
crystallographic inequivalence. The new rule requires both metric-native key
agreement within an input representation bound and tensor agreement within
a separate computation bound. Distinct resolvable variants stay distinct.

From existing Codespace on feature/twin-family-tree-app-v1:

  unzip -o variant_identity_roundoff_repair_v2.zip
  python -u apply_roundoff_repair_v2.py
  python -m pytest -q test_variant_identity_roundoff_v2.py test_variant_identity_repair_2026_10_21.py
  python -m pytest -q tests/test_ball_james_adapter.py tests/twin_app/test_published_cualni_habit.py

Stop if any test fails. Do not run full suite, commit, merge, or deploy until
published regression tests are green. Any unresolved precision warning should
remain visible, not be converted into a permissive pass.

Why v1 broke legitimate cases: IEEE-754 irrational/trigonometric metric entries,
particularly cos(90 deg), are not exactly zero. The v1 exact-binary64
signature check treated their negligible differences as physical inequivalence.
In v2, metric signature tie (<32eps*condition*metric scale) AND tensor tie
(<16eps*condition*stretch scale) are required before merging, otherwise
variants stay separate or a precision warning is raised. The wider 256eps
band is an uncertainty zone, not a permission to merge. These are numerical
representation bounds, not experimental tolerances.

Emergency rollback if the new published tests remain red:
  python -u rollback_to_pre_variant_patch.py
This restores both adapters to their pre-v1 source state. Untracked audit files
are untouched. This restores previous baseline behavior (including the known
false-merging weakness), so it is a temporary safe fallback, not a scientific fix.
