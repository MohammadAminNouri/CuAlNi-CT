# GitHub Dev installation — one commit

Use branch:

```text
feature/twin-family-tree-app-v1
```

The final ZIP already includes Batch 1 + Batch 2 + final hardening. **Do not install the earlier ZIPs separately.**

## Add these paths at repository root

```text
twin_app/
tests/twin_app/
data/benchmarks/bhattacharya_niti_twin_habit_v1.json
data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json
data/benchmarks/niti_higher_order_weak_v1.json
README_TWIN_APP.md
TWIN_APP_SCIENTIFIC_CONTRACT.md
TWIN_APP_GITHUB_DEV_INSTALL.md
```

Do not replace anything under:

```text
app/
src/cualni_cryst/
```

## In github.dev

1. Confirm the branch selector says `feature/twin-family-tree-app-v1`.
2. Add the files/folders above to the repository root.
3. Check Source Control. Only the new standalone-app/test/benchmark/documentation paths should be listed.
4. Use one commit:

```text
Build standalone twin-family tree and habit-plane workbench
```

5. Sync/Push the branch.

## First checks after the files are present

If a terminal is available:

```bash
pip install -e ".[app]"
pytest tests/twin_app -q
streamlit run twin_app/streamlit_app.py
```

The standalone app must launch independently. The existing app remains available exactly as before.

## Merge rule

Do not merge the feature branch into the working app branch until the new tests and the repository's existing scientific tests are green. A numerical benchmark mismatch should be investigated; do not weaken the benchmark merely to get a pass.
