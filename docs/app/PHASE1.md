# Phase 1 — public crystallography workbench

## Boundary

The application layer does not implement crystallographic equations.

```text
Streamlit widgets
      ↓
app.application
      ↓
cualni_cryst.project_io / ProjectState
      ↓
cualni_cryst.CalculationService
      ↓
verified scientific modules
```

The frozen scientific reference remains `freeze/scientific-backend-v3.1-2026-09-24`.
Application development belongs on `app/ui-integration-v1`.

## Normal-user workflow

1. Enter parent/product phase IDs and lattice parameters.
2. Select conventional crystallographic point groups.
3. Enter the exact 3×3 correspondence matrix `C(M←A)`; rational entries such as `1/2` are accepted.
4. Calculate.
5. Inspect compatibility, CMC/SMC, stretch spectrum, CT habit planes, Ball–James solutions, topology, variants, and diagnostics.
6. Export the exact project JSON and the JSON-safe result bundle.

No Python, JSON editing, or coding is required for the interactive path.

## Scientific presentation rules

- Exact CT compatibility is never conflated with the nearest-degeneracy diagnostic.
- The UI does not widen numerical tolerances.
- Fractions are passed as exact text into the existing project loader.
- Invalid point-group/cell combinations, singular correspondence matrices, and invalid metrics fail explicitly.
- Diagnostic NaN/Infinity values are serialized as JSON `null`; this is a presentation conversion only.

## Local run

```bash
python -m pip install -e ".[dev,app]"
streamlit run app/streamlit_app.py
```

## Streamlit Community Cloud

Deploy the repository branch `app/ui-integration-v1` with entry point:

```text
app/streamlit_app.py
```

The root `requirements.txt` installs the package and the `app` optional dependency group.
