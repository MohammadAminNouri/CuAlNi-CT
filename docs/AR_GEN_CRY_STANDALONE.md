# AR · Gen · Cry — separate Streamlit application

**Main Streamlit file:** `ar_gen_cry_app.py` (repository root).

This is an independent application. It must **not** be deployed with
`app/streamlit_app.py`, which remains the separate twin-feature workstation.
The AR/Gen/Cry app imports the shared scientific library only and never imports
twin Streamlit pages or original Python-2 GUI modules.

## Workflow (one screen at a time)

1. **Cry:** parent/product phase metrics; exact symmetry compatibility;
   direct-space length and reciprocal d-spacing. Uses explicit ångströms and degrees.
2. **Gen:** enter `C_(M←A)` as nine integer/rational entries. Calculate exact
   symmetry intersections, cosets, operators, adjacency and Burnside checks.
3. **AR:** optional EBSD `.ang` or `.ctf` upload. Report indexed counts and
   orientation convention. A separate *experimental* OR, supplied in Cartesian
   coordinates as a JSON array, is required for an optional exploratory
   single-parent fit. **No experimental OR is guessed from correspondence**.
4. **Report:** export the actual calculated stages as JSON, missing ones as null.

The default lattice metrics and correspondence are **synthetic demos**.
They are not a measured Cu–Al–Ni 2H/18R phase assignment.

## Run locally

```bash
python -m pip install -e ".[app,dev]"
streamlit run ar_gen_cry_app.py
```

## Deploy as a NEW Streamlit Community Cloud app

1. Create **New app** in Streamlit Community Cloud.
2. Repository: `MohammadAminNouri/CuAlNi-CT`.
3. Branch: `feature/arpge-genova-crystals-python3-integration`.
4. Main file path: **`ar_gen_cry_app.py`**.
5. Deploy separately from the existing twin-feature app.

The root `requirements.txt` installs `.[app]`, as before.
This independent entry point has no implicit multipage discovery of
`app/pages/`; those files belong only to the old `app/streamlit_app.py`.

## Tests

```bash
python -m pytest -q tests/test_ar_gen_cry_pipeline.py \
  tests/test_ar_gen_cry_app.py tests/test_legacy_suite_bridge.py
python -m py_compile ar_gen_cry_app.py src/cualni_cryst/ar_gen_cry_pipeline.py
```

CI: `.github/workflows/ar-gen-cry-standalone.yml` separately exercises
Python 3.11, 3.12 and 3.13, the scientific/unit/negative tests, Streamlit
AppTest and actual HTTP health of the new standalone entry point.

## Scope and caution

- Exact operator groupoids are *theoretical* correspondence outputs, not
  experimental OR identification.
- AR's optional fit is a bounded, exploratory single-parent estimate on
  sampled pixels, not spatial grain segmentation or full parent-map reconstruction.
- The 2021 original sources and their syntax conversions remain under
  `third_party/`, but are **not executed** by this application.
- Full original-source parity, complete legacy dependencies, Cu–Al–Ni 2H/18R
  literature reproduction, indexing and uncertainty validation remain separate
  research tasks. CI green does not certify them.
