# AR / Gen / Cry integration branch

This branch is based on `feature/twin-family-tree-app-v1`; it does NOT
modify `main`. The aim is one coherent **three-stage** research workflow.

1. **Cry:** phases, measured cells, point-group symmetry and metric geometry.
2. **Gen:** exact lattice correspondence, cosets, operators, groupoid tables.
3. **AR:** EBSD ingestion and, separately, parent reconstruction with an
   independently established experimental orientation relationship.

## What is presently executable

- `src/cualni_cryst/legacy_suite_bridge.py` reuses existing exact and metric engines.
- `app/pages/3_AR_Gen_Cry.py` provides a single three-stage
  Streamlit interface. Run with `streamlit run app/streamlit_app.py`.
- `tests/test_legacy_suite_bridge.py` covers exact groupoid consistency,
  metric geometry and unsafe-format rejection.

This is **not** a claim of numerical parity with the legacy 2021 GUIs.

## Full original archive, without any silent omissions

The user-provided `ARPGE_Sept2021.zip` contains 33 Python source files,
legacy phase/transition data, manuals, and obsolete Python 2.7 dependencies.
Those original bytes remain in the user's archive. They have not been
redistributed into this public repository.

To preserve everything and create *separate* automated Python-3 syntax copies,
on Python **3.11**, run:

```bash
python scripts/prepare_arpge_legacy.py ARPGE_Sept2021.zip --output third_party/arpge_suite
```

The tool copies every source and resource except old Windows Python installers.
It leaves all files intact in `original/`, transforms Python source in
`py3_syntax/`, and writes an SHA-256 manifest. For existing research data,
validate bytes and provenance; do **not** unpickle untrusted files.

Automatic `lib2to3` conversion does not fix removed NumPy APIs, obsolete
Tkinter/Canvas imports, hard-coded `C:/Python27` filesystem paths, Soya3D
dependencies, GUI import-time execution, numeric integer-division semantics,
or source-specific scientific bugs. Each algorithm must be parity-tested
against independently justified crystallographic inputs before use.

## Priority validation

- Exact correspondence vs experimental OR: **never equate the two**.
- Gen variants and operators vs existing groupoid engine.
- Cry twin Type I/II candidates vs independent metric/rank-one oracles.
- AR segmentation, parent orientation and variant labels vs known EBSD fixtures.
- 2H and modified 18R: explicitly define phase settings, lattice parameters,
  and experimental OR; the source archive does not supply a validated CuAlNi fit.

## License and provenance

The 2021 archive contains author attributions but no recognized LICENSE/COPYING
file. Public download or access to source alone does not establish permission
to redistribute modified sources. The original archive is kept separate until
terms/permission are documented. The Python bridge in this branch is newly
written and depends on the project's existing modules.
