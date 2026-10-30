# ARPGE / GenOVa / Crystals 2021: independent modern scientific reconstruction

**Branch:** `feature/arpge-genova-crystals-python3-integration`; PR #9 draft.  
**Original:** user-supplied `ARPGE_Sept2021(2).zip` SHA-256 `7b1df4abf1ceac278e64d4cf2ab6dda4e5ab8efab3c80fbed4707be9b789e873`.  
**Scope:** independent replacement without executable Python-2/pickle imports or verbatim republication of unlicensed original source.

## New executable components

- `arpge_archive.py`: safe protocol-0 transition-list decoder (only primitive opcodes); static lattice-metric inspection of NumPy-bearing CRYST pickle records without unpickling; checks every archived plane/axis relationship with direct and reciprocal lattice geometry. **Important legacy convention:** transition index arrays are DAUGHTER-first, PARENT-second although phase names are PARENT, DAUGHTER. The source NiTi AQ and Burgers cases prove this direction.
- `arpge_spatial.py`: full-scan Oxford H5OINA adapter, indexed pixel topology, crystal-symmetry-aware adjacent-pixel union-find daughter-grain regions, per-grain mean orientations and actual contact-weighted adjacency. Candidate parent orientation is obtained from an explicit supplied Cartesian OR and proper daughter/parent symmetry. Every spatial triplet/quadruplet subset is considered under an explicit combinatorial cap, with no arbitrary first-neighbour shortcut. Ambiguous seeds are tracked and not silently accepted. Separate parent-domain graph growth and per-grain variant labels/residuals are exported.
- `arpge_analytics.py`: source-pixel neighbour KAM, GOS against grain mean, cubic Σ3-labelled adjacency and shortest graph-chain domain depth (NOT verified composed Σ3^n CSL order), 2D apparent box-count slope (NOT old LLC/fractal parity), upper-hemisphere stereographic direct poles, reciprocal-plane specimen trace and strictly described **research** .ctf/.ang exports. Export requires a complete square grid and explicit phase lattice. Vendor-native equivalence and 3D diffraction intensity are not claimed.
- `arpge_workflows.py`: explicit deterministic tolerance cycles, source-preserving indexed-hole morphology and contact-weighted operator residual histograms.
- `ar_gen_cry_app.py`: separate user-facing spatial workflow and original ZIP inspector; old twin-feature Streamlit entry point remains unchanged.
- `scripts/audit_arpge_archive.py`: run the 2021 ZIP checksum and static phase/transformation checks locally; no unsafe archive extraction.

## Archive audit, 30 October 2026

57 original *.trans files and 61 original *.cryst metric prefixes were independently decoded without executing source or pickle. All 57 transition records and all 61 crystallographic metric headers are structurally readable. Of the 57 geometric OR candidates **51** have consistent, nonparallel constraints within 2°; **six** refer to unavailable source phase files (Omega, five zirconia transformations). No 1:1 GenOVa operators/legacy object numerical parity is yet established. Original atomic sites/phase CIF are **not** recovered from the static pickle prefix; do not invent them.

## Actual archived Ti–6Al–4V diagnostic

`Ti-6Al-V-AdditiveManuf.ang` contains 263,626 pixels, indexed α phase 245,632 and indexed β phase 17,994. Independently applied archive Burgers OR to a fixed random sample of 1,500 α pixels within 4 pixels of nearest measured retained β. Compare candidates against beta orientation modulo 24 proper cubic and 12 proper hexagonal operators. Median closest angular residual ≈14.06°, 20.8% within 2°. A 15° deliberately perturbed OR has median ≈14.97°, only 0.067% within 2°. At nearest 1-step distance median ≈6.04° for 158 observations.

**Interpretation:** The source is useful for retained-beta validation, but a nearest beta pixel is not a verified same prior-parent grain, absolute frame calibration is unverified, and residuals remain high. **THIS IS NOT A SUCCESSFUL PHYSICAL OR OR MULTI-PARENT PARENT-MAP VALIDATION.** Do not tune OR against these pixels and call it independent testing.

## Missing before complete ARPGE parity

1. Reproduce all legacy GenOVa coset/double-coset operator tables from signed source presets and independent exact groupoid audit, including six presets missing archive phase definitions.
2. Original-specific triplet/quadruplet parent-nucleation/graph-growth numerical parity against controlled original Python 2 execution; our method is independent and deliberately refuses ambiguous first-choice shortcuts. Search caps are explicit; a cap reached is NOT exhaustive.
3. Quantitatively held-out retained β-parent grain segmentation on original Ti64 and bainitic-steel maps. Compare same-domain matched beta rather than nearest generic beta pixel. Original phase identification and sample-frame calibration required.
4. Native hex ANG, all vendor metadata variants, non-square/open boundary maps, packet/Pitsch/KS/NW analysis, crystallographic 3D interfaces and electron-diffraction structure factors/intensities need independent validated implementations.
5. Original Σ3^n composition, LLC and legacy fractal statistic exact parity; current graph-depth and apparent 2D scaling functions are expressly **not** equivalent.
6. Cu–Al–Ni true 18R indexed atomic CIF/basis and independently measured retained/reverted β orientation are absent. No algorithm can certify a physical parent without them.
7. Source-license permission before distributing old Python-2 code verbatim.

## Tests and execution

Run `python -m pytest -q tests/test_arpge_spatial.py tests/test_arpge_analytics.py tests/test_arpge_workflows.py tests/test_arpge_archive.py` in the feature branch with `.[app,dev]` installed. Dedicated standalone workflow additionally tests on Python 3.11, 3.12 and 3.13 and smoke-tests Streamlit. Broad other workflows still have a stale twin-manifest preflight; do not treat that as numerical test failure or hide it.

Offline original archive audit:

```bash
python scripts/audit_arpge_archive.py "/path/to/ARPGE_Sept2021(2).zip" --output original_archive_audit.json
```

The original ZIP and actual original EBSD scans are **not committed** publicly. The user-provided ZIP must remain local or be uploaded through the app.
