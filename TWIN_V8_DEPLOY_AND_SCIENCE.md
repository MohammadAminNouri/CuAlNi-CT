# V8 | Scientific interface, accessibility and reproducibility specification

**Target repository:** `MohammadAminNouri/CuAlNi-CT`  
**Target branch:** `feature/twin-family-tree-app-v1`  
**Streamlit entry point:** `twin_app/streamlit_app.py`

## What this change actually supplies

1. A fully local, dependency-free Streamlit custom navigation component (`twin_app/navigator_frontend/index.html`); no chart-point selection or remote JavaScript CDN. It uses an accessible, interactive overview of **all** calculated family nodes, with an explicit control to expand **all** couples at one sibling level. Every couple uses the immutable scientific variant key from the Python report. Clicking or keyboard-selecting a couple reports that key to Streamlit, which renders one separate, stable physical-results pane. The HTML component is a navigation view, **not** a new solver.
2. A source-derived navigator model preserving every scientific family/couple and distinguishing confirmed exact A/M habit, unresolved twin type, weak route, continuum compatibility, and no exact A/M habit. The two notions **exact habit** and **verified Type I/II** remain independent.
3. A point-group explanation based on the existing exact 32-class registry and metric checks, with optional **unit-cell geometric visualization** of one selected rotation axis or mirror plane. Direct `[uvw]` vectors use `B @ direction`, reciprocal `(hkl)` normals use `B^{-T} @ covector`. This is a selected-element 3D diagram, **not** an exhaustive stereographic projection of the entire group. The explanation of the selected class remains directly under the point-group selector.
4. A single results pane: twin classification status, physical shear magnitude `s`, product-crystal twin plane `K1`, product-crystal shear direction `eta1`, then every relevant true A/M habit branch with its own `lambda`, physical `b` and unit `m`; no hidden scientific relabeling or manufactured plane. Other exact complementary solutions remain accessible. Advanced numerical identities, full frames and residuals are separate from the first view.
5. User-selectable light/dark/high-contrast **presentation modes**, comfortable/compact tree density and standard/large text; motion disabled. Text labels accompany every status colour. The node interface uses semantic buttons, `aria-pressed`, `aria-live`, and keyboard focus outlines.
6. Complete scientific JSON export, with the original input, stable SHA-256 of canonicalized input, report and scientific audit, coordinate conventions, methodology citations and solver numerical-policy context. No published expected benchmark is imported as an answer.

## The scientific engine was deliberately not rewritten

This overlay does **not** replace `twin_app/scientific_engine.py`, the scientific certifier or anything in `src/cualni_cryst/`. It therefore preserves V7's independent tensor audits, rather than risking a new classification or habit calculation while rebuilding a complex interface.

**Do not mistake a visual highlight for mathematical verification.** Exactly compatible *for the rounded entered lattice metrics* is a numerical statement, not a claim of exact physical compatibility within experimental uncertainty. The interface distinguishes an exact rank-one twin, its unverified Type-I/II classification, and the existence of a distinct A/M habit interface.

## Why no unbuilt React Flow / ELK runtime is being claimed

The target workflow is **github.dev plus Streamlit Cloud**, without Codespaces, npm, or a CI frontend build step. A package that imports `@xyflow/react` and `elkjs` but has no bundled compiled assets would fail to deploy. Therefore this release **implements analogous principles** (hierarchical layout, explicit node IDs, keyboard navigation, retained selection and single selection panel) in a locally bundled browser-native component. It does **not** falsely claim to contain the actual React Flow or ELK packages. To migrate to them later, use a reproducible Node lockfile, an asset-build CI check, vendored built assets and a fully tested Streamlit wrapper before switching the live component.

## Independent design and science references

- **IUCr**, *Symmetry*, https://www.iucr.org/what-we-do/education/pamphlets/symmetry — 32 crystallographic point groups and Hermann–Mauguin conventions.
- **IUCr**, *Teaching crystallographic and magnetic point group symmetry using three-dimensional rendered visualizations*, https://www.iucr.org/what-we-do/education/pamphlets/teaching-crystallographic-and-magnetic-point-group-symmetry-using — rotation-axis and mirror-plane teaching practice.
- **AASPIRE Web Accessibility Guidelines** (Raymaker et al., 2019), https://pmc.ncbi.nlm.nih.gov/articles/PMC6485264/ — predictability, explicit labels, font/theme choices, reduced clutter, minimal scrolling. No interface can be declared *universally* autism-accessible without user testing.
- **W3C WCAG 2.2**, https://www.w3.org/TR/WCAG22/ — keyboard, focus, contrast, input labelling; formal accessibility conformance requires independent testing, not a static source audit.
- **React Flow** accessibility guide, https://reactflow.dev/learn/advanced-use/accessibility — node-keyboard interaction reference.
- **Streamlit custom components** documentation, https://docs.streamlit.io/develop/api-reference/custom-components — local custom component deployment.
- **K. Bhattacharya**, *Microstructure of Martensite*, Tables 5.1, 7.2, 7.3 — conventional twin and A/M interface notation; benchmark values remain in tests, never injected as runtime outputs.
- **Ball and James** nonlinear elasticity and **Cayron** correspondence theory — independent scientific methods handled by the existing backend, not solved in frontend JavaScript.

## Validation before a research-grade claim

- Local compilation and the standalone twin-app suite verify UI/data contracts and synthetic numerical properties.
- Chromium/Playwright browser checks verify actual rendered component count, node selection and keyboard focus/roles in a demo model.
- Full GitHub CI must pass Python 3.11, 3.12 and 3.13 with the actual backend installed, including published NiTi and Cu–Al–Ni cases.
- The deployed Streamlit Cloud app must be checked for component callbacks (the iframe-to-Python Streamlit bridge), contrast, browser zoom, screen-reader navigation and a no-habit branch.
- User research with autistic and non-autistic crystallographers is required to validate cognitive accessibility. This release is an **improvement designed from the guidance**, not a medical or accessibility certification.

## github.dev deployment

Extract the ZIP on your computer; **do not drag the ZIP or all flattened files into repository root**. Only copy the listed existing/new paths to their exact directory locations. In particular the browser component must be at:

`CuAlNi-CT/twin_app/navigator_frontend/index.html`

Do not delete `app/`, `src/`, `data/benchmarks/`, or existing test files. The overlay is additive outside the explicitly replaced UI/tests/manifest files.
