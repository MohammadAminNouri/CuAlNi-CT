# P1 scientific closure audit — 15 October 2026

**Scope:** follow-up to P1 checkpoint `bee403e76d275454131a00e9b8a1c2b022502f0c`.  The P0 and P1 checkpoint branches remain unchanged. This audit branch intentionally distinguishes repository tests from scientific and experimental certification.

## Verified status

- P1 GitHub Actions workflow `38590714322`: all Python 3.11, 3.12 and 3.13 jobs passed.
- Existing `.github/workflows/tests.yml` runs pytest and the symbolic command, **not** `tools/run_p0_scientific_acceptance.py` or `tools/run_p1_scientific_acceptance.py`.
- This branch adds a distinct full-pipeline acceptance workflow, including a **strict inconclusive-root gate**. A red gate caused by unresolved root cases is an honest scientific blocker, not permission to weaken the check.

## Work requiring closure (priority order)

1. **P0 scientific acceptance completeness.** `tools/run_p1_scientific_acceptance.py` currently records `NEEDS_BOUNDARY_INVESTIGATION` and `INCONCLUSIVE_CONTINUUM_OR_NEAR_ZERO_POLYNOMIAL` without automatically returning a failure. Strict gate must treat these as unresolved and disclose case/branch provenance.
2. **P0 Cu–Al–Ni app integration.** `check_cualni_specimen` is used by the specimen CLI, not as a mandatory structured validation step in the main interactive input path. The app must distinguish a physically identified phase/setting from a mathematical custom-lattice calculation, without falsely denying valid generic crystallography.
3. **P0 crystallographic correspondence evidence.** Published 2H, modified 18R and reduced 6M cells describe different samples/settings; physical prediction requires sample-matched parent and daughter metrics, rational correspondence, atomic/stacking evidence and independently measured OR. No such complete source-specific data are in P1. The **reference-only** command must never produce predictions.
4. **P1 covariance correctness.** `p1_ensemble.propagate_metric_uncertainty` uses `np.allclose(cov,cov.T,atol=1e-12)` with NumPy's nonzero *default relative tolerance*. This can admit asymmetric covariances at large magnitude. Validate symmetry with a physically documented absolute/relative bound and return explicit conditioning diagnostics; do not manufacture positive definiteness.
5. **P1 orbit completeness and classification.** `p1_evidence._orbit_labels` uses tolerance-based transitive union. Verify pairwise symmetry witnesses for every group member and avoid merging physically distinct tensors through tolerance chaining.
6. **P1 root and habit solution inventory.** Prove numerical residuals and completeness on near-double/discriminant-zero roots, lambda endpoints, spectral degeneracies, scale changes, and monoclinic settings. Distinguish mathematical rank-one compatibility from a unique microstructure; give unreconciled fractions explicit provenance.
7. **P1 EBSD/TEM comparisons.** Require the correct basis embedding/gauge and verified OR/frame convention; compare uncertainty with observed habit plane normals as reciprocal objects. Geometric agreement does not establish phase, variant or experimental authenticity.
8. **P1 UI/report labeling and exports.** Every displayed source must clearly show: numerical equation check, type provenance (direct/transported/nonconventional), incomplete/orbit flags, coordinate frames, and independent experiment status. Exports must retain all input and root-audit conditions.
9. **Scientific manuscript benchmarks.** Run blind, full-solution comparisons against literature, not only selected shears/lambdas. Existing benchmark inputs based on *reconstructed rounded stretch ratios* must not be described as specimen measurements.
10. **Release evidence.** No unqualified reliability claim until the independent acceptance workflow, full multi-version pytest, sensitivity/falsification campaigns, and specimen-specific Cu–Al–Ni comparisons all pass. Keep `main` and green checkpoint branches immutable until then.

## Primary references

- James & Hane, Acta Materialia 48 (2000) 197–222, Table 4. Cu–Al–Ni Cu-14 wt.% Al-4 wt.% Ni parent DO3 and reduced 6M cell.
  https://aem.umn.edu/~james/research/Publications_Files/PDF/67.pdf
- Sari & Aksoy, *Journal of Alloys and Compounds* 417 (2006) 138–142, doi:10.1016/j.jallcom.2005.09.049 (2H and M18R structures depend on thermal history).
- Bhattacharya, *Microstructure of Martensite*, Oxford University Press (2003), twinning modes and austenite–martensite compatibility.

**This document inventories blockers, not completed cures. Do not use green CI alone as certification.**
