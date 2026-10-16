# CuAlNi-CT P1: scientific scope and acceptance contract

**Candidate, not experimentally certified. Based on P0 commit `74ce501ba0d53746064d608eb034decd9b02796d`.**

## What is new

* `p1_cualni.py`: reference-only 2H orthorhombic, monoclinic C2/m and gamma-angle 6M reduced cells, lattice-metric/setting validation; exact matrix C required for simulation. The 6M cell can be rebased to the conventional monoclinic unique-b metric, but doing this is **not** a physical correspondence or proof that reduced 6M equals another 18R cell.
* `p1_numerics.py`: independent stretch tensor from metrics and C, principal-strain and degeneracy diagnostics, 65-digit mpmath polynomial-root enumeration for laminate fractions, strong endpoint/continuum warnings, symmetry-preserving parent/daughter sensitivity operators and conditional linear uncertainty.
* `p1_ensemble.py`: full-solver optional user-declared Gaussian measurement covariance sampling; no arbitrary sigma, branch-switching and numerical failures remain visible and prevent quantitative claims.
* `p1_evidence.py`: proper-rotation parent metric-group validation, macroscopic twin/habit tensor orbits (not unique laminate mechanisms), crystallographic reciprocal normals (not directions), and frame-aware numerical comparison of independent EBSD/TEM habit normals. Every match is **geometric only**.
* `p1_ui.py`: phase-setting reference guide plus numerical classification and validation status next to the actual selected mode branch. Never copies literature result tables into the solver.
* `run_p1_scientific_acceptance.py`: generate full synthetic or specimen reports with root and orbit checks; output JSON in /tmp; speculative correspondence results never labelled experimentally validated.
* `tests/twin_app/test_p1_contracts.py`: seventeen isolated unit tests and negative controls. Existing full suite and P0 metamorphic acceptance must be rerun after installation.

## Exact equations

For passive correspondence C (parent direct coordinates mapped to daughter direct coordinates), with parent/direct G_A and product/direct G_M:

    U^2 = G_A^{-1/2} C^T G_M C G_A^{-1/2}

Here sqrt denotes SPD root; numerical frame is the symmetric-root parent gauge. The laminate is `F(lambda) = U_i + lambda (a outer n)`. Necessary exact habit fraction roots satisfy:

    det(F(lambda)^T F(lambda) - I) = 0

A physical habit requires the middle singular value to equal one and a proper orientation-preserving rank-one factorization. A determinant root **alone is not enough**. At double roots, continuous fractions or endpoints the tool must report **inconclusive/needs investigation**, not silently assume completeness.

For reciprocal plane normals, `n_cart ∝ B^{-T} h` if `G = B^T B`; for directions, `d_cart = B u`. Parent point-group orbit actions are `H → QHQ^T` with metric-preserving proper Q. Orbit equivalence of H=b⊗m does not prove equivalence of internal laminate microstructures.

## Cu–Al–Ni literature and observations

Published references are *not* mutually interchangeable specimens.

* Orthorhombic 2H `gamma1_prime_2H` (Pmmn / mmm): review-cited cell (5.3424, 4.2244, 4.3896) Å, conventional orthorhombic axes. Underlying primary diffraction source must be verified before thesis numerical benchmarking.
* Monoclinic `beta_prime_monoclinic` (C2/m / 2/m): published EBSD indexing/ref cell (13.817, 5.2856, 4.3987) Å, beta 113.6°, conventional unique-b. Do not call this an 18R cell equivalent to the next entry without an explicit full lattice mapping.
* James & Hane reduced 6M (4.430, 5.330, 12.79) Å, gamma 95.68°. This is a **different description/sample** and gamma-unique-axis convention must be rebased for a unique-b 2/m solver.
* Chen et al., Micron 31 (2000) 17–25, DOI 10.1016/S0968-4328(99)00060-8, reports 2H OR and 2H twinning planes. The app cannot automatically construct the complete C from a short OR description and no atomic structure.
* The specimen must have verified composition, parent phase, daughter phase, cell setting, exact C with source, measured OR/frame, measured interface normal/frame and uncertainty before experimental testing. Even complete metadata do not validate model/experiment agreement.

## Installation

1. Save the single-file `CuAlNi_CT_P1_COMPLETE_CANDIDATE.py` in Codespaces root.
2. `python CuAlNi_CT_P1_COMPLETE_CANDIDATE.py --check` — read-only.
3. Only if pass: `python CuAlNi_CT_P1_COMPLETE_CANDIDATE.py --apply` — safeguarded, **no commit or push**.
4. `python CuAlNi_CT_P1_COMPLETE_CANDIDATE.py --test` — focused unit tests.
5. `python CuAlNi_CT_P1_COMPLETE_CANDIDATE.py --scientific-quick` — end-to-end synthetic numerical validation; may take minutes.
6. Run both old P0 deep acceptance and full pytest matrix separately. Do not commit if any fail.

## Known scientific boundaries (cannot be solved by code alone)

* No fully verified specimen-specific 2H/18R/6M parent→daughter rational C or measured OR is included. Lattice metrics alone cannot establish C.
* Some user-input CIFs lack reliable atomic occupancies, stacking sequence, pseudosymmetry and indexed phases; software input validation cannot replace XRD/EBSD/EPMA/TEM structure refinement.
* These equations capture stress-free crystallographic rank-one compatibility, not microstructure selection due to grain constraints, defects, kinetics, texture, interface energy or processing history.
* The endpoint/double-root/continuous-fraction oracle is an **independent diagnostic**; computational completeness for all admissible spectra and nonlinear microstructures is not mathematically certified.
* Local lattice-parameter derivatives do not yield uncertainty for predicted normals across branch switching. A full measurement covariance and branch-aware sampling would be needed.
* Distinct source records, macroscopic symmetry orbits, and physical microstructure variants are separate objects and have deliberately separate labels.
* Real experimental conclusions must compare predictions against independent, quality-controlled, specimen-matched EBSD/TEM with crystallographic frame conversions and experimental uncertainty.
