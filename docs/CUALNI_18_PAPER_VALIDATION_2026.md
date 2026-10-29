# Cu–Al–Ni independent validation campaign: 18 real papers (2026-10-29)

Target branch only: `feature/twin-family-tree-app-v1`.
The record is `data/benchmarks/cualni_18_paper_campaign_2026.json`.
Execution is `python -m twin_app.cualni_paper_campaign_2026`.
Scientific CI: `tests/twin_app/test_cualni_18_paper_campaign_2026.py`.
Streamlit: **04 → Explore lattice changes for compatibility →
Cu–Al–Ni literature · 18-paper reproducibility and missing-evidence audit**.

## What is actually tested

The campaign has **18 distinct DOIs**, with one record and an executable
source-specific check per paper. It DOES NOT claim 18 physically independent
numerical crystallographic validations.

| Source | DOI | Actual tested condition | Evidence limit |
| --- | --- | --- | --- |
| Hane & Shield (1999) | 10.1098/rspa.1999.0482 | 6 exact U matrices, 30 ordered/15 unordered U pairs, 3 compound and 12 noncompound pairs, Type-I/II/compound Mallard-law shear, two representative PTMC fractions, 96 twinned-habit solution records, no compound A/M habit, single-U lambda2 | Numerically reproduces that specific 2H Ball–James/PTMC model. Not experimental CT Eq42/CC1–CC3. |
| Otsuka, Nakamura & Shimizu (1974) | 10.2320/matertrans1960.15.200 | Published 18R conventional cell lengths, no invented monoclinic beta | No full M_M, source C or same-interface 3D data |
| Ritter et al. (1979) | 10.1007/BF02658387 | 2H orthorhombic metric, determinant/volume and 6 independent reciprocal d-spacings | No same-specimen parent metric or C |
| Chen et al. (2000) | 10.1016/S0968-4328(99)00060-8 | Separate {121} and {101} 2H EBSD twin relationships, parent basal/axis family | Qualitative OR families do not determine a unique C or full 3D habit |
| Sari & Aksoy (2006) | 10.1016/j.jallcom.2005.09.049 | 2H and monoclinically distorted M18R coexistence/heat-treatment phase constraints | 'beta near 90°' cannot be rounded to 90° |
| Landa et al. (2007) | 10.1016/j.msea.2006.02.472 | Existing source-registered cubic/2H cells re-evaluated through metric and reciprocal-space checks | Full cell table not re-extracted from primary fulltext in this campaign |
| Nó, Caillard & San Juan (2009) | 10.1016/j.actamat.2008.10.017 | Lock nine separate TEM habit/OR observations | No published per-interface numeric 3D data in accessible abstract |
| Ge et al. (2022) | 10.1016/j.actamat.2021.117598 | Keep Type-I, Type-II and compound 2H modes distinct | Does not establish complete experimental 3D geometry in abstract |
| Zhang et al. (2023) LPBF microstructure | 10.1016/j.jmst.2022.12.037 | Reported 2H lengths and beta=89.43° cannot be idealized as 90° | Other indexed cell angles/source details require full table |
| Pérez-Cerrato et al. (2022) | 10.3390/ma15186284 | Check literature-borrowed C2/m and Pmmn indexing-cell Gram matrices | Those metrics were not newly measured on the LPBF specimen |
| Zhang et al. (2023) deformation | 10.1016/j.jmst.2023.05.032 | Loading-dependent reorientation remains a required input, not inferred from metric | Cannot simulate applied stress and activation from C alone |
| CuAlNi single-crystal shape recovery (2008) | 10.1080/01411590801891616 | Keep X and lambda intersecting habits separate | No unique measured 3D planes or source C in abstract |
| Zárubová et al. (2009) | 10.1051/esomat/200902036 | Retain 2H/18R stress and temperature phase dependence | No numerical 3D twin/habit same-specimen input |
| Neutron single-crystal diffraction (2008) | 10.1080/10238160802442710 | Stress and temperature condition 18R variant selection | Population requires diffraction observations |
| PRB two-step mixed 2H/18R (2016) | 10.1103/PhysRevB.94.024102 | Check reported ordering of three transformation temperatures | Enthalpy and hysteresis are outside geometry-only model |
| CuAlNi thin-film thermal stability (2023) | 10.3390/nano13182605 | Compare borrowed C2/m indexing cell with its reference reproduction | Same quoted metrics are NOT another independent measurement |
| Two-way effect in CuAlNi (1999) | 10.1016/S0921-5093(99)00438-4 | Preserve heat treatment, stress, phase populations as inputs | Not derivable from a zero-stress lattice-only theory |
| Sedlák et al. (2005) | 10.1016/j.actamat.2005.04.013 | Register measured cubic/2H elasticity domain | Stiffness tensors unavailable from accessible source; no constitutive predictor |

## Hane–Shield: the genuine fully numerical paper test

The paper's published metric input is the ordered cubic austenite
`a0=5.836 Å` and orthorhombic 2H lengths
`a=4.382, b=5.356, c=4.222 Å`.

Independently reconstruct
`alpha=sqrt(2)*a/a0`, `beta=b/a0`,
`gamma=sqrt(2)*c/a0`; build **six distinct orthorhombic
stretch tensors** by placing beta along each cubic coordinate axis and
the two signs of the alpha/gamma off-diagonal splitting in the
perpendicular plane.

For each unordered pair, check exact parent cubic 180° operations
`U_j=Q U_i Q`, use the independent Ball–James
`mallard_law_twins` solver, verify Type-I and Type-II rank-one
residuals and compare twin shears with source Table 1.
Approximate numbers from the source's rounded lattice inputs:
- compound shear ≈0.07441 versus printed 0.0747;
- Type-I and Type-II shear ≈0.26067 versus printed 0.261;
- PTMC lower type-I laminate fraction ≈0.29057 versus printed 0.290;
- PTMC lower type-II fraction ≈0.30110 versus printed 0.301.

Numerical source rounding is respected with **declared** absolute
tolerances: 0.0006 for shears and 0.002 for volume fraction.
The test also checks all 12 noncompound unordered U-pair classes
using independent SVD root bracketing and both ± habit branches,
giving **96 actual calculated BJ/PTMC habit-solution records**.
For three compound unordered pairs, no A/M laminate root may appear,
consistent with the published absence of compound twinned-habit
solutions for this alloy. This tests both positive AND negative science
and explicitly distinguishes single-U A/M (lambda2 ≠1) from twinned
austenite/martensite habit planes.

The complete 1999 problem is a six-stretch **nonlinear-elasticity /
Ball–James/PTMC** benchmark. It is **not** a uniquely inferred 2026
Cayron CMC/SMC + same-physical-twin Eq42 experiment.
No undocumented parent→2H correspondence `C` is inserted.

## Remaining exact Eq42/CC1–CC3 experimental prerequisites

An empirical full supercompatibility comparison must have,
for the **same specimen and same actual phase setting**, measured
parent and product lattice matrices with uncertainties, verified
parent→martensite correspondence `C`, identified M_i and U_j
mapping, observed 3D A/M habit and M/M twin plane/direction with
their coordinate frames, loading/temperature, and a physically
comparable CT/BJ twin record. Two independent papers with similar
lattice dimensions do not supply a specimen-matched bundle.

When these are absent, the campaign emits **NOT IDENTIFIABLE** rather
than inventing a measured orientation relationship or declaring
Cayron Eq42 satisfied. This is required for future defensible
comparison to EPFL EBSD single-/multi-crystal datasets.

## Source reuse and published phase representations

The paper by Pérez-Cerrato et al. (2022), the 2021 LPBF alumina
reinforcement literature and a 2023 CuAlNi thin-film study quote
similar `beta3'` conventional C2/m metrics. These are not independent
proof of the same cell in a different specimen. Also, an 18R
conventional long-axis cell (`c≈38 Å`) and another C2/m cell
(`a≈13.8 Å, beta≈113.6°`) must not be equated without an explicit
integer/rational basis transformation plus verified ordering.

The repo retains its previous `data/reference/cualni_literature_states.json`
and `data/benchmarks/cualni_experimental_evidence_registry_2026.json`.
This 18-paper campaign is an ADDITION and does not mutate measured
source entries or the original twin feature output.

**Tests being green establishes numerical and scope gates, not 18
experimentally validated single-crystal supercompatible alloys.**
