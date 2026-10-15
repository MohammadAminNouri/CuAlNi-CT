# Cu–Al–Ni 2H/M18R experimental validation: falsification findings

**Baseline:** green branch `audit/p1-blind-cualni-metrics-2026-10-15`, commit `55a1a64`. All prior work retained. This branch isolates experimental-interface auditing.

## Confirmed implementation defect

In `tools/run_p1_scientific_acceptance.py`, the previous `run_specimen` route collected **only austenite–martensite habit normals**, yet compared **every** `observed_interface_planes` entry with them. This can falsely pass or falsely reject published **martensite–martensite twin** observations from Cu–Al–Ni 2H. The appropriate comparison depends on interface type and reference variant frame.

## Repair

Use `twin_app/p1_interface_validation.py`, with mandatory `interface_type`, specimen-matched `specimen_id`, `source_pair_id`, integer `source_branch`, `observation_reference`, angular uncertainty and explicit `frame`. Product-indexed twin data must be in the **base martensite variant** and compare to `twin_plane_product_crystal` in reciprocal space, not a habit normal. Austenite–martensite habits must compare to `habit_normal_parent_cartesian`; product-indexed measurements require a **variant-specific independently reported OR**, rather than an arbitrary global OR. No guessed correspondence.

Comparison is **geometric only**, never independent experimental certification. Missing inputs are errors, not silent skips.

## Source-level evidence inventory

`data/benchmarks/cualni_experimental_evidence_registry_2026.json` records:
- **Chen et al. (2000), Cu–Al–Ni 2H EBSD:** {121} and {101} are *twin* plane families in 2H product indexing, not A/M habits. Direction [010]2H derives from <001>parent and basal planes from {110}parent.
- **Sari & Aksoy (2006):** both 2H and M18R diffraction under different heat treatments; near-90 monoclinic distortion insufficient for phase discrimination by lattice metric alone.
- **Nó, Caillard & San Juan (2009):** nine measured A/M and M/M OR and interface planes by TEM. Accessible abstract does not provide complete coordinates with measured OR/source specimen matrix C and uncertainty.
- **Zhang et al. (1999):** M18R correspondence and calculated habit-plane tables for Cu–Zn–Al, **not Cu–Al–Ni**. The paper also states input lattice metrics were borrowed from another related Cu–Zn–Al composition. Strong cross-alloy numerical benchmark, not specimen-validated Cu–Al–Ni.

## Genuine remaining release blocker

No reviewed accessible source supplied *in a single paired specimen* (phase-indexed diffraction, parent/product metric, exact rational parent→product correspondence C in the same conventional basis, measured OR tied to an indexed variant, measured twin/habit planes and angular uncertainties). Therefore **complete independent experimental validation for both 2H and M18R is not established**. The software must refuse to claim it. Obtain the full supplementary materials, sample-matched EBSD/TEM raw indexes or collect them in laboratory.

## Adversarial contract

Tests deliberately compare a 2H twin and a different A/M habit in separate frames; incorrect interface routing must fail. Also reject mixed specimens, ambiguous interface types, missing source variant/OR, wrong OR handedness, nonpositive uncertainty and invented matches.

**Never alter the engine's predicted vectors to make Chen's {121}/{101} planes match. Literature is held out as independent evidence.**
