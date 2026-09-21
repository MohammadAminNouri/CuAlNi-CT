# EBSD reliability gate

## Purpose

The experimental EBSD pipeline must not infer an orientation relationship,
variant topology, parent reconstruction, or interface relationship from a phase
population that is present only as unreliable indexing islands.

The reliability gate is an evidence-sufficiency layer, not a new
crystallographic equation. It does not alter CT, orientation symmetry,
twinning, Ball--James/Mallard mathematics, or the frozen EBSD disorientation
definitions.

## Route-specific gating

A single global "phase good/bad" flag is scientifically wrong. Different
calculations require different evidence.

### Grain-level product route

The production map pipeline uses the configured product phase for
product/product boundaries, theory scoring, optional OR refinement, and parent
reconstruction.

Before resolving or scoring the experimental OR, the pipeline now requires the
configured product phase to have retained grains after the user's declared
segmentation and minimum-size rule.

The gate does not require the configured parent phase to be observed. A fully
transformed microstructure can legitimately contain product grains only.

No minimum grain size is changed automatically.

### Direct interphase route

Role-free interphase discovery is stricter. For each anonymous phase at each
segmentation threshold, the default gate requires:

- a minimum supported component size of 2 pixels;
- at least 20% of raw phase pixels to belong to spatial components reaching
  that support size;
- at least 20% of phase pixels to belong to orientation-connected components
  reaching that support size;
- at least 8 supported orientation components.

These are conservative evidence-sufficiency defaults, not material constants.
They are explicit settings and can only be changed deliberately.

The direct interface fit uses only interfaces whose components on both sides
meet the support-size requirement.

A singleton-dominated population is therefore reported, preserved, and
rejected for OR fitting; it is never silently converted into grains.

## Significance is not a substitute for crystallographic fit

A small permutation p-value only says the observed pairing scores better than
the chosen null. It does not make a large crystallographic residual acceptable.

The existing fit/holdout residual and support requirements remain mandatory.
The reliability gate acts before optimization; residual/support acceptance acts
after optimization. Both are required.

## Reporting architecture

Human-readable JSON contains summary evidence only.

Large arrays are stored separately:

- `*_arrays.npz`: component labels, component sizes, component phase IDs;
- `*_manifest.json`: SHA-256 hashes and byte counts;
- `*.json`: compact scientific summary.

The command refuses to write a summary larger than 2 MB or one containing
`component_id`, `component_sizes`, or `component_phase_id` arrays.

This prevents accidental multi-million-line JSON reports while preserving
every numerical array in a reproducible binary sidecar.

## Experimental inference versus theory-only calculations

Low-level crystallographic theory functions remain usable without an EBSD map.
A theoretical CT/OR calculation cannot be phase-reliability-gated because no
experimental phase population exists.

The mandatory gate applies at experimental orchestration boundaries:

- end-to-end EBSD pipeline;
- role-free direct interphase discovery.

This keeps the theoretical mathematics general while preventing unsupported
experimental claims.
