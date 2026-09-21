# Real-map EBSD pipeline

## Purpose

This layer turns the frozen crystallographic backend into a reproducible
real-experiment runner.

It does **not** add a new crystallographic theory.  It orchestrates the already
validated components:

- vendor-neutral EBSD I/O;
- proper SO(3) orientation conventions;
- phase-aware grain segmentation;
- KAM and GOS;
- exact orientation-topology variant generation;
- child/child operator comparison;
- candidate-parent reconstruction;
- variant graph ambiguity;
- bounded child-boundary OR refinement;
- two-sided crystallographic trace validation;
- adversarial convention diagnostics.

The single JSON configuration is part of the scientific record.

## Non-negotiable design rules

The pipeline never guesses a scientific input that can materially change the
interpretation.

It requires explicit phase IDs, point groups, lattices and parent/product
roles.

For CSV/HDF5 data it requires an explicit orientation convention.

It never chooses among multiple projective OR branches unless
`candidate_index` is supplied.

Quality thresholds are explicit rules.  CI, IQ, MAD, BC, BS or any other
vendor metric is not converted to an angular uncertainty.

An experimentally fitted OR remains separate from the configured OR.  It is
used for the final theory only when both conditions are true:

```text
accepted_improvement == true
use_if_accepted == true
```

Overlapping parent-domain candidates are retained as ambiguity.

## OR input

### Direct physical matrix

The internal convention is

\[
x_A=R_{A\leftarrow M}x_M.
\]

A config may provide that proper matrix directly.

If the supplied matrix is `product_from_parent`, the pipeline transposes it
explicitly.

### Two crystallographic parallelisms

A second route uses the metric-native `OrientationKernel`.

Each relation can be direction/direction, plane/plane, direction/plane or
plane/direction.  Planes use reciprocal metric geometry and directions use
direct metric geometry.

Two independent relations determine one or more proper SO(3) branches.
Projective signs are enumerated.  If more than one branch survives, execution
stops until `candidate_index` is supplied.

No EBSD fit is used to resolve that branch automatically.

## Data filtering

Quality filters are a list of explicit rules such as

```json
{"field": "CI", "op": ">=", "value": 0.1}
```

A rejected point becomes unindexed in the working map:

```text
indexed = false
phase_id = 0
orientation = NaN
```

No orientation is interpolated or fabricated.

The original and working map audits are both written to `summary.json`.

## Segmentation

The configured main threshold is used for the working grain map.

A separate ordered threshold sweep reports:

- grain count;
- retained fraction;
- median GOS;
- 95th percentile GOS;
- Adjusted Rand stability to the next threshold.

The sweep is diagnostic.  It does not silently replace the configured main
threshold.

## Theory / experiment separation

The theoretical library is generated from the configured OR before the
experimental boundary network is scored.

The initial and final theory scores are written separately.

If OR refinement is enabled, the fitted OR is reported even when it is not
accepted or not selected for final interpretation.

## Parent domains and variants

The candidate-level variant graph may return multiple parent-domain candidates
containing one child grain.

Pixel/grain domain codes therefore use:

```text
-1  no accepted parent domain
-2  multiple accepted parent domains: ambiguous
>=0 unique domain index
```

A unique variant ID is written only for a uniquely assigned domain with an
accepted variant match.

All `(domain, grain, variant)` alternatives remain in
`grain_variant_assignments.csv`.

## Boundary operators

Every reconstructed product/product grain boundary is compared with the full
theoretical operator library.

The output records:

- best operator;
- best residual;
- second-best residual;
- ambiguity gap;
- accepted/rejected;
- ambiguous/not ambiguous.

No angle-only shortcut is introduced at this level.

## Trace validation

Trace validation is disabled unless the specimen surface normal is explicitly
supplied.

For each boundary the geometric trace is reconstructed from neighbour-edge
midpoints.

Each configured twin plane pair is evaluated in both side assignments:

```text
plane1 -> grain A, plane2 -> grain B
plane2 -> grain A, plane1 -> grain B
```

The output records which side assignment fitted better.

Curved or under-resolved boundaries receive a diagnostic status instead of a
fabricated exact trace score.

## Atomic and reproducible outputs

A run is first written into a staging directory.

Only after every requested calculation and output succeeds is the staging
directory atomically promoted to the requested run directory.

If replacement is explicitly requested, the previous run directory is first
moved to a temporary backup and restored if promotion fails.

Every run records:

- canonical configuration SHA-256;
- raw EBSD input SHA-256;
- absolute input path;
- Git commit / branch / `git describe` when available;
- resolved configuration;
- configured and final OR;
- theory variant/operator matrices;
- map, segmentation and reconstruction summaries.

## Output files

```text
summary.json
resolved_config.json
theory.json

phases.csv
grains.csv
boundaries.csv
parent_domains.csv
grain_variant_assignments.csv
trace_validation.csv
segmentation_sweep.csv

map_fields.npz
```

`map_fields.npz` contains point-level arrays:

```text
x, y, z
indexed
phase_id
grain_id
kam_deg
parent_domain_id
variant_id
variant_residual_deg
```

## Running

Validate the static scientific configuration first:

```bash
python -m cualni_cryst.ebsd_pipeline_cli validate-config path/to/config.json
```

Then run:

```bash
python -m cualni_cryst.ebsd_pipeline_cli run path/to/config.json
```

Configuration validation intentionally does not parse the EBSD file, so it can
be used to audit the crystallographic setup before a large map is processed.
