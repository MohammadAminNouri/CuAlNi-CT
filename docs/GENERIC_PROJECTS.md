# Generic Project / Crystal Input Layer

This milestone removes the James-Hane benchmark as a hard-coded universe.
James-Hane remains a verified preset and regression benchmark.

## What is now generic

A project JSON may define any valid conventional lattice

`a b c alpha beta gamma`

plus one of the 32 conventional crystallographic point groups, or an explicit
list of symmetry matrices for a nonstandard setting.

The same project may store:

- one or many phases;
- arbitrary cell representations;
- stored physical orientation matrices;
- exact correspondence matrices (integer, decimal, or `p/q` entries);
- provenance/sources;
- composition and thermomechanical state;
- explicit basis-change links between alternative representations of one
  physical phase.

## Built-in point groups

The registry covers all 32 crystallographic point groups in standard
conventional settings. Metric preservation is independently checked against the
actual user cell, so e.g. a `4/mmm` point group cannot silently be attached to
an orthorhombic `a != b` metric.

Trigonal built-ins use hexagonal axes. A rhombohedral-axis or other nonstandard
setting can be supplied with explicit symmetry matrices.

## Presets

```bash
cualni-project presets
```

Initial literature presets:

- `preset:james_hane_2000`
- `preset:perez_cerrato_2024_beta3`

The Perez-Cerrato preset contains only the published beta'3 martensite cell:

- C2/m, point group 2/m, unique b
- a = 13.817 A
- b = 5.2856 A
- c = 4.3987 A
- beta = 113.6 deg

No austenite cell or OR is invented.

## Create an arbitrary two-phase project

```bash
cualni-project create-pair my_pair.json \
  --reference-id austenite \
  --reference-cell "5.836 5.836 5.836 90 90 90" \
  --reference-point-group m-3m \
  --moving-id martensite \
  --moving-cell "13.817 5.2856 4.3987 90 113.6 90" \
  --moving-point-group 2/m \
  --unit angstrom
```

This command deliberately does not invent an OR or correspondence.

Validate:

```bash
cualni-project validate my_pair.json
```

Single-phase CalPad:

```bash
cualni-calpad --project my_pair.json cell --phase martensite
cualni-calpad --project my_pair.json inspect --phase martensite "(1 0 1)"
```

Two-phase workbench with an explicit OR:

```bash
cualni-two summary \
  --project my_pair.json \
  --or "euler:10 20 30"
```

If a project contains exactly two phases, the two-phase engine can infer the
phase pair for a user OR. If it contains more than two phases, `--reference`
and `--moving` are mandatory for matrix/Euler/quaternion/axis/parallelism ORs.

## Alternative cell representations

A representation link uses the explicit convention

`u_from = P_from_to @ u_to`

and audits

`M_to = P_from_to.T @ M_from @ P_from_to`.

Example JSON:

```json
{
  "representation_links": [
    {
      "link_id": "6m_to_beta3",
      "from_phase_id": "cell_6m",
      "to_phase_id": "cell_beta3",
      "coordinates_from_from_to": [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ],
      "notes": "Replace with the sourced exact basis change."
    }
  ]
}
```

The software refuses a link whose metric or volume Jacobian does not match.
It never decides that two cells are equivalent merely because their numerical
lattice parameters look similar.

## PTCLab direction

The intended user flow remains:

`load/create crystals -> choose OR -> type indices -> calculate -> immediate table`

The JSON/project layer is the backend source of truth. A future GUI can expose
it with forms, CIF import, databases and saved workspaces without changing the
scientific core.
