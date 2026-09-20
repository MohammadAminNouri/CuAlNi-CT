# Unified theory comparison

This layer compares **Cayron CT**, **Ball–James/cofactor**, **classical PTMC**,
and optional **experimental observations** without merging the theories.

## Scientific contract

The shared crystallographic input is exactly one validated `ProjectState`
transformation:

- parent lattice and symmetry,
- product lattice and symmetry,
- correspondence,
- project numerical policy and provenance.

The three theory backends are called independently from that same state.

Theory-specific assumptions remain explicit:

- a Cayron natural OR is optional and is not inferred from `C`, `F`, `U`, or
  the polar rotation;
- PTMC requires a lattice-invariant mechanism.  The default comparison
  exhaustively enumerates exact twinning LIS relations; slip, a selected twin
  plane, or a selected variant pair are explicit alternatives;
- Ball–James uses its own nonlinear-elasticity adapter and does not consume CT
  or PTMC predictions.


## Exact symmetry boundary for CT

`ProjectState` stores phase symmetry matrices as finite floats because the
generic orientation, Ball--James and PTMC services are numerical.  Cayron
subgroup/coset/double-coset topology is different: group membership must be
exact.

Before the unified layer calls the CT groupoid it reconstructs each stored
crystallographic symmetry coefficient as a bounded exact rational within the
project representation tolerance, then requires:

- no two stored operations collapse to one exact operation;
- exact identity, inverse and closure of the recovered finite group;
- preservation of the supplied phase metric within project tolerance.

If any of these checks fails, CT topology is refused with a diagnostic.  The
comparison layer never computes exact cosets from approximate floating group
membership.

## Branch preservation

There is no one-row-per-theory collapse.

`ComparisonRow` preserves every discrete branch:

- CT A/M habit-plane candidates;
- CT Type-I/II twins;
- every CT closing-gap sign branch;
- CT supercompatibility pairings when an exact A/M habit plane exists;
- Ball–James A/M rank-one branches;
- Ball–James M/M rank-one branches and cofactor margins;
- PTMC discrete habit solutions;
- PTMC exact continuous families;
- PTMC parameter/root diagnostics;
- experimental observations.

Missing quantities are `None` and render as `N/A`.

## No fake score

Residuals are stored as a named dictionary.  Degrees, dimensionless
compatibility residuals, rank-one residuals and cofactor margins are not mixed
into a scalar score and the software does not declare a theory “winner”.

## Rotation semantics

The common table distinguishes:

- a physical parent-from-product OR (`or_parent_from_product`);
- another mathematically meaningful rotation (`rotation_matrix`) plus an
  explicit `rotation_role`.

In particular, Ball–James rank-one habit rotations are **not** silently
relabeled as physical OR matrices.

## PTMC modes

```python
PTMCMode.NONE
PTMCMode.ALL_TWINNING
PTMCMode.TWIN_PLANE
PTMCMode.TWIN_PAIR
PTMCMode.SLIP
```

The request type must match the mode.  Invalid or ambiguous combinations fail
loudly.

## Example

```python
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.theory_unified import TheoryComparisonAdapter

project = james_hane_6m_reference_project()

report = TheoryComparisonAdapter(
    project,
    "do3_to_6m_reference",
).compare()

for row in report.table_rows():
    print(row)
```

For a lightweight comparison that omits CT closing-gap OR enumeration:

```python
report = TheoryComparisonAdapter(
    project,
    "do3_to_6m_reference",
).compare(
    include_ct_closing_gap=False,
)
```

## Experiment

`ExperimentalObservation` is deliberately generic.  It can hold an OR, habit
plane, twin plane/direction, shear, shape vector and uncertainties without
using the observation to fit any theory backend.

The later EBSD milestone will populate this same object after explicit
frame/convention handling and will add theory-to-experiment matching residuals.

---

# Validation freeze

## Backend transparency

The unified layer was tested against separately executed Cayron CT,
Ball–James/cofactor and classical PTMC backends.

The unified layer is an orchestrator only. It must not modify the numerical
results returned by any theory backend.

### Cu-Al-Ni reference transparency

Independent direct theory reports were compared with the raw reports embedded
inside the unified report.

Validated:

- Cayron CT raw report
- Ball–James raw report
- PTMC raw report
- CT twin branch mapping
- Ball–James A/M branch mapping
- Ball–James M/M branch mapping
- PTMC habit-solution mapping
- PTMC continuous-family preservation
- PTMC root/diagnostic preservation
- unique row IDs
- JSON serialization
- explicit N/A handling

## Exact CT topology reconstruction

The Cayron topology was independently reconstructed from the explicit
ProjectState symmetry matrices without relying on the point-group label.

Reference-state results:

- parent exact group order: 48
- product exact group order: 4
- H_C order: 4
- correspondence variants: 12
- correspondence operators: 8
- CT twin branches: 16

Unified/direct discrepancies:

- twin plane: 0
- twin direction: 0
- twin shear: 0

This also validates nonstandard labels such as `2/m (unique b)` when explicit
symmetry matrices are supplied.

## Multi-system random campaign

Five unrelated transformation systems were exercised:

- m-3m -> 4/mmm
- 6/mmm -> mmm
- mmm -> 2/m
- 4/mmm -> -1
- 1 -> 1

Randomized at runtime:

- lattice parameters
- exact rational correspondence
- CMC spectrum
- CT subgroup
- CT variants/operators
- CT twins
- Ball–James branches
- cofactor quantities
- PTMC roots
- PTMC habit solutions

Results:

- deep numerical comparisons: 29338
- maximum unified/direct discrepancy: `0.000e+00`

Legitimate zero-solution states were preserved rather than converted into
artificial theory predictions.

## Positive-branch campaign

Seed:

`9171362065075828906`

Six randomized exact-compatible states were generated with only the class

`lambda1 < 1, lambda2 = 1, lambda3 > 1`

prescribed.

No habit plane, OR, branch count, twin, shear or PTMC root was prescribed.

Coverage:

- Ball–James A/M branches: 72
- Ball–James M/M branches: 216
- PTMC twin relations: 216
- PTMC discrete habit solutions: 576
- PTMC continuous families: 72

Maximum unified/direct discrepancy:

`0.000e+00`

Result:

`PASS`

## Frozen scientific contract

The comparison layer therefore preserves:

- independent CT mathematics;
- independent Ball–James/cofactor mathematics;
- independent classical PTMC mathematics;
- every discrete theory branch;
- exact continuum PTMC branches;
- theory-specific residuals;
- physical OR versus other mathematical rotations;
- explicit missing quantities as N/A;
- experimental rows without fitting theory to experiment.

It deliberately does not compute a scalar theory score or select a theory
winner.

The following objects remain scientifically distinct:

`C != T != F != U`

and

- Cayron natural OR,
- Cayron closing-gap OR,
- Ball–James rank-one rotation,
- PTMC OR,
- polar rotation,
- experimental OR

must never be silently substituted for one another.
