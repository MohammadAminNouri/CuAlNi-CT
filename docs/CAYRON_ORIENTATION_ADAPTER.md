# Cayron orientation adapter

## Scope

This layer adds Cayron closing-gap orientation relationships without changing
the existing generic crystallography, CalPad, two-phase, OR, or PTCLab-ready
architecture.

It is intentionally **not** a universal `T_CT = f(M_A, M_M, C)` function.

Cayron's Correspondence Theory assumes a natural orientation relationship and
then permits small additional rotations to recover compatibility between
variants. The code therefore separates:

- an explicitly supplied **natural OR**;
- exact **Type-I closing-gap ORs**;
- exact **Type-II closing-gap ORs**;
- the existing **polar-decomposition OR comparator**;
- future weak-twin orientation work.

`C`, `R/T`, `F`, and `U` remain separate first-class objects.

## Physical OR convention

The generic orientation engine uses

```text
x_A = R_A_from_M x_M
```

where `A` is the parent/reference phase and `M` is the
martensite/moving phase.

The CT adapter returns the same `OrientationState` type as the general OR
workbench. It therefore plugs into the existing variant, misorientation, EBSD,
two-phase, and future PTCLab layers without creating a second orientation
representation.

## Type-I closing-gap OR

For a parent reflection, the CT twin engine calculates:

- rational `K1`;
- metric-dependent `eta1`;
- twin shear magnitude.

The local closing-gap OR satisfies

```text
K1_A   || K1_M
eta1_A || eta1_M
```

The plane is handled as a reciprocal covector and converted to its physical
Cartesian normal through `B^-T p`. The direction is converted through `B u`.
The two are never identified numerically in a non-cubic cell.

## Type-II closing-gap OR

For a parent proper twofold, the CT twin engine calculates:

- rational `eta2`;
- metric-dependent `K2`;
- twin shear magnitude.

The local closing-gap OR satisfies

```text
eta2_A || eta2_M
K2_A   || K2_M
```

## Proper-rotation construction

For a unit direction `d` lying in a plane with unit physical normal `n`, define

```text
Q = [ d, n x d, n ].
```

`Q` is a right-handed orthonormal frame. For parent and martensite frames,

```text
R_A_from_M = Q_A Q_M^T.
```

Crystallographic planes and directions are projective in the parallelism
statement, so all sign branches are generated explicitly. No branch is hidden.

Only tiny floating-point loss of direction/plane incidence is corrected by an
explicit projection. The original incidence residual and correction magnitude
are reported. Inputs outside the configured tolerance fail loudly.

No generic SO(3) repair is applied.

## Natural OR and branch selection

A natural OR must be supplied explicitly from:

- literature;
- experiment;
- an atomistic/physical model;
- or an explicitly labelled comparison hypothesis.

It is never synthesized from correspondence and metrics alone.

When a natural OR is supplied, every closing-gap candidate is compared through
the existing `OrientationService.compare_orientations()` machinery. The
selection quantity is the **symmetry-reduced disorientation**, not merely the
raw matrix angle.

This follows Cayron's 2022 comparison of closing-gap ORs with the natural OR:
the reported deviation is the lowest disorientation among orientation
variants.

The minimum-deviation branch is reported, but this is not treated as proof that
the branch is experimentally realized.

## Full 48-operation cubic symmetry remains essential

For cubic `m-3m`:

```text
48 full crystallographic operations
= 24 proper + 24 improper
```

The CT correspondence/groupoid layer uses the full crystallographic group.

- parent mirrors feed the Type-I route;
- parent proper 180-degree rotations feed the Type-II route;
- other operators remain polar/weak/nonconventional candidates.

The OR itself is always represented by a proper matrix in `SO(3)`.

Therefore the 48-operation crystallographic group and the 24-operation proper
rotation subgroup have different jobs and are never collapsed into one.

## Generic parent symmetry: no cubic-only classifier

The CT twin engine now validates order-two parent operations directly against
the supplied parent metric:

```text
G^T M_A G = M_A
G^2 = I
```

Then:

```text
mirror reflection: det(G) = -1, tr(G) = +1
proper twofold   : det(G) = +1, tr(G) = -1
```

These are basis-invariant statements and work in non-orthonormal
crystallographic bases.

This removes the earlier cubic signed-permutation assumption from the Type-I
and Type-II twin entry points.

## Twin provenance checks

Before constructing an OR from a `CTTwin`, the adapter verifies that it belongs
to the bound transformation:

```text
C_int = C g_A C^-1
plane_M || C^-T plane_A
direction_M || C direction_A
```

A twin object from another correspondence cannot silently pass through the OR
adapter.

## Weak twins

Weak twins are deliberately **not** forced into the exact Type-I/II rotation
construction.

Cayron's weak-plane construction is tolerance/search based and allows small
intrinsic distortion. Treating such a plane as if two exact parallelisms
defined a perfect proper rotation would erase the quantity that makes the
plane "weak".

The adapter therefore raises `NotImplementedError` for a generic weak-twin OR.
The existing weak-plane search layer should later feed a dedicated
weak-orientation fit with explicit distortion/residual metrics.

## Polar comparator

`CayronOrientationAdapter.polar_comparator()` delegates the already existing
polar-decomposition OR.

It preserves

```text
theory_origin = POLAR_CORRESPONDENCE
```

and is never silently relabelled as Cayron CT.

A user may explicitly pass it as a natural-OR hypothesis, but the report emits
a warning that CT does not establish `R_polar = R_natural`.

## Cu-Al-Ni

Nothing in this adapter depends on one James-Hane numerical parameter set.

The adapter consumes the transformation state registered by the project:

- parent metric;
- martensite metric;
- exact cell/basis setting;
- correspondence;
- symmetry;
- explicitly supplied natural OR, if any.

The James-Hane DO3 -> 6M state remains a regression benchmark only.

Other Cu-Al-Ni states can use the same adapter once their own phase metrics,
cell setting, correspondence, and OR provenance are supplied. Missing
quantities must remain missing rather than being borrowed from the 6M
benchmark.

## Main source basis

The implementation follows the mathematical distinctions and closing-gap
logic in:

- C. Cayron, *The Correspondence Theory and Its Application to NiTi Shape
  Memory Alloys*, Crystals 12 (2022) 130.
- C. Cayron, Acta Materialia 316 (2026) 122399, especially the metric-native
  Type-I/II twin equations and the separation of direct directions from
  reciprocal planes.
- C. Cayron, *Groupoid of orientational variants*, Acta Crystallographica A 62
  (2006) 21-40, for variant/operator topology.

The independent Ball-James, Mallard, cofactor, PTMC, polar, literature, and
experimental branches remain separate comparison paths.

## Minimal API workflow

The adapter is deliberately downstream of project state and the generic OR
service:

```python
from cualni_cryst.ct_orientation import CayronOrientationAdapter
from cualni_cryst.orientation import OrientationService

service = OrientationService(project)
adapter = CayronOrientationAdapter(service, "my_transformation")

# natural_or is created independently from literature, experiment, or a
# separately documented physical model.  CT does not manufacture it here.
validated_natural = adapter.natural_orientation(natural_or)

# g_A must be an exact parent mirror belonging to the crystallographic
# operator under study.
report_I = adapter.type_i_from_parent_reflection(
    g_A,
    natural_orientation=validated_natural.state,
)

# The report retains every proper projective-sign branch and also exposes the
# branch(es) having the minimum symmetry-reduced deviation from the supplied
# natural OR.
selected = report_I.selected
```

For an exact correspondence double coset, use `adapter.from_operator(...)`.
The full operator is passed through unchanged so that the parent symmetry
member responsible for every Type-I/Type-II result remains auditable.

## What a successful calculation proves -- and what it does not

A near-zero closing-gap parallelism residual proves that the generated proper
rotation satisfies the mathematical Type-I/Type-II parallelisms for the
provided CT twin elements and coordinate conventions.  It does **not** prove
that:

- the supplied correspondence is the physically realized correspondence;
- the supplied natural OR is the physical natural OR;
- that twin/operator is selected in a specimen;
- CT is experimentally correct for that material state;
- the closing-gap OR equals `R_polar`, a Ball-James rotation, or a PTMC OR.

Those are separate comparisons to be performed at the theory-validation and
experimental layers.  This separation is intentional and is part of the
scientific contract of the module.
