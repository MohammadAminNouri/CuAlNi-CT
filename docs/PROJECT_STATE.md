# Project / Scientific State

## Purpose

This milestone creates the single scientific state that future calculations and
the GUI will consume.

The GUI must never become the owner of lattice parameters, correspondences,
symmetry matrices, provenance, composition, or numerical tolerances.

Instead:

```text
user interface
    ↓ edits a scientifically typed state
ProjectState
    ↓
calculation service
    ↓
CT / Ball-James / PTMC / cofactor / plots
```

This separation is essential for reproducibility and for PTCLab-like
responsiveness without PTCLab-like monolithic GUI/calculation coupling.

---

## PhaseState

A `PhaseState` keeps separate:

- physical phase identity;
- cell representation;
- `Lattice`;
- `CrystalBasisRef`;
- point-group label;
- symmetry operators;
- provenance.

For the long-period Cu-Al-Ni branch this means the software can state:

```text
physical phase:
    long-period Cu-Al-Ni martensite

computational cell representation:
    6M

basis:
    reference_6M_unique_b
```

rather than treating "18R/M18R/6M" as interchangeable words.

---

## TransformationState

A transformation stores explicit endpoint phase IDs and the exact
`Correspondence` object.

For a registered parent and product:

\[
u_M=C_{M\leftarrow A}u_A,
\]

\[
p_M=C_{M\leftarrow A}^{-T}p_A.
\]

`ProjectState.map_direction()` and `ProjectState.map_plane()` apply these
different direct/dual laws and change the `CrystalBasisRef` to the correct
target basis.

Round-trip and incidence invariance are unit-tested.

The mapped object receives `COMPUTATION_DERIVED` provenance rather than
silently inheriting the status of the input object.

---

## Composition

Composition has an explicit scale:

- weight percent;
- atomic percent;
- weight fraction;
- atomic fraction.

Atomic and weight composition are **not** silently converted.

A balance element is also explicit.  For the James-Hane benchmark the stored
input is:

```text
Al = 14 wt%
Ni = 4 wt%
balance = Cu
```

and the resolved state becomes:

```text
Al = 14 wt%
Ni = 4 wt%
Cu = 82 wt%
```

The software records that balance rule rather than pretending 82 wt% Cu was an
independently measured value.

---

## Thermomechanical state

The state can hold:

- absolute temperature;
- symmetric Cauchy stress tensor in MPa;
- processing/loading history;
- provenance.

Unknown values remain `None` / `NOT_EVALUABLE`.

They are not replaced by room temperature, zero stress, or another convenient
default.

This is critical because phase stability and transformation response in
Cu-Al-Ni depend on thermomechanical state.

---

## Source integrity

A project carries a `SourceRef` registry.

If a phase, transformation, composition, or thermomechanical state claims a
`source_key`, project validation checks that the key actually exists.

A source label can therefore be followed to a concrete reference instead of
becoming an unverifiable string.

---

## Symmetry validation

For each registered point-group operator \(g\), the project audit checks

\[
|\det g|=1
\]

and

\[
\boxed{
g^T M g=M.
}
\]

This is evaluated using the project's explicit numerical policy.

A matrix that looks like a plausible permutation but does not preserve the
actual monoclinic metric is rejected.

---

## Representation validation

Each transformation builds the existing `RepresentationBridge` from the
registered phase lattices and selected Cartesian conventions.

The project validator checks metric/Cartesian parity before the state is
considered scientifically valid.

The project therefore ties together:

```text
PhaseState
    ↕
typed Direction / Plane
    ↕
Correspondence
    ↕
Metric representation
    ↔ Cartesian representation
```

without duplicating physics.

---

## What ProjectState does not do

`ProjectState` is deliberately not a giant calculation class.

It does not calculate:

- variants;
- double cosets;
- twins;
- CMC/SMC;
- PTMC laminates;
- cofactor conditions;
- EBSD reconstruction.

Those belong to the next **Dependency + Calculation Service** layer.

This boundary is intentional:

> ProjectState owns scientific inputs and identities.
> Calculation services own derived results.

That division is what will eventually allow an easy PTCLab-like interface to
remain rigorous and testable.
