# Typed crystallographic objects

## Why this layer exists

A raw array such as

```python
np.array([1, 1, 0])
```

has no crystallographic meaning by itself.

It could mean a direct direction

\[
[110]
\]

or a reciprocal plane covector

\[
(110).
\]

Those objects obey different transformation laws in a non-Cartesian crystal
basis.  CuAlNi-CT therefore makes the distinction part of the Python type
system.

The two fundamental types are:

```python
Direction(...)
Plane(...)
```

and they cannot be silently interchanged.

---

## Direct directions

A direction stores coordinates \(u\) in a named direct crystallographic basis.

Its physical length is

\[
|u| = \sqrt{u^T M u}.
\]

Under a direct-space crystallographic operator \(g\),

\[
\boxed{u' = gu}.
\]

Metric normalization is available as a derived view, but never overwrites the
stored crystallographic indices.

---

## Reciprocal planes

A plane stores the reciprocal covector \(p=(hkl)\).

Its reciprocal magnitude is

\[
|p|_* = \sqrt{p^T M^{-1}p}
\]

and the no-\(2\pi\) interplanar spacing used by this package is

\[
d_{hkl} = \frac{1}{|p|_*}.
\]

Under the same direct-space operator \(g\), a plane transforms dually:

\[
\boxed{p' = g^{-T}p}.
\]

This distinction is essential for monoclinic 6M and for future EBSD,
stereographic, habit-plane, and twin-plane work.

---

## Incidence is representation independent

A direction lies in a plane when

\[
p^Tu=0.
\]

The normalized incidence residual is

\[
r_{\rm inc}
=
\frac{|p^Tu|}
{\sqrt{u^TMu}\sqrt{p^TM^{-1}p}}.
\]

This is exactly the absolute cosine between the physical direction and the
plane normal.

In any Cartesian realization \(B\),

\[
x=Bu,
\qquad
g_c=B^{-T}p,
\]

and therefore

\[
g_c^T x = p^T u.
\]

The test suite verifies this for every supported Cartesian convention,
including the PTCLab-compatible frame.

---

## Angles are explicit about orientation sense

The API deliberately distinguishes:

- `direction_angle_deg`: oriented direct-vector angle in \(0^\circ\ldots180^\circ\);
- `axis_angle_deg`: projective direction-axis angle, identifying \(u\) and \(-u\);
- `plane_normal_angle_deg`: oriented reciprocal-normal angle;
- `interplanar_angle_deg`: projective plane angle, identifying \(p\) and \(-p\);
- `direction_plane_angle_deg`: acute direction/plane angle.

This prevents hidden sign conventions from contaminating variant,
misorientation, pole-figure, or twin calculations.

---

## Coordinate-basis identity

Each object carries a `CrystalBasisRef`:

```text
phase_id
basis_id
cell_representation
```

Two objects in different crystallographic bases cannot be compared by the
same-lattice angle functions.  The software requires an explicit
orientation/correspondence map first.

This is intentional.

A vector in DO3 coordinates and a vector in 6M coordinates do not become
comparable merely because both happen to contain three numbers.

---

## Provenance

Every typed object may carry:

```text
status
source_key
uncertainty
notes
```

using the existing `DataStatus` vocabulary.

A future GUI can therefore display, for example:

```text
[1 1 0]A
USER_MEASURED
source: ebsd_map_001
uncertainty: ±0.5°
```

rather than separating displayed geometry from scientific provenance.

---

## Symmetry equivalents

The same direct-space symmetry operators generate both kinds of objects, but
with their mathematically correct actions:

\[
u' = gu,
\qquad
p' = g^{-T}p.
\]

Direction equivalence has an explicit `projective` option because opposite
directions may or may not be considered equivalent depending on the task.

Plane families default to projective equivalence because \((hkl)\) and
\((-h,-k,-l)\) describe the same unoriented plane family.

No sign convention is hidden.

---

## What this milestone deliberately does not do

This layer does not yet introduce:

- cross-phase orientation relationships;
- EBSD Euler-angle conventions;
- semantic `HabitPlane` / `TwinPlane` wrappers;
- project-wide phase lookup;
- GUI state.

Those depend on the next `Project / Scientific State` layer.

The important foundation is now locked:

> direct vectors and reciprocal covectors are different mathematical types,
> live in explicitly identified crystallographic bases, transform differently,
> and produce representation-independent physical geometry.
