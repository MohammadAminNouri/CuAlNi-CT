# Crystallography Console

## Goal

This is the first deliberately user-facing layer of CuAlNi-CT.

It is designed to retain the ease of a PTCLab-style crystallography calculator
while making coordinate type, representation, provenance and mathematical
derivation explicit.

The console does **not** contain a second crystallography implementation.

```text
human input
    ↓
safe parser
    ↓
Direction / Plane
    ↓
ProjectState
    ↓
verified metric + Cartesian core
    ↓
structured report
    ↓
terminal renderer now / PySide6 renderer later
```

---

## Easy input

The parser accepts:

```text
[1 0 -1]       direction
[1, 0, -1]     direction
(1 1 0)        plane
<1 0 0>        direction / axis family
⟨1 0 0⟩        direction / axis family
{1 1 0}        plane family
[1/2 0 -1]     fractional direction
```

Unicode minus signs are normalized.

Compact notation such as

```text
[10-1]
```

is deliberately rejected because it becomes ambiguous when multi-digit or
fractional indices are allowed.

Bare

```text
1 0 -1
```

is accepted only by the programmatic API when an explicit kind is supplied.

The program never guesses whether three numbers mean \([uvw]\) or \((hkl)\).

---

## Phase names

Users do not have to memorize internal IDs.

For the reference project, all of these are intended to be easy:

```text
6m
DO3
austenite_do3
martensite_long_period
```

Resolution is conservative:

- exact phase ID wins;
- exact normalized aliases are accepted;
- a unique partial match is accepted;
- multiple matches produce an ambiguity error;
- no match lists the valid phase IDs.

---

## Single-object inspection

For a direct direction \(u\), the console reports

\[
|u|=\sqrt{u^TMu}
\]

and the metric-unit coordinates

\[
\hat u=\frac{u}{\sqrt{u^TMu}}.
\]

For a plane covector \(p\),

\[
|p|_*=\sqrt{p^TM^{-1}p},
\]

\[
d_{hkl}=\frac{1}{|p|_*},
\]

and the metric-normalized reciprocal coordinates are reported.

Every object is also shown in every registered Cartesian representation:

- legacy \(a\parallel x,\ b\in xy\);
- PTCLab-compatible \(a\parallel x,\ c\in xz\);
- symmetric \(M^{1/2}\) frame.

The console independently compares the Cartesian norm with the metric norm and
reports the maximum representation residual.

---

## Units

The current `Lattice` object stores dimensional lengths but not a separate
machine-readable unit tag.

Therefore the console deliberately prints:

```text
lattice-length units
1 / lattice-length units
```

rather than silently labeling an arbitrary future project in Ångström.

The James-Hane reference parameters are conventionally supplied in Å, but unit
metadata should become explicit in a later project-schema migration.

---

## Pair comparison

### Direction / direction

The oriented angle is

\[
\cos\theta=
\frac{u^TMv}
{\sqrt{u^TMu}\sqrt{v^TMv}}.
\]

The console also reports the unoriented/projective axis angle

\[
\theta_{\rm axis}=\min(\theta,180^\circ-\theta).
\]

### Plane / plane

The projective interplanar angle uses the reciprocal metric:

\[
\cos\theta=
\frac{|p^TM^{-1}q|}
{\sqrt{p^TM^{-1}p}\sqrt{q^TM^{-1}q}}.
\]

The oriented normal angle is also reported separately.

### Direction / plane

The normalized incidence residual is

\[
r=
\frac{|p^Tu|}
{\sqrt{u^TMu}\sqrt{p^TM^{-1}p}},
\]

and

\[
\alpha=\sin^{-1}(r)
\]

is the acute direction-plane angle.

The `lies in plane` Boolean uses the project's explicit algebraic tolerance.

Each relation is recomputed in every Cartesian convention and compared with the
metric result.

---

## Symmetry families

The console does not blur oriented and projective objects.

For direction input:

```text
[1 0 0]
```

opposite senses remain distinct when generating ordinary symmetry equivalents.

For a family input:

```text
<1 0 0>
```

opposite senses are identified as one unoriented axis family.

Planes use projective equivalence by default because

\[
(hkl)\equiv(-h,-k,-l)
\]

for an unoriented lattice-plane family.

The exact direct action / inverse-transpose dual action remains in the typed
object layer.

---

## Cross-phase mapping

Cross-phase comparison is never performed by comparing raw triplets.

The user explicitly chooses a `TransformationState`.

Directions use

\[
u_M=C_{M\leftarrow A}u_A,
\]

whereas planes use

\[
p_M=C_{M\leftarrow A}^{-T}p_A.
\]

Reverse mapping uses the corresponding inverse laws already defined in
`ProjectState`.

The result receives computation-derived provenance.

---

## Terminal interface

Run the reference-project console with

```bash
python -m cualni_cryst.crystallography_console
```

Useful commands:

```text
phases
use 6m
inspect [1 0 1]
inspect (0 1 1)
inspect <1 0 0>
compare [1 0 1] ; (0 1 1)
transformations
map do3_to_6m_reference [1 0 0]
derive on
quit
```

One-shot mode is also supported:

```bash
python -m cualni_cryst.crystallography_console \
  inspect --phase 6m --derive "[1 0 1]"

python -m cualni_cryst.crystallography_console \
  compare --phase 6m "[1 0 1]" "(0 1 1)"
```

The interactive CLI is a thin renderer over structured reports. The future
PySide6 application will consume those same reports.

---

## Reliability rules

The console:

- validates `ProjectState` before use;
- never decides that `[uvw]` and `(hkl)` are interchangeable;
- never compares different phase bases without an explicit transformation;
- never hides Cartesian convention;
- reports a metric/Cartesian parity residual;
- never silently promotes a family representative into a different physical
  object;
- retains project provenance;
- does not duplicate CT/PTMC/Ball-James equations inside the UI.

---

## Why this architecture is useful for the final product

The eventual application can render the same structured result as:

```text
PROJECT TREE | CRYSTALLOGRAPHY WORKSPACE | SCIENTIFIC INSPECTOR
```

with tabs for:

```text
Metric
Cartesian
PTCLab
Symmetry
Derivation
Provenance
```

without changing any scientific calculation code.

That is the intended advantage over a monolithic calculator: usability and
scientific traceability are separate concerns but operate on the same state.
