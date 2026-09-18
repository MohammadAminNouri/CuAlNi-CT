# CalPad parity I

This milestone extends the verified human-facing console toward the
single-phase calculation workflow described in the PTCLab manual, while
retaining CuAlNi-CT's stricter direct/reciprocal and metric/Cartesian model.

## User workflow

After installation:

```bash
cualni-calpad
```

or:

```bash
python -m cualni_cryst.calpad
```

Interactive examples:

```text
phases
use 6m
cell

inspect [1 0 1]
compare [1 0 1] ; (0 1 1)

normal (1 0 1)
normal [1 0 1]

low [1 0 1] direction max=3 limit=15
low [1 0 1] plane max=3 limit=15
low (1 0 1) plane max=3 limit=15
low (1 0 1) direction max=3 limit=15

derive on
```

One-shot examples:

```bash
cualni-calpad cell --phase 6m

cualni-calpad normal \
  --phase 6m \
  --max-index 12 \
  --derive \
  "(1 0 1)"

cualni-calpad lowindex \
  --phase 6m \
  --candidate plane \
  --max-index 3 \
  --limit 20 \
  "[1 0 1]"
```

## 1. Phase/cell report

The report shows:

- phase identity;
- physical phase;
- cell representation;
- basis;
- point group;
- number of registered symmetry operations;
- direct cell parameters;
- direct-cell volume;
- direct metric \(M\);
- reciprocal metric \(M^{-1}\);
- reciprocal cell parameters in the package's no-\(2\pi\) convention;
- metric/inverse residual;
- provenance.

For the reciprocal cell:

\[
a^*=\sqrt{(M^{-1})_{11}},\qquad
b^*=\sqrt{(M^{-1})_{22}},\qquad
c^*=\sqrt{(M^{-1})_{33}}.
\]

The reciprocal angles follow from the reciprocal metric. The direct and
reciprocal volumes obey

\[
V^*=\frac{1}{V}
\]

in the no-\(2\pi\) convention.

## 2. Plane -> physical normal

For a plane covector \(p=(hkl)\), a physical direct vector with the same
normal direction has crystal coefficients \(n_c\) satisfying

\[
B n_c \parallel B^{-T}p.
\]

Since

\[
M=B^TB,
\]

we obtain

\[
\boxed{n_c\propto M^{-1}p}.
\]

Only for special metrics, including cubic cells up to a scalar factor, can the
same numeric triplet be used casually for plane and direction.

For monoclinic 6M this shortcut is generally wrong.

The program reports:

1. raw direct coefficients;
2. projectively normalized coefficients;
3. metric-unit coefficients;
4. unit Cartesian normal in every registered Cartesian convention;
5. nearest low-index lattice axis up to the requested index bound;
6. angular mismatch of that low-index approximation;
7. round-trip residual.

The low-index approximation is never called exact unless its angular mismatch
falls below the explicit numerical-policy projective-angle tolerance.

## 3. Direction -> plane with parallel physical normal

For a direct direction \(u\), the reciprocal plane covector whose physical
normal is parallel to that direction satisfies

\[
B^{-T}p\parallel Bu,
\]

thus

\[
\boxed{p\propto B^TBu=Mu}.
\]

Again, the result is generally not an integer Miller plane in a non-cubic
lattice. The exact real metric result and the nearest low-index approximation
are displayed separately.

## 4. Low-index tables

Candidate integer triplets are:

- bounded by \(|index|\le N\);
- reduced to primitive integer triplets;
- stripped of simple scalar multiples;
- optionally projective, identifying opposite senses.

The actual metric is then used for ranking.

### Direction against direction

Projective/axis mode:

\[
\theta_{\rm axis}
=
\min(\theta,180^\circ-\theta),
\]

where

\[
\cos\theta
=
\frac{u^TMv}{|u|_M|v|_M}.
\]

Oriented mode retains the full \(0^\circ\) to \(180^\circ\) direction angle.

### Plane against plane

Projective plane angle:

\[
\cos\theta
=
\frac{|p^TM^{-1}q|}
{|p|_*|q|_*}.
\]

Oriented mode instead reports the oriented reciprocal-normal angle.

### Direction against plane

\[
r=
\frac{|p^Tu|}
{\sqrt{u^TMu}\sqrt{p^TM^{-1}p}},
\]

\[
\alpha=\sin^{-1}(r).
\]

The table also reports the incidence residual.

## 5. Why the search is intentionally finite

A real metric normal in a low-symmetry lattice need not correspond to any
small integer direction or plane.

Therefore the user supplies a finite low-index search bound.

For example:

```text
normal (1 0 1) max=12
```

means:

> compute the exact metric normal first, then find the nearest primitive
> lattice direction with every index magnitude <= 12.

Increasing the bound can produce a closer rational approximation. It does not
change the exact metric result.

## 6. Architecture

The workflow remains:

```text
ProjectState
    ↓
Direction / Plane
    ↓
metric geometry
    ↓
CalPadService structured result
    ↓
terminal renderer now / PySide6 later
```

No scientific formula lives only in a button callback.

## 7. Next milestone

After CalPad parity I is verified, the next foundational feature is an explicit
`OrientationState` and orientation-relationship engine.

That is required before reproducing PTCLab's genuine two-phase Calpad
operations. Correspondence \(C\) and orientation relationship \(R\) must remain
separate mathematical objects.
