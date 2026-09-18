# Orientation relationships: the C != R boundary

## Cayron variant topology correction

The orientation layer now follows Cayron's intersection-group definition rather
than counting all raw left/right symmetry products.  For

```text
x_A = R_A<-M x_M
```

we calculate

```math
H_T^A = G_A \cap R_{A\leftarrow M} G_M R_{A\leftarrow M}^{-1}
```

and define orientation variants as the left cosets `G_A / H_T`.  Orientation
operators are the double cosets `H_T g H_T`.

For the current truth-locked DO3 -> 6M polar candidate the full point groups
give `48/4 = 12` orientation variants and 8 orientation operators; the proper
SO(3) parts give `24/2 = 12` physical orientation representatives.  The old
24-row output was therefore a list of raw symmetry-related matrices, not 24
crystallographically distinct orientation variants.

`H_T` is also compared independently with the exact correspondence subgroup
`H_C`.  Their equality is reported only when it is actually obtained for the
chosen OR candidate.  It is never assumed as a general law.

See `docs/CAYRON_CT_PARALLEL_AUDIT.md` for the complete theory-to-code audit.


This milestone establishes the orientation layer required for genuine
PTCLab-style two-phase calculations and for fair Cu-Al-Ni comparisons among
Cayron CT, Ball-James, PTMC, literature ORs and experiment.

## Core distinction

Correspondence:

\[
u_M=C_{M\leftarrow A}u_A,\qquad
p_M=C_{M\leftarrow A}^{-T}p_A.
\]

Orientation relationship:

\[
\boxed{x_A=R_{A\leftarrow M}x_M}.
\]

Therefore:

\[
\boxed{C\neq R}.
\]

The code never substitutes one for the other.

## Accepted orientation inputs

- proper 3x3 rotation matrix;
- explicit bounded SO(3) repair of a near-rotation;
- axis-angle;
- reference crystallographic axis `[uvw]` + angle;
- standard Cartesian Hamilton quaternion `(w,x,y,z)`;
- explicit ZXZ Euler angles;
- two crystallographic parallelisms.

Matrix repair is never silent.

## Explicit Euler convention

`zxz_active` means exactly

\[
R=R_z(\phi_1)R_x(\Phi)R_z(\phi_2)
\]

on Cartesian column vectors.

`zxz_passive` is the transpose/inverse.

At \(\Phi=0^\circ\) or \(180^\circ\), Euler angles are marked singular, while
matrix, axis-angle and quaternion remain valid.

## Parallelism solver

The OR solver uses metric-correct physical vectors from each phase.

Plane signs are intrinsically unoriented. Direction signs can be treated as
axes or as oriented vectors. When sign ambiguity produces more than one exact
proper rotation, every candidate is returned instead of silently selecting one.

## Representation parity

Every OR can be expressed in all nine pairs of the three existing Cartesian
conventions:

- legacy `a || x, b in xy`;
- PTCLab `a || x, c in xz`;
- symmetric metric `B=M^(1/2)`.

For frame rotations \(Q_A,Q_M\),

\[
R^{new}=Q_A R^{old}Q_M^{-1}.
\]

Mapped direct vectors and reciprocal plane normals are audited across all nine
representations.

## Proper-symmetry OR variants

Only proper point-group operations are used for SO(3) orientation variants:

\[
R'=S_A R S_M^{-1}.
\]

For the current reference project this means 24 proper cubic rotations and 2
proper monoclinic rotations. Improper operations remain available elsewhere
for correspondence/groupoid crystallography.

## Orientation mapping

For a moving direct vector,

\[
u_A=B_A^{-1}R_{A\leftarrow M}B_Mu_M.
\]

For a moving plane,

\[
p_A=B_A^TR_{A\leftarrow M}B_M^{-T}p_M.
\]

These are physical orientation mappings, not correspondence mappings.

## Polar rotation candidate

The existing transformation bridge gives

\[
F=B_MC B_A^{-1}=R_{\rm polar}U.
\]

The OR service exposes

\[
R_{A\leftarrow M}=R_{\rm polar}^T
\]

as a **comparison candidate**.

It is explicitly not declared to be Cayron's \(T\), a Ball-James rotation, a
PTMC OR, or an experimental OR.

## Quaternion warning

The current OR output uses the standard Cartesian Hamilton quaternion.

It is **not** Cayron's 2026 crystallographic metric quaternion. A future Cayron
adapter must keep those two quaternion algebras separate and cross-check them.

## Research comparison contract

Future theory adapters should emit candidates into the same OrientationState:

```text
experimental EBSD OR
literature OR
polar(C + metrics)
Cayron CT T
Ball-James R
PTMC R
```

Then compare:

- raw misorientation;
- symmetry-reduced disorientation;
- selected variant;
- habit plane;
- shear direction and magnitude;
- twin relation;
- compatibility residual;
- experimental uncertainty.

An OR match alone is not enough to establish that a transformation theory is
correct.

## CLI examples

```bash
cualni-or polar \
  --transformation do3_to_6m_reference \
  --representations \
  --variants
```

```bash
cualni-or matrix \
  --reference do3 \
  --moving 6m \
  --matrix "1 0 0; 0 1 0; 0 0 1"
```

```bash
cualni-or axis-angle \
  --reference do3 \
  --moving 6m \
  --axis "[1 1 0]" \
  --angle 5
```

```bash
cualni-or euler \
  --reference do3 \
  --moving 6m \
  --phi1 10 --Phi 20 --phi2 30 \
  --euler-convention zxz_active
```

```bash
cualni-or quaternion \
  --reference do3 \
  --moving 6m \
  --q "1 0 0 0"
```

```bash
cualni-or parallel \
  --reference do3 \
  --moving 6m \
  --ref1 "(0 1 0)" \
  --mov1 "(0 1 0)" \
  --ref2 "[1 0 0]" \
  --mov2 "[1 0 0]"
```

```bash
cualni-or compare-polar \
  --transformation do3_to_6m_reference \
  --matrix "1 0 0; 0 1 0; 0 0 1"
```

## Next milestone

CalPad parity II should consume registered OrientationState objects for genuine
two-phase calculations, cross-phase equivalent-angle tables and coordinate
transformations.

After that, dedicated theory adapters can emit \(T_{CT}\), \(R_{BJ}\),
\(R_{PTMC}\) and experimental EBSD ORs into the same comparison schema.
