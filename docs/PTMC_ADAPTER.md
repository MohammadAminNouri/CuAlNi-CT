# Independent classical PTMC adapter

`cualni_cryst.ptmc_adapter` implements classical **single-shear
phenomenological theory of martensite crystallography (PTMC)** as an
independent theory branch.

It does **not** import or consume predictions from Cayron CT/CMC/SMC or from
`BallJamesAdapter`.  It starts from the same crystallographic state used by the
other theory adapters:

- parent lattice;
- product lattice;
- lattice correspondence;
- parent/product crystallographic symmetry;
- one PTMC-specific lattice-invariant-shear (LIS) specification.

This preserves the project rule that theories share **inputs**, not internal
answers.

## 1. PTCLab-style public workflow

The common front-end can remain:

```text
CRYSTAL A
a b c alpha beta gamma
point group

CRYSTAL B
a b c alpha beta gamma
point group

TRANSFORMATION
correspondence C

PTMC lattice-invariant shear
[twin] plane
or
[slip] plane + direction

CALCULATE
```

The user is not required to type metric matrices, stretch tensors, symmetry
matrices, or rotations in the normal workflow.

For conventional point-group settings:

```python
model = PTMCCrystalInput(
    parent_lattice=A,
    product_lattice=M,
    correspondence=C,
    parent_point_group="m-3m",
    product_point_group="2/m",
)
```

For nonstandard settings use `PTMCAdapter(ProjectState, transformation_id)`,
where explicit symmetry matrices already belong to the registered phase state.

This follows PTCLab's useful input philosophy: lattice parameters, lattice
correspondence, and twin/slip LIS system.  The backend here is intentionally
more explicit about conventions, residuals, branches, and provenance.

## 2. Transformation stretch and variants

The correspondence convention is

\[
u_M=C_{M\leftarrow A}u_A.
\]

With direct metrics \(M_A,M_M\) and the symmetric parent metric basis
\(B_A=M_A^{1/2}\),

\[
U^2
=
B_A^{-T}C^T M_M C B_A^{-1}.
\]

For a proper parent symmetry \(g\), the correspondence variant is

\[
C_i=Cg^{-1}.
\]

The physical parent rotation corresponding to \(g\) is

\[
Q=B_AgB_A^{-1}\in SO(3),
\]

and the resulting stretch satisfies

\[
U_i=QU_0Q^T.
\]

Each candidate is independently audited by reconstructing

\[
F_{C_i}=B_MC_iB_A^{-1}=R_{\mathrm{polar},i}U_i.
\]

Improper crystallographic operations remain part of the full crystallographic
group but are not used as physical rotations.

## 3. Twinning LIS

The PTMC twinning route independently enumerates exact martensite/martensite
rank-one relations

\[
\widehat R U_j-U_i=a\otimes n.
\]

For a volume fraction \(f\) of the `other` variant,

\[
\bar F(f)=U_i+f\,a\otimes n,
\qquad 0\le f\le1.
\]

The classical habit compatibility problem is

\[
R\bar F(f)-\delta I=b\otimes m.
\]

`delta=1` gives a true invariant-plane strain.  A non-unit `delta` is retained
only as the explicit Bowles--Mackenzie dilatational extension; the report then
sets `true_invariant_plane=False`.

### PTCLab-style twin-plane input

PTCLab asks only for the twin plane and obtains a consistent shear direction.
The adapter supports the same high-level behavior:

```python
model.analyze_twin_plane(
    PTMCTwinPlaneInput(
        plane_product=(h, k, l),
    )
)
```

The program does **not** invent a shear direction.  It first enumerates exact
M/M rank-one twin relations and compares their deformed twin-plane normal with
the supplied product `(hkl)`.  If none matches within the explicit angular
tolerance, the calculation fails with the nearest exact relation and its
angular mismatch.

An advanced exact variant-pair route is also available through
`PTMCTwinPairInput`.

## 4. Slip LIS

For a product slip direction \(s\) lying in product plane \(q\),

\[
s\cdot q=0.
\]

The direction and reciprocal-plane normal are converted with the actual product
metric and the selected variant's polar embedding.  In the stretch frame,

\[
S(\gamma)=I+\gamma\,s\otimes q
\]

and

\[
F_{\rm pre}(\gamma)=S(\gamma)U_i.
\]

The shear magnitude \(\gamma\) is **solved**; it is not silently assumed.

The public input is:

```python
model.analyze_slip(
    PTMCSlipSystemInput(
        plane_product=(h, k, l),
        direction_product=(u, v, w),
    )
)
```

The incidence relation `(hkl)·[uvw]=0` is checked before calculation.

## 5. Exact single-shear parameter equation

Both twinning and slip have the rank-one affine form

\[
F(x)=F_0+xA,\qquad \operatorname{rank}A=1.
\]

Compatibility with a uniformly dilated parent plane requires

\[
g(x)
=
\det\!\left(F(x)^T F(x)-\delta^2I\right)
=0.
\]

For rank-one \(A\), \(g(x)\) is at most quadratic.

The engine:

1. reconstructs its three coefficients from evaluations at \(-1,0,1\);
2. verifies the quadratic independently at additional points;
3. solves the quadratic with a cancellation-resistant formula;
4. retains only roots in the allowed domain;
5. checks that \(\delta\) is the **middle** singular value, not merely any
   singular value.

No scanning grid is used to generate the roots.

### Degenerate cofactor-like case

If the polynomial is identically zero, the solution is an exact continuum.
The program returns a `PTMCContinuousFamily`.

It does **not** sample five fractions and claim those samples are the complete
solution set.

## 6. Habit-plane solutions

For every admissible LIS parameter the adapter independently solves

\[
RF_{\rm pre}-\delta I=b\otimes m.
\]

All admissible rank-one branches are retained.

For each solution the report contains:

- LIS mechanism: twin or slip;
- variant indices;
- twin branch where applicable;
- twin fraction \(f\) or slip shear \(\gamma\);
- \(F_{\rm pre}\);
- habit rotation \(R\);
- macroscopic deformation;
- rank-one vector \(b\);
- habit-plane normal \(m\);
- habit plane in parent crystal coordinates;
- habit plane in product crystal coordinates;
- total shape-vector magnitude;
- tangential macroscopic shape-shear vector and magnitude;
- normal shape component;
- effective LIS shear;
- variant volume fractions for twinning;
- invariant/scaled-invariant line where defined;
- complete numerical residuals.

## 7. Orientation relationship

For variant \(i\),

\[
F_{C_i}=R_{\mathrm{polar},i}U_i.
\]

If the PTMC habit solution is \(R_h\), the base-variant physical OR in the
symmetric metric frames is

\[
R_{A\leftarrow M}^{\rm PTMC}
=
R_hR_{\mathrm{polar},i}^T.
\]

For the other member of a twinned laminate,

\[
R_{A\leftarrow M,j}^{\rm PTMC}
=
R_h\widehat R R_{\mathrm{polar},j}^T.
\]

Both are audited as proper rotations.

The same OR is also converted into the repository's PTCLab-style conventional
Cartesian frames.  This lets the simple front end show familiar OR matrices
without changing the theory-native metric backend.

## 8. Invariant line

For `delta=1`, the intersection of the final habit plane and the rotated LIS
plane gives the classical invariant line when those planes are not parallel.

For `delta != 1` the same geometric line is checked against uniform scaling by
`delta`; it is therefore not mislabeled as an exactly invariant line.

The report stores parent and product crystal coordinates and a direct
deformation residual.

## 9. Scientific separation

The adapter deliberately has no imports from:

- `ct`;
- `ct_orientation`;
- `ct_weak_orientation`;
- `twinning_ct`;
- `ball_james_adapter`;
- `compatibility_atlas`.

The rank-one proposition needed inside PTMC is reproduced locally and tested
from its own equations.  This avoids generating PTMC predictions by reusing a
Ball--James report.

Cross-comparison belongs in the later unified comparison milestone.

## 10. Numerical behavior

Scientific invalid input fails loudly:

- non-SPD metrics;
- singular correspondence;
- orientation-reversing lattice deformation;
- symmetry that does not preserve the supplied metric;
- slip direction not lying in the slip plane;
- non-rank-one affine LIS increment;
- invalid variant indices;
- nonexistent exact twin-plane match.

Rounded literature inputs are never silently changed to satisfy an exact
compatibility equation.

When an exact root is absent, bounded searches retain a nearest middle-stretch
residual for diagnosis, but that near miss is **not** promoted to a PTMC
solution.

## 11. Sources and model scope

The implementation follows the classical single-shear PTMC/WLR/BM structure:
lattice deformation plus lattice-invariant shear (twinning or slip), followed
by the rotation that produces an invariant-plane strain.

Useful references:

- Wechsler, Lieberman & Read, *Trans. AIME* 197 (1953), 1503-1515.
- Bowles & Mackenzie, *Acta Metallurgica* 2 (1954), classical crystallographic
  theory papers.
- Wayman, *Introduction to the Crystallography of Martensitic
  Transformations*.
- PTCLab user manual, section 4.1.1, for the user-facing lattice /
  correspondence / twin-slip workflow.

This milestone is intentionally **single shear**.  PTCLab's double-shear model
is a separate theory extension and is not silently folded into the classical
solver.
