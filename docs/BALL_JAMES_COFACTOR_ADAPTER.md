# Independent Ball–James / cofactor adapter

## Scope

This milestone implements the nonlinear-elasticity branch independently from
Cayron CT, CMC, SMC and PTMC.  Its only physical inputs are the parent/product
metrics, lattice correspondence and crystallographic symmetry registered in the
project state.

It does **not** obtain habit planes, twins, rotations, or cofactor results by
transforming a Cayron prediction.

## Coordinate convention

The lattice correspondence is the repository convention

\[
u_M=C_{M\leftarrow A}u_A.
\]

The product metric is pulled back to the parent crystal basis and then whitened
with the symmetric parent basis \(B_A=M_A^{1/2}\):

\[
G=M_A^{-1/2}C^T M_M C M_A^{-1/2},\qquad U=G^{1/2}.
\]

The same principal stretches are independently checked from the generalized
metric eigenproblem

\[
C^T M_M C v_i=\mu_i M_A v_i,
\qquad \lambda_i=\sqrt{\mu_i}.
\]

Parent crystallographic symmetry matrices \(g\) are represented in exactly the
same orthonormal frame by

\[
Q=M_A^{1/2}gM_A^{-1/2}.
\]

Only \(\det Q=+1\) operations generate physical stretch variants:

\[
U_i=Q_iU_0Q_i^T.
\]

Improper parent operations remain part of the crystallographic project state,
but are not used as physical rotations in the nonlinear-elasticity orbit.

## General rank-one compatibility

For any two orientation-preserving deformation gradients \(F_o,F_b\), the
adapter solves

\[
R F_o-F_b=a\otimes n.
\]

Following Ball–James / James–Hane Proposition 1, define

\[
A=F_oF_b^{-1},\qquad C_r=A^TA.
\]

For nontrivial \(C_r\), exact compatibility exists iff the ordered eigenvalues
\(\mu_1\le\mu_2\le\mu_3\) satisfy

\[
\mu_1>0,\qquad \mu_2=1.
\]

Both \(\kappa=\pm1\) branches are returned.  Each result carries the raw
outer-product residual, orthogonality residual and determinant of \(R\).

### Austenite / one martensite variant

\[
R U_i-I=b\otimes m.
\]

The adapter returns both exact branches whenever \(\lambda_2=1\) within the
explicit numerical tolerance.  Rounded literature data that do not satisfy
this condition are **not** projected silently to compatibility.

### Martensite / martensite

Every ordered pair \(i\ne j\) is tested independently:

\[
R U_i-U_j=a\otimes n.
\]

This includes the uncommon cases where the rank-one equation has solutions
although the pair is not connected by a parent 180° rotation.

## Independent Mallard-law audit

Whenever a proper parent twofold \(Q\) satisfies

\[
U_i=QU_jQ^T,
\]

Type-I and Type-II Mallard solutions are calculated independently from the
closed-form formulas reproduced by James & Hane (2000, Eqs. 16–17).  These are
then matched against the general Proposition-1 solutions by rotation and
rank-one outer product.  Mallard law is therefore a **cross-check**, not the
source of the general M/M solution family.

If multiple distinct parent twofolds generate the same compatible branch, the
branch is marked as compound-by-multiple-twofolds while preserving all symmetry
indices.

The physical twin shear is reported as

\[
s=\lVert U_j^{-T}n\rVert\,\lVert a\rVert.
\]

The corresponding normal in the deformed martensite configuration is also
reported.

## Cofactor conditions

For every M/M branch written

\[
R U_i-U_j=a\otimes n,
\]

the cofactor conditions are evaluated using \(U=U_j\):

\[
\mathrm{CC1}:\quad \lambda_2-1=0,
\]

\[
\mathrm{CC2}:\quad
a\cdot U\,\operatorname{cof}(U^2-I)n=0,
\]

\[
\mathrm{CC3}:\quad
\operatorname{tr}(U^2)-\det(U^2)
-\frac{|a|^2|n|^2}{4}-2\ge0.
\]

The **minus sign** in CC3 follows Theorem 2 and its proof in Chen, Srivastava,
Dabade & James (2013).  The abstract of that paper contains an inconsistent
plus sign; this repository follows the theorem/proof, not the abstract typo.

The raw CC1 residual, CC2 value and CC3 margin are always retained.  Boolean
flags are only tolerance classifications.  A sampled check of the middle
singular value of

\[
U+f a\otimes n,\qquad 0\le f\le1,
\]

is reported as an independent numerical audit; it does not replace CC1–CC3.

## James–Hane Cu–Al–Ni benchmark

The project-bound adapter consumes the existing DO3→6M reference project.  The
full parent crystallographic group remains 48 operations, while the physical
proper subgroup contains 24 rotations.  The independent nonlinear-elasticity
orbit is required to reproduce the 12 cube-edge 6M stretch variants described
by James & Hane.

The rounded measured lattice parameters are analyzed as supplied.  They are not
silently changed to the exact compatibility angle.

## Public API

```python
from cualni_cryst.ball_james_adapter import BallJamesAdapter

report = BallJamesAdapter(project, "do3_to_6m_reference").analyze()
```

The report contains:

- base stretch \(U_0\);
- generalized metric spectrum and residuals;
- every distinct stretch variant with symmetry provenance;
- all exact A/M rank-one branches;
- all exact ordered M/M rank-one branches;
- Mallard Type-I/II matches and compound classification;
- twin shear and deformed twin-plane normal;
- CC1, CC2 and CC3 numerical margins for every M/M branch;
- numerical audits and JSON-safe serialization.


## PTCLab-style crystal-first input

The numerical kernel accepts metrics and symmetry matrices, but ordinary users do not
need to type those backend objects.  The public convenience layer takes the same
high-level crystal information used in PTCLab: two lattice cells, their conventional
point groups, and the lattice correspondence.  It derives the metrics, full symmetry
groups and proper rotation subgroups internally and validates that each point group
actually preserves the supplied metric.

```python
from cualni_cryst.ball_james_adapter import BallJamesCrystalInput
from cualni_cryst.correspondence import Correspondence
from cualni_cryst.lattice import Lattice

parent = Lattice.cubic(5.836, length_unit="angstrom")
product = Lattice.monoclinic_unique_b(4.430, 5.330, 12.79, 95.68, length_unit="angstrom")

report = BallJamesCrystalInput(
    parent_lattice=parent,
    product_lattice=product,
    correspondence=Correspondence(C),
    parent_point_group="m-3m",
    product_point_group="2/m",
).analyze()
```

Thus the future GUI can expose only `a,b,c,alpha,beta,gamma`, symmetry and
correspondence, while the advanced pane shows `M_A`, `M_M`, full/proper symmetry,
`U_i`, rank-one branches, cofactor margins and numerical residuals.  Nonstandard
crystallographic settings remain supported through `ProjectState` with explicit
symmetry matrices rather than being silently coerced into a conventional setting.

## Scientific boundaries

- No CT module is imported.
- No CMC/SMC result is consumed.
- No PTMC result is consumed.
- Numerical tolerance is not experimental uncertainty.
- Absence of an exact solution for rounded measurements is reported as such.
- The adapter does not rank transformation theories or claim experimental
  correctness; it produces independent predictions for later comparison.

## Primary references

1. J. M. Ball and R. D. James, *Fine phase mixtures as minimizers of energy*,
   Arch. Rational Mech. Anal. **100** (1987) 13–52.
2. R. D. James and K. F. Hane, *Martensitic transformations and shape-memory
   materials*, Acta Materialia **48** (2000) 197–222.
3. X. Chen, V. Srivastava, V. Dabade and R. D. James, *Study of the cofactor
   conditions: conditions of supercompatibility between phases*, J. Mech.
   Phys. Solids **61** (2013) 2566–2587.
