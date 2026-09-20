# Theory map and mathematical contracts

This document is the compact scientific specification for the independent
calculation branches in CuAlNi-CT.  It deliberately separates correspondence,
orientation, deformation, stretch, twinning, classical PTMC, cofactor theory,
and experimental orientation data.

Agreement between two branches is something to **test**, not something to
assume.

## 0. Coordinate convention and typed objects

For parent phase `A` and martensite phase `M`, direct crystallographic
coordinates and reciprocal plane covectors are different mathematical objects.

The package correspondence convention is

\[
\boxed{u_M=C_{M\leftarrow A}u_A}.
\]

Therefore plane covectors obey the dual law

\[
\boxed{p_M=C_{M\leftarrow A}^{-T}p_A}.
\]

For a direct metric tensor \(M\),

\[
\|u\|^2=u^TMu,\qquad
\|p\|_*^2=p^TM^{-1}p.
\]

If \(B\) is a Cartesian realization of the crystallographic basis,

\[
x=Bu,\qquad n\propto B^{-T}p,\qquad B^TB=M.
\]

Thus `(hkl)` and `[hkl]` must never be identified numerically in a
non-cubic lattice.

The code keeps separate:

- `C`: lattice correspondence;
- `T` or physical `R`: orientation relationship;
- `F`: deformation/distortion gradient;
- `U`: positive symmetric stretch from a polar decomposition.

No code path may silently assume

\[
C=T,\qquad T=R_{\rm polar},\qquad F=U
\]

or

\[
H_C=H_T=H_F=H_U.
\]

## 1. Full crystallographic symmetry versus proper rotations

For cubic `m-3m` the crystallographic point group contains exactly **48**
operations:

- 24 proper rotations, determinant \(+1\);
- 24 improper operations, determinant \(-1\).

The full 48 operations are required for Cayron correspondence topology because
parent mirrors are genuine symmetry elements and are the generators of CT
Type-I transformation twins.

The 24 proper operations are used when a calculation genuinely lives in
\(SO(3)\), for example physical orientation matrices and EBSD disorientations.

The program therefore maintains both sets.  It must never replace the full
crystallographic group by the proper subgroup in a CT coset/double-coset
calculation.

For a relation represented by a matrix \(X\), the corresponding parent
intersection subgroup has the generic form

\[
H_X^A=G_A\cap X^{-1}G_MX,
\]

with the exact placement of \(X\) determined by the mapping convention.  For
the package correspondence \(C=C_{M\leftarrow A}\),

\[
\boxed{H_C^A=G_A\cap C^{-1}G_MC}.
\]

Correspondence variants are left cosets

\[
V_i^C=g_iH_C^A,\qquad
N_C=\frac{|G_A|}{|H_C^A|},
\]

and intervariant correspondence classes are double cosets

\[
\boxed{O_k^C=H_C^Ag_kH_C^A}.
\]

The complete double coset is the invariant object; a minimum-angle
disorientation is only one representative/label.  Variant/operator structure
is a groupoid, not in general a group.

## 2. Natural OR, closing-gap OR, and polar rotation

Cayron CT does **not** provide a universal black-box map

\[
(M_A,M_M,C)\mapsto T_{CT}
\]

that predicts one unique physical OR for every transformation.

The 2022 CT construction assumes a **natural OR**.  For a particular
intercorrespondence/operator, compatibility can require a small operator-
specific deviation, yielding a **closing-gap OR**.

The program must therefore distinguish at least:

- `NaturalOR`;
- `ClosingGapOR(TypeI, operator)`;
- `ClosingGapOR(TypeII, operator)`;
- `WeakTwinOrientation(operator)`;
- `PolarOR`;
- `BallJamesOR`;
- `PTMCOR`;
- `ExperimentalOR` / `EBSDOR`.

A polar rotation is a useful comparator.  It is not automatically Cayron's
natural OR.

## 3. Cayron Type-I and Type-II transformation twins

For a parent symmetry \(g_A\),

\[
C_{\rm int}=Cg_AC^{-1}.
\]

If \(g_A\) is a parent reflection on \(p_A\), CT gives a Type-I candidate.
The rational mirror plane is inherited by correspondence,

\[
\boxed{K_{1,M}=C^{-T}p_A}.
\]

The complementary shear direction and shear amplitude depend on the
martensite metric.

If \(g_A\) is a proper parent 180-degree rotation about \(a_A\), CT gives a
Type-II candidate with rational direction

\[
\boxed{\eta_{2,M}=Ca_A}.
\]

The complementary \(K_2\) plane and shear amplitude depend on the martensite
metric.

Compound twins are special relations possessing both Type-I and Type-II
character.  Operators lacking a relevant two-fold symmetry must not be forced
into a classical exact twin classification.  Cayron's axial weak-twin
framework treats such cases through explicit length/angular misfit criteria.

## 4. CMC and exact single-variant A/M compatibility

With \(u_M=Cu_A\),

\[
\boxed{CMC=C^TM_MC-M_A}.
\]

A parent direction preserves length exactly when

\[
\boxed{u_A^TCMC\,u_A=0}.
\]

After parent-metric whitening, let the CMC eigenvalues be
\(q_1,q_2,q_3\).  Exact A/M IPS compatibility is the degeneracy condition

\[
\boxed{q_i=0,\qquad q_jq_k\le0}.
\]

The degeneracy orders are retained explicitly:

- first order: one zero eigenvalue and opposite signs for the other two;
- second order: two zero eigenvalues;
- third order: all three zero.

A nearest-zero projection for measured lattice parameters is only a
diagnostic.  Its nonzero residual must always be reported.

## 5. SMC

SMC means **shear by metric correspondence**.  It is neither a reciprocal CMC
nor \(CMC^{-1}\).

With package convention \(C=C_{M\leftarrow A}\),

\[
\boxed{
SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}
}.
\]

For an exact CMC habit-plane covector \(m_A\),

\[
\boxed{d_A=SMC\,m_A}.
\]

Here \(m_A\) is a reciprocal covector and \(d_A\) is a direct-space vector.
For a non-cubic parent the direct unit normal is

\[
n_A=M_A^{-1}m_A
\]

after the appropriate reciprocal normalization.

## 6. Cayron A/M/M shear-shear supercompatibility

Let \(m_A\) be the normalized A/M habit-plane covector, \(d_A\) the A/M IPS
shear vector, \(n\) the direct unit normal of the M/M twin plane, and \(a\)
the M/M twin shear vector, all represented in the parent frame.  Cayron's
intercompatibility condition is

\[
\boxed{2(m_A^Tn)d_A=a}.
\]

The program reports the residual, rather than a yes/no claim alone.

The supported logical implication is that the CT A/M compatibility, M/M twin
compatibility and shear/shear condition imply the cofactor/supercompatibility
conditions in Cayron's construction.  The repository does **not** assume that
full CT and PTMC/cofactor theory are formally equivalent in both directions.

## 7. Metric-native bridge to the transformation stretch

Define

\[
G_C=C^TM_MC.
\]

In the symmetric parent metric-whitened orthonormal basis,

\[
\widehat G=M_A^{-1/2}G_CM_A^{-1/2}.
\]

The positive transformation stretch satisfies

\[
\boxed{U^2=\widehat G}.
\]

Equivalently, directly in crystallographic coordinates one solves the
generalized metric eigenproblem

\[
\boxed{G_Cv_i=\mu_iM_Av_i},
\qquad
\lambda_i=\sqrt{\mu_i}.
\]

The eigenvectors can be chosen \(M_A\)-orthonormal:

\[
V^TM_AV=I.
\]

The normalized CMC is

\[
D=M_A^{-1/2}CMC\,M_A^{-1/2}=U^2-I,
\]

so

\[
\boxed{q_i=\lambda_i^2-1}.
\]

This exact bridge is now tested both through whitening and independently
through the generalized eigenproblem, which is particularly important for
non-cubic parent cells.

## 8. Ball-James rank-one compatibility and Mallard law

Two deformation gradients are compatible across a planar interface if there
exists \(R\in SO(3)\) and vectors \(a,n\) such that

\[
\boxed{RF_1-F_2=a\otimes n}.
\]

For single-variant martensite against austenite,

\[
RU-I=b\otimes m.
\]

For a positive-definite three-dimensional stretch, exact single-variant A/M
compatibility requires

\[
\boxed{\lambda_2=1}.
\]

For symmetry-related stretches \(U_j=QU_iQ^T\), where \(Q\) is an appropriate
parent two-fold symmetry, Mallard's law gives independent Type-I/II rank-one
twin solutions.  Those calculations remain algorithmically independent of
the CT twin formulas.

## 9. Cofactor conditions

For the specified twin system \((a,n)\), Chen-Srivastava-Dabade-James give

\[
\boxed{CC1:\ \lambda_2=1},
\]

\[
\boxed{CC2:\ a\cdot U\,\operatorname{cof}(U^2-I)n=0},
\]

and

\[
\boxed{
CC3:\ \operatorname{tr}(U^2)-\det(U^2)
-\frac{|a|^2|n|^2}{4}-2\ge0
}.
\]

**The sign in front of \(\det(U^2)\) is minus.**

Under the theorem hypotheses these conditions are necessary and sufficient
for the crystallographic-theory equations to admit a solution for every twin
volume fraction \(f\in[0,1]\).  Therefore \(|\lambda_2-1|\) alone is not a
full cofactor test.

## 10. Classical PTMC laminate branch

A twin relation must include its relative rotation, for example

\[
\boxed{\widehat R U_2-U_1=a\otimes n}.
\]

Once the compatible pair is placed in one common reference frame, the
laminate average can be written

\[
\overline F(f)=U_1+f\,a\otimes n,\qquad 0\le f\le1.
\]

The austenite/laminate habit-plane problem is then

\[
\boxed{R\overline F(f)-I=b\otimes m}.
\]

For generic metrics only discrete volume fractions satisfy the habit-plane
condition.  Under the cofactor conditions compatibility extends to every
fraction.

The PTMC implementation must not call CMC/SMC to manufacture its answer.

## 11. Cu-Al-Ni is not one fixed parameter set

James-Hane is an essential benchmark, not the definition of Cu-Al-Ni.

Published Cu-Al-Ni work contains, depending on composition, order, heat
treatment, stress and temperature, several martensitic states and several cell
representations, including:

- long-period monoclinic beta-prime martensite described by 18R/M18R,
  reduced/6M-like, or conventional monoclinic cells;
- orthorhombic gamma-prime / 2H martensite;
- reported L21/DO3/B2-like parent descriptions in different contexts;
- source-specific axis settings and lattice-parameter conventions.

Every literature state must keep its own:

1. composition;
2. temperature/stress/history;
3. physical phase identity;
4. space/point group;
5. exact cell setting and basis;
6. lattice parameters and uncertainty if available;
7. correspondence \(C\), only when independently established;
8. OR, only when measured/proposed;
9. twin/habit observations;
10. provenance status.

Two cells may represent the same physical lattice only after an explicit basis
change \(P\) is established and audited:

\[
M_{\rm new}=P^TM_{\rm old}P.
\]

Directions, reciprocal planes, correspondence matrices, ORs and twin indices
must all be transformed consistently.  Similar numbers are not proof of cell
equivalence.

## 12. Comparison rule

All theories receive the same sourced physical state and remain independent:

```text
same sourced crystal state
    |-- Cayron CT: C -> H_C -> variants/operators -> twins -> CMC/SMC
    |-- Ball-James: U_i -> rank-one connections
    |-- Mallard: symmetry-related U_i/U_j -> Type-I/II solutions
    |-- classical PTMC: compatible laminate -> f, habit plane, shape strain
    |-- cofactor: CC1/CC2/CC3
    |-- polar decomposition: R_polar comparator
    `-- experiment: EBSD OR/operator/trace/habit observations
```

The comparison layer reports separate residuals.  It must not create a single
artificial "winner score".
