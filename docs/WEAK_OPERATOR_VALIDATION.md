# Higher-order CT operators and axial weak twins

## Scope

This milestone completes the **non-twofold operator → weak-twin** route while
preserving the exact Type-I/Type-II backbone.

The implementation is deliberately layered:

1. exact finite-group/operator classification;
2. exact correspondence and common-sublattice arithmetic;
3. metric-native weak-plane geometry;
4. published correspondence-conditioned blind validation;
5. higher-order CT-blind validation on NiTi.

No expected weak plane is fed into the NiTi solver.

## 1. Route selection happens at the double-coset level

For a parent operation `g_A`,

```text
order 2, mirror             -> Type I
order 2, proper twofold     -> Type II
proper order 3 / 4 / 6      -> possible axial weak route
higher-order improper       -> report only, never coerce
```

There is one essential extra rule.

A CT operator is a double coset.  A double coset may contain a higher-order
representative **and** an exact mirror/twofold representative.  If an exact
Type-I or Type-II representative exists, that operator remains a classical
exact operator.  The weak search is run only on non-identity double cosets
that contain eligible proper 3/4/6-fold operations **and no classical exact
representative**.

This prevents false "weak twin" labels caused by selecting the wrong
representative of an otherwise classical operator.

The 3/4/6 restriction is not a numerical heuristic.  It is the
three-dimensional crystallographic restriction for proper lattice rotations.

## 2. Exact higher-order CT construction

With package convention

\[
u_M=C_{M\leftarrow A}u_A,
\]

the product intercorrespondence inherited from a parent operation is

\[
C_{\mathrm{int}}
=
C_{M\leftarrow A}\,g_A\,C_{A\leftarrow M}.
\]

For an eligible higher-order proper rotation, the code:

- proves its exact matrix order;
- extracts the exact one-dimensional `+1` eigenspace;
- reduces that axis to a primitive projective integer direction;
- maps the axis through the base correspondence;
- proves that `C_int` has the same order and determinant;
- proves the product axis is invariant under `C_int`;
- only then enters the weak-plane search.

The weak-plane search remains the already-hardened exact CT-constrained search

\[
p_2\parallel C_{\mathrm{int}}^{-T}p_1,
\qquad p_1^T u=0.
\]

## 3. Generalized twin index

For rational

\[
C=A/d,
\qquad A\in\mathbb Z^{3\times3},
\]

the domain common-sublattice index is calculated exactly from the Smith normal
form of `A`:

\[
q_{\rm domain}
=
\prod_i \frac{d}{\gcd(d,|s_i|)}.
\]

The inverse correspondence is audited independently to obtain the codomain
index.

For an equal-volume same-lattice twin,

\[
|\det C|=1,
\]

the two indices must agree and their common value is reported as Cayron's
generalized twin index `q_g`.

For a rational map with `|det C| != 1`, the code **does not** call either index
a generalized twin index.  It returns both common-sublattice indices and
`q_g=None`.

This makes the API meaningful beyond the special twin case without
over-generalizing the physics.

## 4. Generalized strain, generalized shear and weak-plane geometry

A positive-definite metric `M` is represented physically by a basis `B` with

\[
B^TB=M.
\]

A same-phase correspondence and distortion become

\[
\widehat C=BCB^{-1},
\qquad
\widehat F=BFB^{-1}.
\]

Cayron's generalized strain is

\[
\varepsilon_g^2
=
\operatorname{tr}(MCM^{-1}C^T)-3
=
\|\widehat C\|_F^2-3.
\]

The generalized shear is

\[
s_g
=
\|\widehat F-I\|_F.
\]

The new geometry report additionally SVD-decomposes

\[
H=\widehat F-I.
\]

It reports the best rank-one residual.  Therefore a conventional simple shear
and a genuinely non-rank-one weak distortion are not conflated.

### Intrinsic weak-plane distortion

For the source weak plane, an orthonormal physical in-plane basis `E` is built
from the invariant axis and its transverse in-plane direction.  The two
singular values of

\[
\widehat F E
\]

measure intrinsic in-plane length/angle distortion independently of
Miller-index scaling.

The ranking score is

\[
d_{\rm plane}
=
\sqrt{(\sigma_1-1)^2+(\sigma_2-1)^2}.
\]

This gives a simple metric-native notion of "weakness" and is invariant under
crystallographic basis changes.

## 5. Published Mg blind regressions

Source:

C. Cayron, *The concept of axial weak twins*, Acta Materialia **236** (2022)
118128, DOI `10.1016/j.actamat.2022.118128`.

The benchmark is **correspondence-conditioned blind**:

```text
VISIBLE TO SOLVER:
    Mg metric
    exact printed correspondence C
    invariant axis

HIDDEN UNTIL COMPARISON:
    weak plane pair
    q_g
    s_g
    epsilon_g
```

The search recovers:

```text
a-axis basal/prismatic:
q_g = 2
s_g ~ 0.092
epsilon_g ~ 0.130

a-axis basal/pyramidal:
q_g = 4
s_g ~ 0.107
epsilon_g ~ 0.137

second a-axis sister:
plane pair corresponding to (01-11)||(01-14)
q_g = 2
s_g ~ 0.118

conventional extension sister:
(01-12)||(01-12)
q_g = 2
s_g ~ epsilon_g ~ 0.130

a+2b weak twin:
(-2110)||(-2116), projectively represented in the 3-index code
q_g = 4
s_g ~ 0.088
epsilon_g ~ 0.113
```

### Source inconsistency retained, not hidden

For the second a-axis sister, the paper text reports `epsilon_g ~ 0.137` while
printing the same correspondence matrix as the extension-family
`epsilon_g ~ 0.130` cases.

Because Eq. (6)/(7) makes `epsilon_g` a function of `C` and the metric, the
printed correspondence gives

```text
epsilon_g = 0.130086...
```

The benchmark records both facts but **does not use the conflicting source
epsilon as a pass/fail target**.  No expected value is edited merely to obtain
a green test.

## 6. Higher-order NiTi blind validation

Source:

C. Cayron, *The Correspondence Theory and Its Application to NiTi Shape
Memory Alloys*, Crystals **12** (2022) 130,
DOI `10.3390/cryst12020130`.

Input is only the already locked B2/B19' metrics, point groups and base
correspondence.

The code must:

1. build the exact CT correspondence groupoid;
2. identify non-classical double cosets without using operator numbers;
3. find eligible proper higher-order elements;
4. derive the parent and product rational axes;
5. derive `C_int`;
6. enumerate weak planes;
7. rank them by intrinsic in-plane distortion;
8. only then compare with Cayron's published result.

The blind target is

\[
(1\bar 3 3)_{B19'}
\parallel
(31\bar 1)_{B19'},
\qquad
[011]_{B19'},
\]

with approximately

```text
q_g = 2
s_g = 0.2911
epsilon_g = 0.2912
reticular misorientation = 84 deg
```

The two looser published alternatives

```text
(1-11)||(11-1)
(0-11)||(100)
```

must also occur in the same source-consistent higher-order search.

## 7. Generality and failure semantics

The hardening tests additionally cover:

- every symmetry element in all 32 crystallographic point groups;
- exact matrix-order classification;
- exact Smith-normal-form index arithmetic;
- equal-volume vs non-equal-volume rational maps;
- metric-native strain/shear geometry;
- rank-one/simple-shear diagnostics;
- exact unimodular basis covariance of weak-twin observables;
- improper higher-order operations;
- singular correspondences;
- incompatible metrics;
- non-crystallographic fivefold rotations.

"General" therefore means:

> any mathematically valid finite three-dimensional crystallographic input is
> processed according to the route its exact symmetry supports; invalid,
> non-crystallographic, physically ambiguous or theory-unsupported inputs are
> rejected or explicitly labelled rather than silently coerced.

## Scientific boundary: generic GenOVa enumeration

This milestone does **not** claim a byte-for-byte reconstruction of every
undocumented internal candidate-ordering choice in GenOVa's generic
A/B/C/D supercell search.

The published equations and the CT-constrained problem needed for
Correspondence Theory are implemented and tested.  A future exact GenOVa
clone would require a separately sourced specification of its complete
supercell enumeration and tie-breaking rules.
