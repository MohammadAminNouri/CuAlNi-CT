# Experimental EBSD validation and adversarial hardening

## Objective

The EBSD engine already provides convention-safe ingestion, phase-aware grain
segmentation, KAM/GOS, parent reconstruction, variant assignment and
matrix-based operator classification.

This milestone adds the **independent experimental-validation layer**.

The goal is not to make every input produce an answer.  The goal is:

> any mathematically interpretable EBSD input must either produce an auditable
> crystallographic result or a precise statement of what is invalid,
> ambiguous, convention-dependent, under-resolved or unidentifiable.

That distinction is essential for PTCLab-style scientific software.

## 1. Theory is generated before experiment is scored

For a supplied physical OR

\[
x_A=R_{A\leftarrow M}x_M,
\]

parent symmetry generates raw variants

\[
R_i=S_A^{(i)}R_{A\leftarrow M}.
\]

Two raw variants are the same product orientation when

\[
R_iS_M=R_j
\]

for a proper product symmetry \(S_M\).

The variant enumerator therefore performs the right product-symmetry quotient
and independently cross-checks the count with the metric-native
`OrientationKernel` intersection subgroup

\[
H_T^A=G_A^+\cap R G_M^+R^{-1},
\qquad
N_T=\frac{|G_A^+|}{|H_T^A|}.
\]

Experimental EBSD orientations are never used to manufacture the theoretical
variant library.

## 2. Full OR quotient distance

Two OR matrices are equivalent when

\[
R'=S_A R S_M^{-1}.
\]

The implementation compares ORs through

\[
\Delta_{\rm OR}
=
\min_{S_A\in G_A^+,\,S_M\in G_M^+}
\operatorname{angle}
\left(S_A R_1S_MR_2^T\right).
\]

This is used when judging a fitted experimental OR against an initial or
literature OR.

## 3. Segmentation is a sensitivity parameter, not a measured constant

No single grain threshold is declared physically true.

For every requested threshold the code records:

- number of reconstructed grains;
- retained indexed fraction;
- median GOS;
- 95th-percentile GOS;
- pixel partition.

Consecutive partitions are compared by the Adjusted Rand Index.  A stable
plateau is evidence that the map partition is insensitive over that range; it
does not convert the threshold itself into a material constant.

This follows the parent-reconstruction literature, where the initial child
segmentation is an analysis choice and affects later reconstruction.

## 4. Internal theory-consistency score

For every neighbouring child-grain pair, two independent residuals are
calculated.

### Operator residual

The measured product/product relation

\[
\Delta=g_i^Tg_j
\]

is compared on SO(3) with the complete theoretical boundary-operator orbit
under proper product symmetry.

### Candidate-parent residual

Each child orientation independently gives the parent candidate set

\[
g_A^{(k)}
=
g_M
\left(R_{A\leftarrow M}^{(k)}\right)^T.
\]

For a child pair, the minimum parent-symmetry disorientation between the two
candidate sets is reported.

The consistency score stores the complete residual arrays and their robust
median / 90th-percentile summaries.  The scalar combined score is only a
ranking statistic; it is **not** called a probability or confidence.

## 5. Convention hypotheses: diagnosis without automatic correction

The engine can compare explicit user-supplied convention hypotheses against the
same transformation theory.

Examples:

- crystal-to-sample vs sample-to-crystal;
- degrees vs radians after explicit raw-data conversion;
- alternative documented crystal-frame corrections;
- vendor/reference-frame transforms.

A hypothesis is declared internally distinguishable only when one candidate
wins by a declared residual gap.

### Exact non-identifiability

If every orientation is left-multiplied by one common sample rotation

\[
g_i' = Q_{\rm sample}g_i,
\]

then

\[
(g_i')^Tg_j'
=
g_i^Tg_j.
\]

Therefore child/child crystallography **cannot determine the absolute common
sample rotation**.

The adversarial suite explicitly tests that the software returns a tie in this
case.  Claiming to detect such a rotation from internal misorientations would be
a mathematical error.

An external specimen direction, detector/stage calibration, surface trace or
other sample-frame observable is required.

## 6. Experimental OR refinement

The supplied OR remains a hypothesis.  It may be refined from child/child
grain-boundary data without using reconstructed parent orientations.

For a small correction vector \(\omega\),

\[
R(\omega)=\exp([\omega]_\times)R_0.
\]

At each trial OR the program regenerates the complete variant and operator
library and calculates the measured child-boundary residuals.  A robust Huber
loss is evaluated on a declared lower fraction of the residual distribution.

This implements the same scientific principle as the iterative
parent-to-child OR fitting introduced by Nyyssönen and used in generic MTEX
parent reconstruction: transformation-related child/child boundaries constrain
the OR.

The fit is bounded, deterministic, multi-start, and reports:

- initial/fitted OR;
- raw correction angle;
- symmetry-reduced OR change;
- initial/fitted residual distributions;
- robust objective improvement;
- optimizer status.

A fit is never silently substituted for the initial OR.

## 7. Candidate-level variant graph

A graph node is

\[
(\text{child grain},\text{candidate parent variant}).
\]

Nodes belonging to neighbouring child grains are linked only when their parent
candidate orientations agree within the declared parent-symmetry tolerance.

Connected candidate components are reconstructed independently.  Their actual
reconstruction inliers form parent-domain candidates.

If one child grain belongs to multiple valid parent-domain candidates, the
grain is marked ambiguous.  The algorithm does **not** force a unique merge.

This is deliberately aligned with the variant-graph philosophy in:

- R. Hielscher, T. Nyyssönen, F. Niessen & A. A. Gazder,
  *The variant graph approach to improved parent grain reconstruction*,
  Materialia 22 (2022) 101399, DOI 10.1016/j.mtla.2022.101399.
- F. Niessen, T. Nyyssönen, A. A. Gazder & R. Hielscher,
  *Parent grain reconstruction from partially or fully transformed
  microstructures in MTEX*, J. Appl. Cryst. 55 (2022) 180–194,
  DOI 10.1107/S1600576721011560.

Our implementation remains independent and uses the package's own SO(3),
symmetry and CT machinery.

## 8. Plane-trace validation

An EBSD orientation assigns a crystallographic plane normal to the sample frame.

For sample surface normal \(n_s\) and transformed plane normal \(n_p\),

\[
t_{\rm predicted}
\propto
n_s\times n_p.
\]

The measured grain-pair boundary trace is estimated from the physical
midpoints of neighbour edges and a PCA in the specimen surface.

The report includes:

- number of boundary edges;
- number of unique trace points;
- boundary span;
- line-shape / linearity ratio.

A strongly curved boundary is not silently compared with one infinite straight
crystallographic trace.

### Two-sided twin check

For a CT twin, if

\[
p_2\propto C_{\rm int}^{-T}p_1,
\]

both crystal sides independently predict a sample-frame plane and trace.

A hypothesis is scored by:

1. side-1 trace residual;
2. side-2 trace residual;
3. physical sample-plane-normal coherence.

The worse trace residual is retained so one good side cannot hide one bad side.

## 9. Harsh adversarial campaign

The frozen campaign includes deterministic perturbations:

- orientation noise;
- random missing/unindexed measurements;
- random orientation outliers;
- wrong active/passive matrix direction;
- wrong crystal-frame rotation;
- global sample-frame rotation.

Required behavior is asymmetric:

```text
global sample rotation:
    MUST remain internally unidentifiable

wrong crystal frame / active-passive:
    SHOULD degrade fixed-theory consistency unless symmetry-equivalent

missing points:
    MUST reduce evidence, never synthesize orientations

outliers:
    MUST appear as residual/inlier failures rather than being fit exactly
```

This distinction is more important than simply making every corruption produce
an error message.

## 10. What remains external

No EBSD-only algorithm can independently infer all of the following in every
experiment:

- chemical phase identity;
- absolute detector/specimen reference frame;
- calibrated angular uncertainty from CI/IQ/MAD/BC/BS alone;
- a physical OR when only one child variant is present;
- a unique trace plane from a strongly curved/under-resolved boundary;
- the sample surface normal if it is not provided by geometry.

The engine explicitly exposes these as required metadata or
non-identifiabilities.

That is the intended meaning of a general crystallographic program: not
"always return a number", but "return the physically justified result and
state exactly what the data cannot determine."

## Performance hardening without reducing the mathematics

The first experimental-validation implementation was intentionally literal and
therefore too slow for routine use.  The bottleneck was not the optimizer
itself; each OR objective evaluation rebuilt full operator classes and then
repeated a redundant four-level product-symmetry quotient for every boundary.

The optimized backend keeps the **same scientific search** and reduces only
algebraic redundancy.

For measured child rotation \(\Delta\), theoretical operator \(Q\) and proper
product symmetry group \(G\), the reference calculation was

\[
\min_{S_1,S_2,A,B\in G}
\theta\!\left(
S_1^T\Delta S_2
\left(A^T Q B\right)^T
\right).
\]

Using group closure and cyclic invariance of the trace of an SO(3) rotation,

\[
\operatorname{tr}
\left(
S_1^T\Delta S_2 B^TQ^TA
\right)
=
\operatorname{tr}
\left(
L\Delta KQ^T
\right),
\]

where \(L=A S_1^T\in G\) and \(K=S_2B^T\in G\).

The complete quotient is therefore exactly

\[
\boxed{
\min_{L,K\in G}
\left\{
\theta(L\Delta KQ^T),
\theta(L\Delta KQ)
\right\}.
}
\]

The second branch is the inverse/transposed operator.  Thus the original
four group loops collapse to two **without dropping any symmetry-equivalent
state**.

All \(L\Delta K\) matrices for every measured boundary are now computed once.
They are stored as unit quaternions.  Candidate theoretical operators are also
converted to unit quaternions, and the geodesic angle uses

\[
\theta
=
2\operatorname{atan2}
\left(
\sqrt{1-|q_1\!\cdot q_2|^2},
|q_1\!\cdot q_2|
\right).
\]

The absolute quaternion dot product handles the \(q\sim -q\) double cover and
the `atan2` expression retains stability close to both 0 and 180 degrees.

### OR-refinement objective

The objective needs only the minimum residual over all physical variant pairs.
It therefore does **not** need to partition those pairs into named operator
classes on every trial OR.

For every trial OR the code still:

1. regenerates the complete parent-symmetry variant set;
2. performs the full right product-symmetry quotient;
3. constructs every physical variant-pair operator;
4. compares every measured boundary through the full quotient above;
5. applies the same trimming, Huber loss, bounded correction and seven
   deterministic Powell starts.

Full operator classes are rebuilt for the final fitted OR and for reporting.

This is a change in computational architecture, not in the objective.

### Reference-equivalence tests

The hardening suite now explicitly compares:

- optimized operator residuals vs the frozen slow
  `classify_boundary_operator()` implementation;
- direct variant-pair objective residuals vs the same frozen class-based
  reference;
- optimized candidate-parent compatibility vs the original
  symmetry-reduced definition;
- optimized and reference OR variant counts for all 32 crystallographic point
  groups.

The scientific test fails if the optimized path disagrees with the slower
reference formulation beyond roundoff-scale tolerance.

No test case, symmetry operation, optimizer start, OR correction range,
expected scientific result, or adversarial corruption was removed to gain
speed.
