# Blind EBSD OR discovery

This milestone tests the inverse problem without supplying the answer.

The production solver receives only:

```text
parent lattice + point group
product lattice + point group
measured product-grain orientations
product-grain adjacency
```

It does **not** receive:

```text
literature OR
true OR
expected variant IDs
expected operator classes
expected parent orientation
CT correspondence
oracle result
```

## Mathematics

For a physical OR \(R=R_{A\leftarrow M}\),

\[
R_i=S_A^{(i)}R
\]

and therefore

\[
Q_{ij}
=
R_i^T R_j
=
R^T (S_A^{(i)})^T S_A^{(j)} R.
\]

The observed child/child rotation is compared under the complete proper product
symmetry quotient.  The exact final objective is the same frozen
variant-pair/operator machinery used by the experimental-validation layer.

The low-discrepancy global scan is only a proposal mechanism.  It uses
conjugated parent operations to locate promising regions of SO(3).  No result
is accepted from this relaxed stage.

Every retained seed is reranked and optimized with:

- complete parent/product variant enumeration;
- exact product-symmetry quotient;
- every physical variant-pair operator;
- trimmed Huber boundary objective.

## Identifiability

The code distinguishes four outcomes:

```text
identified
ambiguous
insufficient_evidence
inconsistent
```

A small residual alone is not enough to report `identified`.

The best OR must also have sufficient boundary support and more than one
accepted operator class, and it must be separated from inequivalent local
minima in the full parent/product OR quotient.

## Sealed campaign

`tools/run_ebsd_blind_campaign.py` creates hidden synthetic truth in the
evaluator process, writes only measured observations plus phase
crystallography to a challenge directory, and launches the solver in a
separate subprocess.

The oracle OR is never serialized into the challenge JSON or NPZ and is only
used after the solver subprocess exits.

The campaign covers:

- tetragonal -> orthorhombic;
- hexagonal -> orthorhombic;
- cubic -> tetragonal;
- orthorhombic -> monoclinic;
- low angular noise;
- generic random ORs rather than famous named ORs.

This is an inverse-validation campaign.  The solver cannot pass by reproducing
hard-coded literature answers.


## Critical inverse-problem gauge: parent-group normalizer

The first sealed campaign exposed a mathematically important identifiability
issue rather than a bad local minimum.

Child/child data do not observe the physical OR matrix \(R\) directly.  They
observe the conjugated parent subgroup

\[
H(R) = R^T G_A R.
\]

If \(N\) is in the SO(3) normalizer of the parent proper group,

\[
N^T G_A N = G_A,
\]

then

\[
H(NR)=R^T N^T G_A N R=R^T G_A R=H(R).
\]

Therefore \(R\) and \(NR\) are exactly indistinguishable from child/child
operator data, even when they are far apart under the ordinary physical OR
quotient.

This is not a numerical tolerance issue.  It is an exact gauge of the inverse
problem.

Important examples include:

- tetragonal proper \(D_4\): a 45-degree parent-axis rotation normalizes the
  group although it is not itself in \(D_4\);
- hexagonal proper \(D_6\): the corresponding extra gauge includes a 30-degree
  parent-axis rotation;
- orthorhombic proper \(D_2\): the SO(3) normalizer is substantially larger and
  includes permutations of the three twofold axes.

This explains why a candidate can have essentially perfect boundary residuals
yet be tens of degrees from the hidden oracle in the ordinary OR quotient.

The repaired backend never hard-codes those normalizers.  It compares the
complete finite subgroup embeddings directly:

\[
H_1 = R_1^T G_A R_1,\qquad H_2 = R_2^T G_A R_2,
\]

then minimizes a symmetric SO(3) Hausdorff distance over proper product-frame
conjugations

\[
H_2 \mapsto S_M^T H_2 S_M.
\]

The standard physical OR quotient remains available for externally specified
ORs.  It is simply not the correct equivalence relation for a blind inverse
problem based only on child/child EBSD relations.

Accordingly the solver now claims only an

```text
identified_observable_class
```

rather than pretending that child/child data uniquely determine one physical
OR representative.
