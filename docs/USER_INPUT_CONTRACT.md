# User Input Contract — professor-ready hardening v1

## Goal

A normal crystallographer works with familiar crystallographic quantities:

- phase name;
- unit-cell lengths `a, b, c`;
- angles `alpha, beta, gamma`;
- point group;
- lattice correspondence;
- plane/direction parallelisms.

The user does **not** need to know the internal metric tensor, choose a Cartesian
embedding, or know the repository's internal matrix direction convention.

The backend remains metric-native internally and cross-checks every supported
Cartesian convention automatically.

## Non-negotiable safety rule: no bare correspondence matrix

A matrix alone is scientifically ambiguous.

The facade accepts either:

```python
CorrespondenceInput.from_matrix(
    matrix,
    matrix_maps="parent_to_product",
)
```

or:

```python
CorrespondenceInput.from_matrix(
    matrix,
    matrix_maps="product_to_parent",
)
```

The safer alternative is three exact basis mappings:

```python
CorrespondenceInput.from_basis_mappings(
    [
        BasisVectorMapping(parent=(...), product=(...)),
        BasisVectorMapping(parent=(...), product=(...)),
        BasisVectorMapping(parent=(...), product=(...)),
    ]
)
```

The program derives the exact matrix itself.

## Internal convention

After compilation:

\[
u_M = C_{M\leftarrow A}u_A
\]

and reciprocal plane covectors obey:

\[
p_M = C_{M\leftarrow A}^{-T}p_A.
\]

Users do not need to use these symbols in the normal interface.

## Cartesian conventions

The normal interface exposes none.

Every compiled transformation is cross-checked under every supported pair of
Cartesian embeddings. Physical principal stretches must be invariant and each
embedding must pass the direct/reciprocal/deformation parity audit.

If those representations disagree, the calculation fails instead of choosing
one silently.

## Exactness policy

- integers and `p/q` strings are exact;
- decimal strings/floats preserve the supplied decimal;
- `0.333333` is **not** silently changed to `1/3`;
- singular matrices are rejected;
- dependent basis mappings are rejected;
- contradictory verification mappings are rejected before theory runs;
- direct-space vectors and reciprocal-space planes remain distinct.

## GUI contract

Future GUI code calls this facade. It must not:

- calculate a metric tensor itself;
- transpose/invert a correspondence heuristically;
- implement CT/PTMC/Ball-James equations;
- select a Cartesian convention to make a result agree;
- silently rationalize decimals;
- hide a failed representation audit.


## Handedness contract

`Correspondence` as a low-level algebraic object may represent any invertible
lattice map. The physical transformation workflow is stricter: `det(F) > 0`.

Every Cartesian lattice embedding used by the package is right-handed, so

`sign(det(F)) = sign(det(C_M_from_A))`.

A negative correspondence determinant is therefore rejected at the normal-user
boundary and again by `RepresentationBridge`. The software never silently flips
an axis to make an improper map pass.
