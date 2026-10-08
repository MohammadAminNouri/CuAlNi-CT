# Twin-family app scientific contract

This file states the rules the standalone app is not allowed to violate.

## 1. Input state is the only runtime source of answers

Runtime outputs are calculated from the entered parent lattice, product lattice, parent point group, product point group, correspondence, and (only when weak-plane enumeration is requested) an explicitly supplied primitive-node basis.

The production package must not identify an alloy/material and return memorized literature outputs.

## 2. Correspondence convention is explicit

Canonical backend action:

```text
u_M = C_(M<-A) u_A
```

An entered `M -> A` matrix is exactly inverted at the UI boundary. It is not transposed, approximated, or interpreted by alloy name.

## 3. Topological and metric variants are distinct

The discrete correspondence topology produces variants `M_i`. The nonlinear-elasticity metric orbit produces stretches `U_j`.

The app calculates and retains an explicit map

```text
M_i -> U_j
```

and allows many `M_i` to map to the same `U_j` when the supplied metric is degenerate.

## 4. Twin families are relation families, not a flat pair list

For an unordered correspondence pair `M_i <-> M_j`, the app keeps both directed operator classes and groups the pair by the operator/inverse-operator family. Symmetry-equivalent couples remain visible under the same family.

## 5. Classical classification must be independently cross-locked

A pair-specific exact rank-one relation alone is not enough to print a Type-I/Type-II/Compound label.

For Type I and Compound, the pair's physical `K1/eta1` and shear are compared with the independently derived discrete Type-I representation. For Type II, the exact parent twofold provenance and shear are cross-locked because the discrete Type-II representation naturally exposes conjugate `K2/eta2`, not physical `K1/eta1`.

If the routes do not agree, the output must remain **classification cross-lock unresolved**.

## 6. Compound means same physical twin, not two labels

Compound classification requires independent Type-I and Type-II descriptions of the same physical rank-one branch. The Type-I route must match physical `K1/eta1` geometry and shear; the Type-II route must match exact parent-twofold provenance and shear. The code never establishes Compound by directly equating Type-I `K1/eta1` with Type-II conjugate `K2/eta2`. Merely finding a mirror and a twofold somewhere in the same operator family is insufficient.

## 7. Weak route is separate

A complete operator family containing an exact mirror/twofold route remains classical even if it also contains a higher-order representative.

Only a genuinely higher-order family without the exact classical route may enter the weak-plane path.

Weak-plane enumeration requires an explicit primitive-node basis. The app never infers Bravais centering from the point group.

## 8. Habit planes belong to one exact twin branch

For one exact M/M branch,

```text
R U_other - U_base = a tensor n
```

the laminate path uses that exact branch and solves the A/M compatibility problem. Habit solutions are indexed by `(base stretch variant, other stretch variant, rank-one branch)`.

The app may display zero, discrete, or continuous compatible-fraction results. It must never fabricate a discrete habit plane when the exact result is “none” or a continuum.

## 9. Frames are labelled and cross-checked

The app distinguishes parent orthonormal quantities from parent/product crystallographic direct and reciprocal coordinates. Plane covectors and direct directions are never treated as the same coordinate object. Before a habit branch is exposed, the displayed parent reciprocal plane must be projectively consistent with `B_A^T m_cart`, the displayed parent direct shape vector must reconstruct the orthonormal `b`, and the laminate fractions must be finite, lie in `[0,1]`, and sum to one.

## 10. The root tree has one stable reading order

The result hierarchy is always `Root -> family -> correspondence-variant pair -> physical twin branch -> A/M habit branch`. Representative calculations appear first; symmetry-equivalent repetitions and numerical detail may be progressively disclosed, but no result changes meaning or location after calculation without a new explicit Calculate action.

## 11. Literature is validation, not production logic

Published NiTi twin/habit, CuAlNi orthorhombic twin/habit, and higher-order weak cases are stored only under `data/benchmarks/` and are consumed only by tests. Production Python files must not contain their answer constants.
