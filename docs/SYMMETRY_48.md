# Why the cubic parent has 48 symmetry operations

For cubic `m-3m` (`O_h`) the **full crystallographic point group has 48
operations**.  This is the number that must be used in Cayron-style
correspondence topology for the cubic Cu-Al-Ni parent.

Exact inventory in the conventional cubic basis:

| class | count |
|---|---:|
| identity | 1 |
| proper 180° rotations | 9 |
| proper 120°/240° rotations | 8 |
| proper ±90° rotations | 6 |
| inversion | 1 |
| mirror reflections | 9 |
| order-4 rotoinversions | 6 |
| order-6 rotoinversions | 8 |
| **total** | **48** |

Thus:

\[
|G_{m\bar3m}|=48,\qquad |G_{m\bar3m}^{+}|=24.
\]

## Why CT needs all 48

A CT Type-I transformation twin is generated from a **parent reflection**.
If the code prematurely restricts the parent symmetry to the 24 proper
rotations, all nine parent mirrors disappear and the Type-I/operator
classification is damaged.

The full group is therefore used for:

- correspondence intersection subgroup \(H_C\);
- left cosets / correspondence variants;
- double cosets / intercorrespondence operators;
- ambivalent versus polar operator classification;
- reflection-derived Type-I candidates;
- two-fold-rotation-derived Type-II candidates;
- groupoid composition.

## Why 24 is also important

A physical orientation matrix or EBSD disorientation is a proper rotation in
\(SO(3)\).  Therefore the 24-element proper cubic subgroup is used for
rotation-valued tasks such as:

- symmetry-reduced OR/disorientation comparison;
- proper orientation variant representatives;
- quaternion representations of physical rotations.

The program calculates full and proper topology separately.  It never treats
"48 versus 24" as a contradiction: they answer different crystallographic
questions.

## Required software assertions

The implementation must continuously prove:

```text
full cubic m-3m operations       = 48
proper rotations                 = 24
improper operations              = 24
parent mirrors                   = 9
proper parent two-fold rotations = 9
```

and all 48 matrices must preserve the cubic metric.

For a specific transformation, equality such as

\[
H_C=H_T
\]

or equality of correspondence/orientation operator counts is a computed
result, not a general law.
