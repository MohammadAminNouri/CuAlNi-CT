# Verification matrix

| Claim / module | Independent check |
|---|---|
| 48-element cubic full point group | exact closure, inverses, determinants, 24 proper rotations |
| Correspondence subgroup | exact SymPy intersection after basis conjugation |
| Variant count | exact left-coset partition + Lagrange theorem |
| Operator count | explicit double-coset partition + independent Burnside count |
| 6M correspondence | reproduces independent James–Hane Eq. (10) stretch family |
| 2H correspondence | reproduces cubic→orthorhombic stretch family + literature orientation anchors |
| CMC/Ball–James bridge | numerical identity \(D=U^2-I\) for cubic parent |
| CT twin formulas | rank-one/twin results can be compared against Ball–James/Mallard law |
| Cofactor CC1–CC3 | regression tests against theorem equations; all-f verification helper |
| EBSD orientation conventions | explicit crystal→sample internal convention; round-trip tests to be expanded per vendor |

A passing unit test proves only the stated mathematical identity, not physical validity of an input dataset.
