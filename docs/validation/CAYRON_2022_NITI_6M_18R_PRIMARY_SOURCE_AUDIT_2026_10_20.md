# Source-grounded CT audit for supervisor — 20 October 2026

Status: MATHEMATICAL SOURCE ORACLE VERIFIED IN INDEPENDENT RECOMPUTATION.
New application-level regression tests are subject to the current GitHub CI run.
NOT a 2H/18R specimen-specific experimental validation.

## Cayron's NiTi higher-order weak twin (2022)

C. Cayron, *Crystals* **12** (2022) 130,
DOI: 10.3390/cryst12020130, Section 5, Table 3/5.

B19-prime monoclinic metric: a=2.889 Å, b=4.120 Å, c=4.622 Å,
alpha=gamma=90 degrees, beta=96.8 degrees.
Published product-to-product inter-correspondence (NOT the parent-to-product C):

    C_int = [[0, -1, 1], [1/2, 1/2, 1/2], [-1/2, 1/2, 1/2]]

C_int [0,1,1]^T = [0,1,1]^T.
C_int^(-T) [1,-3,3]^T = [3,1,-1]^T.
C_int [3,1,0]^T = [-1,2,-1]^T.
C_int [3,0,-1]^T = [-1,1,-2]^T.
The inter-correspondence has det=1 and common-sublattice index q_g=2.

Independently evaluated physical metric and Eq. (3) orientation sign
branches yield generalized strain 0.291253477712293 and minimum
generalized shear 0.291139820240297. Cayron reports respectively
approximately 0.2912 and 0.2911.

CAUTION: An accessible article text rendition writes [-1,-1,-2] for
the image of [3,0,-1] despite the displayed C_int giving [-1,1,-2].
The test uses the exact published matrix multiplication; the printed
vector discrepancy requires checking the publisher PDF/typesetting,
not changing C_int or forcing a matching answer.

2019 conventions: Cayron, *Acta Cryst. A* **75** (2019) 411–437,
DOI: 10.1107/S205327331900038X; C=Q P^(-1) for paired DIRECT
supercell columns and reciprocal plane mapping C^(-T).
Corrigendum DOI: 10.1107/S2053273319009276: preserving a metric
does not by itself make an operation a physical point-group element.
Use the independently specified crystallographic point group.

## NiTi historical habit reference-vector semantics

The immutable Bhattacharya-derived NiTi fixture has a Type-II
shear~0.2804 branch with b=[0.91387,-0.37102,-0.16490],
whose Euclidean norm is 1.0000031136, a direction-like number.
Other b vectors in the same fixture have much smaller physical
magnitudes. It is scientifically unsafe to treat this vector as
the full physical b in the rank-one shape matrix I + b ⊗ m without
reconstructing the amplitude from the actual source.

The separately locked Otsuka–Ren (2005)/Jin–Weng (2002) physical
Type-II habit oracle provides a UNIT strain direction and a SEPARATE
shape strain magnitude 0.13078. Use this explicit physical magnitude
for the supported NiTi habit check. The legacy numerical fixture
has not been silently edited and its source's original table has NOT
been conclusively checked for the suspected normalization ambiguity.

## Reduced gamma-angle 6M cell is not a measured conventional 18R cell

The *specific passive coordinate rebase* of the James–Hane Cu–Al–Ni
gamma-angle 6M metric uses unimodular P:

    P = [[1,0,0], [0,0,1], [0,-1,0]]
    G_unique-b = P^T G_gamma P
    C_(new M <- A) = P^(-1) C_(old M <- A)
    u_new = P^(-1) u_old
    h_new = P^T h_old

The unit-determinant basis transport preserves all physical lengths,
parent-pulled-back product metrics, determinant/volume, and reciprocal
plane/direct-vector incidence. The new helper rejects a missing,
singular or non-rational source C. The numerical test uses a declared
HYPOTHETICAL C, not an inferred Cu–Al–Ni correspondence.

A conventional 18R reported from a separate specimen/diffraction
experiment cannot be relabelled from this 6M representation by
matching cell dimensions or monoclinic angles. Additional experimental
stacking/structure evidence and an EXPLICIT, independently established
basis/supercell mapping are required. No such source-backed 6M-to-18R
cross-sample mapping has been asserted.

## Presentation rule

This audit reproduces mathematical correspondence, index and CT strain/
shear, and tests the exact basis-setting invariants. It does NOT imply
that numerical 2H and 18R interface predictions have been compared
with independent, specimen-specific indexed 3D EBSD/TEM normals.
