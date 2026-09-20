# Core source set used to design and validate the code

The code is organized around primary/theory sources, but no single paper is
allowed to define all Cu-Al-Ni inputs.  Each literature parameter set remains
source-specific.

## Cayron / correspondence-theory backbone

1. C. Cayron, **Groupoid of orientational variants**, Acta Crystallographica A
   62 (2006) 21–40.
   - left cosets = variants;
   - double cosets = operators;
   - groupoid composition / crystallographic signature.

2. C. Cayron, **The transformation matrices (distortion, orientation,
   correspondence), their continuous forms, and their variants**, Acta
   Crystallographica A 75 (2019) 411–437.
   - distinction between F, T, C;
   - separate variant/intersection groups;
   - warning against conflating correspondence and stretch variants.

3. C. Cayron, **The Correspondence Theory and Its Application to NiTi Shape
   Memory Alloys**, Crystals 12 (2022) 130.
   - correspondence variants/operators;
   - natural versus closing-gap OR;
   - Type-I/Type-II construction;
   - weak/polar operators.

4. C. Cayron, **The concept of axial weak twins**, Acta Materialia 236 (2022)
   118128.
   - weak-plane and weak-twin search outside exact classical twin cases.

5. C. Cayron, **Compatibilities and supercompatibility conditions in shape
   memory alloys determined from correspondence, metrics and symmetries**,
   Acta Materialia 316 (2026) 122399.
   - CMC and its degeneracy orders;
   - SMC = shear by metric correspondence;
   - A/M + M/M + shear/shear supercompatibility.

6. C. Cayron, B2 -> B19' hard-sphere model, Acta Materialia 270 (2024)
   119870.
   - separate physical model for a natural atomistic/distortion path;
   - not assumed to be a universal Cu-Al-Ni model.

## Independent continuum / PTMC / compatibility references

7. J. M. Ball and R. D. James, **Fine phase mixtures as minimizers of energy**,
   Arch. Rational Mech. Anal. 100 (1987) 13–52.

8. R. D. James and K. F. Hane, **Martensitic transformations and shape-memory
   materials**, Acta Materialia 48 (2000) 197–222.
   - nonlinear-elasticity and compatibility review;
   - important Cu-based 6M benchmark;
   - not treated as the universal Cu-Al-Ni material state.

9. X. Chen, V. Srivastava, V. Dabade, R. D. James, **Study of the cofactor
   conditions**, J. Mech. Phys. Solids 61 (2013) 2566–2587.
   - CC1/CC2/CC3;
   - arbitrary twin-volume-fraction compatibility under theorem hypotheses.

10. M. Bevis and A. G. Crocker, **Twinning modes in lattices**, Proc. Roy.
    Soc. A (1969).
    - K1, K2, eta1, eta2 and metric-dependent twinning elements.

11. WLR (1953) and Bowles-Mackenzie (1954).
    - classical phenomenological theory.

## Cu-Al-Ni-specific crystallographic evidence

The program deliberately uses several independent Cu-Al-Ni sources because
phase identity and crystallographic parameters vary with composition, order,
thermal history and loading.

12. K. Otsuka, T. Ohba, M. Tokonami, C. M. Wayman (1993), revised long-period
    martensite cell description.
    - basis/cell interpretation behind the reduced 6M treatment.

13. K. Otsuka, K. Shimizu (1974), classical Cu-Al-Ni morphology/PTMC.

14. K. Otsuka, T. Nakamura, K. Shimizu (1974), stress-induced beta1' 18R
    martensite.
    - Cu-14.2Al-4.3Ni wt%;
    - long-period lattice and matrix/martensite OR observations.

15. X. Chen et al. (2000), EBSD of 2H martensite.
    - Cu-12.55Al-4.84Ni wt%;
    - 2H basal plane from parent {110};
    - [010]2H from parent <001>;
    - observed {121}2H and {101}2H mirror-related variant pairs.

16. M. Landa et al. (2007), cubic and 2H single-crystal benchmark.
    - Cu-13.8Al-4.1Ni wt%;
    - explicit parent and 2H lattice parameters and lattice correspondence.

17. U. Sari, I. Aksoy (2006), coexistence/selection of 2H and M18R in
    Cu-11.92Al-3.78Ni wt% depending on heat treatment.

18. A. Ibarra et al. (2006), in-situ TEM of superelastic Cu-Al-Ni.
    - L21 beta3 parent;
    - conventional C2/m beta3' and Pmmn gamma3' cells;
    - martensite variant and OR observations.

## Status of the DO3 -> 6M reference correspondence

The repository no longer treats the old "derive this later" note as an open
task.

The selected reference 6M basis is explicitly recorded in
`reference_6m.py`, the package correspondence convention is explicit, and the
derived stretch family is cross-locked against the independently published
James-Hane cube-edge family.

That verification establishes the **internal reference branch**.  It does not
mean that the same correspondence or the same lattice parameters are valid
for every Cu-Al-Ni specimen or every long-period cell setting.

## Publication-grade rule

Before a number is presented as an experimental/material conclusion rather
than a software benchmark, its exact primary source, specimen state, cell
setting, uncertainty and any basis conversion must be recoverable from
provenance.
