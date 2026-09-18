# Core source set used to design the code

The code is written around equations and distinctions from these primary/theory sources. Page/equation numbers should be checked again before publication-grade reporting.

1. C. Cayron, **Groupoid of orientational variants**, Acta Crystallographica A 62 (2006) 21–40.
   - left cosets = variants;
   - double cosets = operators;
   - groupoid composition / crystallographic signature.

2. C. Cayron, **The transformation matrices (distortion, orientation, correspondence), their continuous forms, and their variants** (2018/2019 preprint/publication).
   - F/T/C distinction;
   - different variant types and intersection groups;
   - warning that stretch and correspondence variants are not generally identical.

3. C. Cayron, **Correspondence Theory**, Crystals 12 (2022) 130.
   - correspondence variants/operators;
   - Type-I / Type-II / weak-junction construction;
   - EBSD disorientation connection.

4. C. Cayron, Acta Materialia 316 (2026) 122399.
   - metric-centered CT;
   - CMC, SMC;
   - A/M + M/M + shear/shear supercompatibility;
   - link to conventional stretch/cofactor theory.

5. C. Cayron, **The crystallographic quaternions and their product law** (2026).
   - cross tensor and direct use of non-Cartesian crystallographic bases.

6. C. Cayron, **The crossmetric tensor and the geometrical meaning of the imaginary numbers** (2026).
   - crossmetric extension and groupoid interpretation of quaternion composition.

7. R. D. James and K. F. Hane, **Martensitic transformations and shape-memory materials**, Acta Materialia 48 (2000) 197–222.
   - Cu-based 18R/M18R/6M issue;
   - 6M correspondence importance;
   - cube-edge monoclinic stretch variants;
   - Cu–Al–Ni lattice example and compatibility discussion.

8. J. M. Ball and R. D. James, **Fine phase mixtures as minimizers of energy**, Arch. Rational Mech. Anal. 100 (1987) 13–52.
   - nonlinear elasticity and rank-one compatibility.

9. X. Chen, V. Srivastava, V. Dabade, R. D. James, **Study of the cofactor conditions: Conditions of supercompatibility between phases**, J. Mech. Phys. Solids 61 (2013) 2566–2587.
   - CC1/CC2/CC3 and arbitrary twin-volume-fraction compatibility.

10. M. Bevis and A. G. Crocker, **Twinning modes in lattices**, Proc. Roy. Soc. A (1969).
    - K1, K2, eta1, eta2 and metric-dependent twinning elements.

11. K. Otsuka and X. Ren, **Physical metallurgy of Ti–Ni-based shape memory alloys**, Prog. Mater. Sci. 50 (2005) 511–678.
    - broader martensite/twinning/phase context, including comparisons to Cu–Al–Ni.

12. Otsuka/Wayman/Saburi Cu-based martensite literature.
    - Cu–Al–Ni 18R/2H transformations and self-accommodation.

## Primary-source task still required

Before a Cu–Al–Ni DO3→6M correspondence is hard-coded, re-derive it from the primary Otsuka–Ohba–Tokonami–Wayman 1993 cell description and/or Hane's original 6M treatment, record the exact basis convention, and verify that `U` reconstructed from `C, M_A, M_M` reproduces the published cube-edge stretch family.
