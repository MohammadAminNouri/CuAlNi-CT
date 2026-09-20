from __future__ import annotations

from .provenance import SourceRef

SOURCES: dict[str, SourceRef] = {
    "cayron2006_groupoid": SourceRef(
        key="cayron2006_groupoid",
        citation="C. Cayron, Groupoid of orientational variants, Acta Crystallographica A62 (2006) 21–40.",
        doi="10.1107/S010876730503686X",
        notes="Cosets, double cosets, operators, groupoid composition and Burnside counting.",
    ),
    "cayron2019_ftc": SourceRef(
        key="cayron2019_ftc",
        citation="C. Cayron, The transformation matrices (distortion, orientation, correspondence), their continuous forms, and their variants, Acta Crystallographica A75 (2019) 411–437.",
        doi="10.1107/S2053273319003472",
        notes="Distinguishes F, T, C and their variant sets; warns against conflating stretch and correspondence variants.",
    ),
    "cayron2022_ct": SourceRef(
        key="cayron2022_ct",
        citation="C. Cayron, The Correspondence Theory and Its Application to NiTi Shape Memory Alloys, Crystals 12 (2022) 130.",
        doi="10.3390/cryst12020130",
        notes="Correspondence variants/operators, Type-I/II transformation twins, closing-gap ORs and weak-plane discussion.",
    ),
    "cayron2022_weak_twins": SourceRef(
        key="cayron2022_weak_twins",
        citation="C. Cayron, The concept of axial weak twins, Acta Materialia 236 (2022) 118128.",
        notes="Axial weak planes/twins and explicit length/angular misfit search for non-classical interfaces.",
    ),
    "cayron2026_supercompat": SourceRef(
        key="cayron2026_supercompat",
        citation="C. Cayron, Compatibilities and supercompatibility conditions in shape memory alloys determined from correspondence, metrics and symmetries, Acta Materialia 316 (2026) 122399.",
        doi="10.1016/j.actamat.2026.122399",
        equations=("18-25", "26-30", "31-42", "50"),
        notes="CMC, SMC, CT Type-I/II twins and shear/shear supercompatibility.",
    ),
    "cayron2024_hardsphere": SourceRef(
        key="cayron2024_hardsphere",
        citation="C. Cayron, Hard-sphere model of the B2 to B19' phase transformation, Acta Materialia 270 (2024) 119870.",
        notes="Separate physical/atomistic model for a natural B2->B19' distortion path; not a generic Cu-Al-Ni law.",
    ),
    "cayron2026_quaternion": SourceRef(
        key="cayron2026_quaternion",
        citation="C. Cayron, The crystallographic quaternions and their product law (2026 preprint).",
        notes="Cross tensor and quaternion product directly in non-Cartesian crystallographic bases.",
    ),
    "cayron2026_crossmetric": SourceRef(
        key="cayron2026_crossmetric",
        citation="C. Cayron, The crossmetric tensor and the geometrical meaning of the imaginary numbers (2026 preprint).",
        notes="Crossmetric tensor and source-target quaternion interpretation.",
    ),
    "james_hane2000": SourceRef(
        key="james_hane2000",
        citation="R.D. James, K.F. Hane, Martensitic transformations and shape-memory materials, Acta Materialia 48 (2000) 197–222.",
        doi="10.1016/S1359-6454(99)00295-5",
        equations=("9", "10", "12-17", "24", "25"),
        notes="Important Cu-based 6M benchmark and nonlinear-elasticity review; not a universal Cu-Al-Ni parameter set.",
    ),
    "otsuka_ohba1993": SourceRef(
        key="otsuka_ohba1993",
        citation="K. Otsuka, T. Ohba, M. Tokonami, C.M. Wayman, New description of long period stacking order structures of martensites in β-phase alloys, Scripta Metallurgica et Materialia 29 (1993) 1359–1364.",
        doi="10.1016/0956-716X(93)90139-J",
        notes="Primary source for the revised reduced/6M description of long-period martensite.",
    ),
    "otsuka_shimizu1974": SourceRef(
        key="otsuka_shimizu1974",
        citation="K. Otsuka, K. Shimizu, Morphology and Crystallography of Thermoelastic Cu–Al–Ni Martensite Analyzed by the Phenomenological Theory, Transactions of the Japan Institute of Metals 15 (1974) 103–108.",
        doi="10.2320/matertrans1960.15.103",
        notes="Classical Cu-Al-Ni PTMC/morphology source.",
    ),
    "otsuka_nakamura_shimizu1974": SourceRef(
        key="otsuka_nakamura_shimizu1974",
        citation="K. Otsuka, T. Nakamura, K. Shimizu, Electron Microscopy Study of Stress-induced Acicular β1' Martensite in Cu–Al–Ni Alloy, Transactions of the Japan Institute of Metals 15 (1974) 200–206.",
        doi="10.2320/matertrans1960.15.200",
        notes="Cu-14.2Al-4.3Ni wt% stress-induced 18R β1' crystallography, habit and OR observations.",
    ),
    "bevis_crocker1969": SourceRef(
        key="bevis_crocker1969",
        citation="M. Bevis, A.G. Crocker, Twinning modes in lattices, Proceedings of the Royal Society A (1969).",
        notes="General lattice twinning elements K1, K2, eta1, eta2 and metric dependence.",
    ),
    "ball_james1987": SourceRef(
        key="ball_james1987",
        citation="J.M. Ball, R.D. James, Fine phase mixtures as minimizers of energy, Archive for Rational Mechanics and Analysis 100 (1987) 13–52.",
        notes="Nonlinear elasticity, rank-one compatibility, transformation stretch wells.",
    ),
    "chen2013_cofactor": SourceRef(
        key="chen2013_cofactor",
        citation="X. Chen, V. Srivastava, V. Dabade, R.D. James, Study of the cofactor conditions: Conditions of supercompatibility between phases, Journal of the Mechanics and Physics of Solids 61 (2013) 2566–2587.",
        doi="10.1016/j.jmps.2013.07.004",
        equations=("CC1", "CC2", "CC3"),
        notes="Necessary and sufficient cofactor conditions for all twin fractions under the theorem hypotheses.",
    ),
    "wlr1953": SourceRef(
        key="wlr1953",
        citation="M.S. Wechsler, D.S. Lieberman, T.A. Read, On the theory of the formation of martensite, Transactions AIME 197 (1953) 1503–1515.",
    ),
    "bowles_mackenzie1954": SourceRef(
        key="bowles_mackenzie1954",
        citation="J.S. Bowles, J.K. Mackenzie, The crystallography of martensitic transformations I–II, Acta Metallurgica 2 (1954) 129–147.",
    ),
    "chen2000_ebsd_2h": SourceRef(
        key="chen2000_ebsd_2h",
        citation="X. Chen et al., Orientation relationships of martensite variants determined by electron backscatter diffraction, Micron 31 (2000) 17–25.",
        doi="10.1016/S0968-4328(99)00060-8",
        notes="Cu-12.55Al-4.84Ni wt% 2H EBSD: basal planes inherit parent {110}; [010]2H inherits parent <001>; observed {121}2H and {101}2H twin mirrors.",
    ),
    "landa2007_2h": SourceRef(
        key="landa2007_2h",
        citation="M. Landa et al., Temperature dependence of elastic properties of cubic and orthorhombic phases in Cu–Al–Ni shape memory alloy near their stability limits, Materials Science and Engineering A 462 (2007) 320–324.",
        doi="10.1016/j.msea.2006.02.472",
        notes="Cu-13.8Al-4.1Ni wt% benchmark; parent a0=0.5835 nm and 2H a=0.4389, b=0.5342, c=0.4224 nm used with an explicit lattice-correspondence figure.",
    ),
    "sari_aksoy2006": SourceRef(
        key="sari_aksoy2006",
        citation="U. Sari, I. Aksoy, Electron microscopy study of 2H and 18R martensites in Cu–11.92 wt% Al–3.78 wt% Ni shape memory alloy, Journal of Alloys and Compounds 417 (2006) 138–142.",
        doi="10.1016/j.jallcom.2005.09.049",
        notes="Shows heat-treatment-dependent coexistence/selection of 2H and modified-18R martensites; useful guard against treating Cu-Al-Ni as one fixed martensite.",
    ),
    "ibarra2006_tem": SourceRef(
        key="ibarra2006_tem",
        citation="A. Ibarra et al., 'In situ' and 'Post-mortem' TEM study of the super-elastic effect in Cu–Al–Ni shape memory alloys, Materials Science and Engineering A 438–440 (2006) 787–790.",
        doi="10.1016/j.msea.2006.01.105",
        notes="Reports L21 parent and beta3'/gamma3' martensite structural parameters used for in-situ TEM indexing and OR work.",
    ),
}


def get_source(key: str) -> SourceRef:
    try:
        return SOURCES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown source key {key!r}") from exc
