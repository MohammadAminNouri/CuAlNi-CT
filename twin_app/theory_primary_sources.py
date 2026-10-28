from __future__ import annotations

"""Primary, bibliographically verified foundations for the current theory panel.

This is scientific source provenance, not a new theoretical method, solver,
optimizer, UI input, preference, or measurement. Citation records are static.
All equations follow the stated base/configuration and refer to the particular
kind of interface tested, not a generalized universal phase prediction.
"""


REFERENCES = (
    {
        "key": "Cayron2026",
        "theory": "Cayron CT, CMC, SMC and A/M/M",
        "citation": (
            "C. Cayron, Compatibilities and supercompatibility conditions in "
            "shape memory alloys determined from correspondence, metrics and symmetries, "
            "Acta Materialia 316 (2026) 122399."
        ),
        "doi": "10.1016/j.actamat.2026.122399",
        "used_for": (
            "M/M twin operators (Eqs. 19–25), CMC (Eq. 37), "
            "SMC and shear/shear condition (Eq. 43), "
            "one-way implication to cofactor criteria (§§2.3, 6)."
        ),
        "boundary": (
            "Source gives a NiTi B2→B19′ worked example, not an "
            "experimentally calibrated Cu–Al–Ni predictor; the printed "
            "§5.3 O2 Type-I parameter family is internally disputed."
        ),
    },
    {
        "key": "Cayron2022",
        "theory": "Correspondence variants, twins and ORs",
        "citation": (
            "C. Cayron, The Correspondence Theory and Its Application to NiTi "
            "Shape Memory Alloys, Crystals 12 (2022) 130."
        ),
        "doi": "10.3390/cryst12020130",
        "used_for": (
            "Symmetry/correspondence operators, Type-I/II twin construction, "
            "closing-gap OR and weak-plane distinction."
        ),
        "boundary": (
            "Natural-OR selection is an additional assumption; weak planes "
            "are not automatically exact rank-one twins, and a candidate OR "
            "is not an observed variant."
        ),
    },
    {
        "key": "Cayron2019",
        "theory": "Correct crystallographic representations",
        "citation": (
            "C. Cayron, The transformation matrices (distortion, orientation, "
            "correspondence), their continuous forms and their variants, "
            "Acta Crystallographica A 75 (2019) 411–437."
        ),
        "doi": "10.1107/S205327331900038X",
        "used_for": (
            "Distinguishing distortion, orientation and correspondence "
            "matrices, crystallographic versus Cartesian/reciprocal frames "
            "and coset variant counting."
        ),
        "boundary": (
            "Read with Cayron's 2019 corrigendum. Numbers of orientation, "
            "stretch and correspondence variants need not agree."
        ),
    },
    {
        "key": "Cayron2019Correction",
        "theory": "Actual point groups versus lattice metric automorphisms",
        "citation": (
            "C. Cayron, The transformation matrices (distortion, orientation, "
            "correspondence), their continuous forms and their variants. "
            "Corrigenda, Acta Crystallographica A 75 (2019) 777."
        ),
        "doi": "10.1107/S2053273319009276",
        "used_for": (
            "Corrects the false equivalence between g^T M g = M "
            "and membership in the full crystal point group."
        ),
        "boundary": (
            "Metric preservation is necessary, NOT sufficient for a genuine "
            "point-group operation. Supplied symmetry must be independently "
            "grounded in the actual structure and settings."
        ),
    },
    {
        "key": "BallJames1987",
        "theory": "Ball–James nonlinear compatibility",
        "citation": (
            "J. M. Ball and R. D. James, Fine phase mixtures as minimizers "
            "of energy, Archive for Rational Mechanics and Analysis 100 "
            "(1987) 13–52."
        ),
        "doi": "10.1007/BF00281246",
        "used_for": (
            "Nonlinear energy wells SO(3)U_i, rank-one compatibility "
            "and the middle-stretch condition for A/M interfaces."
        ),
        "boundary": (
            "Kinematic admissibility or an energy-minimizing sequence "
            "does not establish kinetics, actual variant populations "
            "or specimen-specific twin selection."
        ),
    },
    {
        "key": "JamesHane2000",
        "theory": "Ball–James applications to SMA variants",
        "citation": (
            "R. D. James and K. F. Hane, Martensitic transformations "
            "and shape-memory materials, Acta Materialia 48 (2000) 197–222."
        ),
        "doi": "10.1016/S1359-6454(99)00295-5",
        "used_for": (
            "Finite-stretch compatibility, twin modes and rank-one "
            "A/M and M/M interface constructions used in the BJ adapter."
        ),
        "boundary": (
            "All rotations/vectors must be expressed in the same orthonormal "
            "frame before using RU_j−U_i=a⊗n. Twin branches are not "
            "experimental variant counts."
        ),
    },
    {
        "key": "HaneShield1998",
        "theory": "Symmetry, twinning-mode enumeration",
        "citation": (
            "K. F. Hane and T. W. Shield, Symmetry and microstructure in "
            "martensites, Philosophical Magazine A 78 (1998) 1215–1252."
        ),
        "doi": "10.1080/01418619808239984",
        "used_for": (
            "Enumeration of symmetry-equivalent twinning systems and "
            "nonlinear compatibility microstructures."
        ),
        "boundary": (
            "Symmetry-equivalent and sign-reversed descriptions must not "
            "be counted as experimentally distinct interfaces."
        ),
    },
    {
        "key": "WLR1953",
        "theory": "Classical PTMC (WLR)",
        "citation": (
            "M. S. Wechsler, D. S. Lieberman and T. A. Read, "
            "On the theory of the formation of martensite, Transactions "
            "of the AIME 197 (1953) 1503–1515."
        ),
        "doi": "",
        "used_for": (
            "Habit-plane and shape deformation predictions from "
            "lattice deformation with lattice-invariant deformation."
        ),
        "boundary": (
            "The chosen lattice-invariant shear/twin mechanism "
            "is part of PTMC's physical hypothesis."
        ),
    },
    {
        "key": "BowlesMackenzie1954",
        "theory": "Classical PTMC (BM)",
        "citation": (
            "J. S. Bowles and J. K. Mackenzie, The crystallography "
            "of martensite transformations I, Acta Metallurgica 2 "
            "(1954) 129–137."
        ),
        "doi": "10.1016/0001-6160(54)90102-9",
        "used_for": (
            "Macroscopic invariant-plane shape and accommodating "
            "lattice-invariant deformation; foundational PTMC formulation."
        ),
        "boundary": (
            "A twinned laminate's A/M habit plane is not identical "
            "to the martensite-to-martensite twinning plane."
        ),
    },
    {
        "key": "ChenEtAl2013",
        "theory": "Cofactor conditions",
        "citation": (
            "X. Chen, V. Srivastava, V. Dabade and R. D. James, "
            "Study of the cofactor conditions: Conditions of "
            "supercompatibility between phases, Journal of the "
            "Mechanics and Physics of Solids 61 (2013) 2566–2587."
        ),
        "doi": "10.1016/j.jmps.2013.08.004",
        "used_for": (
            "Theorem 2: CC1 λ2(U)=1; "
            "CC2 a·U cof(U²−I)n=0; "
            "CC3 tr(U²)−det(U²)−|a|²|n|²/4−2≥0. "
            "For a valid symmetry-related M/M twin, "
            "these conditions are equivalent to A/M habit compatibility "
            "for every fraction f∈[0,1]."
        ),
        "boundary": (
            "The published ABSTRACT has a contradictory PLUS det(U²) in CC3; "
            "Theorem 2 AND ITS PROOF use MINUS det(U²). "
            "The theorem assumes an admissible nonzero rank-one twin "
            "with symmetry-related positive-definite stretches."
        ),
    },
    {
        "key": "TahseenDabade2026",
        "theory": "Limit of classical cofactor criteria for compound-domain microstructures",
        "citation": (
            "M. Tahseen and V. Dabade, Revisiting the cofactor conditions: "
            "Elimination of transition layers in compound domains, Journal "
            "of the Mechanics and Physics of Solids 206 (2026) 106409."
        ),
        "doi": "10.1016/j.jmps.2025.106409",
        "used_for": (
            "Clarifies that conventional cofactor satisfaction and "
            "elimination of transition layers for compound laminates "
            "are not synonymous; develops additional extreme "
            "compatibility conditions."
        ),
        "boundary": (
            "This app does NOT implement the new extreme-compatibility theory. "
            "Do not claim a compound cofactor pass proves a stress-free "
            "triple cluster or all possible compound laminate interfaces."
        ),
    },
    {
        "key": "Bhattacharya2003",
        "theory": "Interpretation and microstructure foundations",
        "citation": (
            "K. Bhattacharya, Microstructure of Martensite: Why It Forms "
            "and How It Gives Rise to the Shape-Memory Effect, "
            "Oxford University Press (2003)."
        ),
        "doi": "10.1093/oso/9780198509349.001.0001",
        "used_for": (
            "Nonlinear elasticity, laminates, interfaces, "
            "micromechanics and the difference between candidate "
            "kinematics and realized microstructure."
        ),
        "boundary": (
            "A crystallographically admissible interface need not be "
            "selected in a constrained single crystal or polycrystal."
        ),
    },
    {
        "key": "CuAlNi2006",
        "theory": "Phase-specific Cu–Al–Ni crystallography",
        "citation": (
            "U. Sari and I. Aksoy, Electron microscopy study of 2H and 18R martensites in "
            "Cu–11.92 wt% Al–3.78 wt% Ni shape memory alloy, "
            "Journal of Alloys and Compounds 417 (2006) 138–142."
        ),
        "doi": "10.1016/j.jallcom.2005.09.049",
        "used_for": (
            "Experimental phase distinctions: orthorhombic 2H and "
            "modified monoclinic 18R (M18R), with composition/heat-treatment "
            "dependent occurrence."
        ),
        "boundary": (
            "Phase-specific lattice setting, supercell/reduced cell "
            "and parent→martensite correspondence must be established "
            "for the actual specimen. NiTi analytical C1–C3 formulas "
            "cannot be transferred by relabeling."
        ),
    },
)


def citation_rows():
    return [
        {
            "Theory/role": r["theory"],
            "Primary reference": r["citation"],
            "DOI": r["doi"] if r["doi"] else "DOI not supplied (1953 primary)",
            "What it supports": r["used_for"],
            "Assumptions and caveats": r["boundary"],
        }
        for r in REFERENCES
    ]


def thesis_reference_notes():
    lines = [
        "## Primary academic literature for the compatibility calculations",
        "",
        "These references support mathematical methods, not experimental "
        "validation of a specific Cu–Al–Ni specimen. Preserve each result's "
        "basis, lattice setting, correspondence, tolerance and branch identity.",
    ]
    for r in REFERENCES:
        lines.extend((
            "", f"**{r['key']} — {r['theory']}**",
            r["citation"] + (f" DOI: {r['doi']}" if r["doi"] else ""),
            f"Use: {r['used_for']}",
            f"Conditions / limitations: {r['boundary']}",
        ))
    return "\n".join(lines) + "\n"


def render_academic_sources():
    import pandas as pd
    import streamlit as st

    with st.expander("Primary papers behind each equation and necessary assumptions", expanded=False):
        st.caption(
            "Source-to-equation audit for the EXISTING CT, Ball–James, PTMC "
            "and cofactor calculations. This adds no new criterion, physical "
            "model, correspondence or numerical parameters."
        )
        st.dataframe(pd.DataFrame(citation_rows()),
                     hide_index=True, width="stretch")
        st.warning(
            "Two source-critical guards: (i) Cayron (2019) corrigendum: "
            "metric-preserving lattice operations are not automatically "
            "crystal point-group symmetries; (ii) Chen et al. (2013): "
            "CC3 uses MINUS det(U²) in Theorem 2, despite the "
            "contradictory abstract. The backend already follows Theorem 2."
        )
        st.caption(
            "For Cu–Al–Ni, confirm the parent symmetry, 2H/modified-18R "
            "crystal setting, direct/reciprocal basis conversions and "
            "the parent→product correspondence. A green CI alone is "
            "not independent validation of experimentally selected twins."
        )
