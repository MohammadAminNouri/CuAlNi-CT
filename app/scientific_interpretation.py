from __future__ import annotations

"""Pure interpretation of already-computed scientific results.

No solver is called here. The functions translate backend outputs into explicit
claims with the exact criterion and numerical evidence that justify each claim.
This prevents presentation code from inventing or silently strengthening a
scientific conclusion.
"""

from typing import Any, Mapping, Sequence
import math

from app.scientific_types import Evidence, Finding, sci


def _value(obj: Any) -> str:
    value = getattr(obj, "value", obj)
    return str(value)


def _abs(value: Any) -> float | None:
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rows(unified: Any, *, theory: str | None = None, kind: str | None = None) -> list[Any]:
    output = []
    for row in getattr(unified, "rows", ()):
        if theory is not None and _value(getattr(row, "theory", "")) != theory:
            continue
        if kind is not None and _value(getattr(row, "prediction_kind", "")) != kind:
            continue
        output.append(row)
    return output


def transformation_findings(result: Mapping[str, Any]) -> tuple[Finding, ...]:
    summary = result["summary"]
    ct = result["ct_detail"]
    bj = result["ball_james_detail"]

    ct_exact = bool(summary["ct_exact_compatible"])
    bj_exact = bool(bj["lambda2_exact"])
    existence_agree = ct_exact == bj_exact
    nearest = float(ct["nearest_zero_residual"])
    lambda2 = float(summary["lambda2"])
    lambda2_residual = abs(lambda2 - 1.0)

    ct_finding = Finding(
        title="Correspondence Theory — single-variant A/M compatibility",
        conclusion=(
            "Exact CT A/M compatibility is satisfied."
            if ct_exact
            else "Exact CT A/M compatibility is not satisfied."
        ),
        rationale=(
            "CT requires the normalized correspondence-metric condition to have the required exact degeneracy. "
            + (
                f"That condition is satisfied here (degeneracy order {ct['degeneracy_order']})."
                if ct_exact
                else f"The nearest CMC eigenvalue is {sci(nearest, 6)} from zero, so no exact degeneracy exists."
            )
        ),
        how=(
            "The parent and product metric tensors are combined with the lattice correspondence C to form the CMC. "
            "The generalized metric spectrum is then classified by its zero eigenvalues and inertia."
        ),
        physical_meaning=(
            "A single martensite variant has an exact CT-compatible austenite/martensite interface for this lattice state."
            if ct_exact
            else "A homogeneous single martensite variant does not satisfy the exact CT interface condition for this lattice state."
        ),
        limitation=(
            "This does not decide whether a twinned or otherwise lattice-invariant martensitic mixture can form a compatible interface."
        ),
        evidence=(
            Evidence(
                "Nearest CMC zero residual",
                "0 within the project's exact-eigenvalue tolerance",
                sci(nearest, 6),
                "exact degeneracy" if ct_exact else "not exact",
            ),
            Evidence(
                "CMC inertia (−,0,+)",
                "must contain the required zero eigenvalue structure",
                str(ct["inertia"]),
                str(ct["reason"]),
            ),
        ),
        tone="good" if ct_exact else "warn",
        status="Exact theory result; approximate nearest-degeneracy planes are kept separate.",
    )

    bj_finding = Finding(
        title="Ball–James — single-variant A/M compatibility",
        conclusion=(
            "The Ball–James single-variant rank-one criterion is satisfied."
            if bj_exact
            else "The Ball–James single-variant rank-one criterion is not satisfied."
        ),
        rationale=(
            "For the single-variant compatibility test, the middle principal stretch must satisfy λ₂ = 1. "
            f"Here λ₂ = {lambda2:.9f}, so |λ₂−1| = {sci(lambda2_residual, 6)}."
        ),
        how=(
            "The right stretch U is diagonalized independently of the CT CMC classification. "
            "If λ₂ = 1, the two Ball–James rank-one branches are constructed and audited."
        ),
        physical_meaning=(
            "The single martensite variant admits a classical rank-one austenite/martensite connection."
            if bj_exact
            else "No exact classical single-variant rank-one A/M connection is obtained for this state."
        ),
        limitation="This is a single-variant result; PTMC may restore macroscopic compatibility by introducing lattice-invariant shear.",
        evidence=(
            Evidence("Middle principal stretch λ₂", "λ₂ = 1", f"{lambda2:.9f}", "satisfied" if bj_exact else "not satisfied"),
            Evidence("|λ₂−1|", "0 within exact tolerance", sci(lambda2_residual, 6), "exact" if bj_exact else "finite mismatch"),
            Evidence("Ball–James A/M branches", "2 branches when the criterion is satisfied", len(bj.get("solutions", [])), "constructed" if bj.get("solutions") else "none"),
        ),
        tone="good" if bj_exact else "warn",
    )

    agreement = Finding(
        title="Independent CT ↔ Ball–James existence check",
        conclusion=(
            "CT and Ball–James independently give the same A/M existence classification."
            if existence_agree
            else "CT and Ball–James give different A/M existence classifications; this requires investigation."
        ),
        rationale=(
            f"CT exact-compatible = {ct_exact}; Ball–James λ₂ criterion = {bj_exact}. "
            "The two criteria are evaluated through different native formulations before this comparison."
        ),
        how="The UI compares only the final existence classifications; it does not use one theory to generate the other.",
        physical_meaning=(
            "The two independent descriptions agree on whether a homogeneous single martensite variant can satisfy the exact A/M interface condition."
            if existence_agree
            else "The two independent descriptions are not presently giving the same physical classification."
        ),
        limitation="Agreement of existence does not by itself prove equality of every habit-plane or shape-strain branch; exact branch equivalence is tested separately.",
        evidence=(
            Evidence("CT exact A/M", "independent CT criterion", "yes" if ct_exact else "no", ""),
            Evidence("Ball–James exact A/M", "independent λ₂/rank-one criterion", "yes" if bj_exact else "no", ""),
        ),
        tone="good" if existence_agree else "bad",
    )

    return ct_finding, bj_finding, agreement


def topology_finding(result: Mapping[str, Any]) -> Finding:
    topology = result["topology"]
    parent_order = int(topology["parent_group_order"])
    product_order = int(topology["product_group_order"])
    variants = int(topology["n_variants"])
    operators = int(topology["n_operators"])
    return Finding(
        title="Variant and operator topology",
        conclusion=f"The transformation generates {variants} stretch variants and {operators} operator classes.",
        rationale=(
            "Variants and operators answer different group-theoretic questions, so their counts are not expected to be equal."
        ),
        how=(
            "The registered parent/product symmetry groups and the correspondence are used to construct the transformation groupoid; variants arise from cosets and operators from the corresponding double-coset structure."
        ),
        physical_meaning="The counts describe symmetry-distinct transformation states and symmetry-distinct relations between them.",
        limitation="The counts are topological classifications; they do not rank which variant will be observed experimentally.",
        evidence=(
            Evidence("Parent group order", "registered crystallographic symmetry", parent_order, ""),
            Evidence("Product group order", "registered crystallographic symmetry", product_order, ""),
            Evidence("Stretch variants", "coset count", variants, ""),
            Evidence("Operator classes", "double-coset count", operators, ""),
        ),
        tone="neutral",
    )


def orientation_finding(analysis: Mapping[str, Any]) -> Finding:
    report = analysis["report"]
    audit = report["audit"]
    parity = report["parity"]
    n_variants = int(report["orientation_variant_count"])
    n_operators = int(report["orientation_operator_count"])
    residual = float(audit["maximum_residual"])
    parity_residual = float(parity["maximum_residual"])
    return Finding(
        title="Physical orientation relationship",
        conclusion=f"The supplied OR is a proper rotation and generates {n_variants} symmetry-distinct orientation variants.",
        rationale=(
            f"The SO(3) audit residual is {sci(residual, 3)} and representation-parity residual is {sci(parity_residual, 3)}."
        ),
        how=(
            "R is interpreted only as the Cartesian mapping x_parent = R(parent←product) x_product. "
            "The inverse mapping is R(product←parent) = Rᵀ; the lattice correspondence C is never substituted for R."
        ),
        physical_meaning="The OR specifies how physical Cartesian directions of the two lattices are oriented relative to one another, including all symmetry-equivalent variants.",
        limitation="An entered or polar-derived OR is not automatically an experimentally preferred OR; provenance determines what can be claimed.",
        evidence=(
            Evidence("OR variants", "proper-symmetry cosets", n_variants, ""),
            Evidence("OR operators", "proper-symmetry double cosets", n_operators, ""),
            Evidence("SO(3) residual", "≈ 0", sci(residual, 3), "proper rotation"),
            Evidence("Representation parity", "≈ 0", sci(parity_residual, 3), "frame-conversion consistency"),
        ),
        tone="good",
    )


def martensite_pair_finding(pair: Mapping[str, Any], *, unique_systems: int) -> Finding:
    relations = list(pair.get("relations", []))
    ptmc_count = sum(len(item.get("ptmc", [])) for item in relations if isinstance(item, Mapping))
    cofactor_all = sum(bool(item.get("cofactor", {}).get("satisfied")) for item in relations if isinstance(item, Mapping))
    max_residual = max((float(item.get("mallard_residual", 0.0)) for item in relations), default=0.0)
    return Finding(
        title="Selected martensite-variant pair",
        conclusion=(
            f"{len(relations)} Mallard generator/branch relations reduce to {unique_systems} unique physical twin system(s)."
            if relations
            else "No registered order-two parent symmetry produced a Mallard twin for this variant pair."
        ),
        rationale=(
            "Generator branches that reconstruct the same physical rank-one dyad are grouped before reporting, so duplicate symmetry constructions are not presented as different twins."
        ),
        how=(
            "The chosen stretch variants are tested through the registered parent twofold symmetries, Mallard's construction, the cofactor conditions, and classical single-shear PTMC."
        ),
        physical_meaning=(
            "The reported systems are physically distinct compatible relations between the two selected martensite variants."
            if relations
            else "This particular pair has no compatible Mallard twin within the registered symmetry construction."
        ),
        limitation="PTMC and cofactor conclusions depend on the selected twin system; they are not automatically statements about every variant pair.",
        evidence=(
            Evidence("Raw generator/branch relations", "before physical deduplication", len(relations), ""),
            Evidence("Unique physical twin systems", "rank-one dyad equivalence", unique_systems, ""),
            Evidence("Largest Mallard residual", "≈ 0", sci(max_residual, 3), "numerical construction audit"),
            Evidence("PTMC solutions across relations", "classical single-shear roots", ptmc_count, ""),
            Evidence("Relations satisfying all cofactor conditions", "CC1 ∧ CC2 ∧ CC3", cofactor_all, ""),
        ),
        tone="good" if relations else "neutral",
    )


def ebsd_audit_finding(audit: Mapping[str, Any]) -> Finding:
    n_points = int(audit["n_points"])
    n_indexed = int(audit["n_indexed"])
    fraction = float(audit["indexed_fraction"])
    so3 = float(audit["maximum_so3_residual"])
    return Finding(
        title="EBSD import and orientation audit",
        conclusion=f"Loaded {n_points} points; {n_indexed} are indexed ({fraction:.1%}).",
        rationale=f"The maximum SO(3) orientation residual is {sci(so3, 3)}; the importer does not guess vendor/sample frame corrections.",
        how="The file is parsed using its explicit format contract, orientation matrices are audited for proper-rotation consistency, and coordinate/indexing statistics are reported before any reconstruction is attempted.",
        physical_meaning="The map has passed the first integrity layer required before segmentation, OR fitting, parent reconstruction or trace validation.",
        limitation="A clean import does not validate phase mapping, segmentation thresholds, OR choice, or reconstruction quality; those are separate stages.",
        evidence=(
            Evidence("Points", "file content", n_points, ""),
            Evidence("Indexed fraction", "descriptive", f"{fraction:.5f}", ""),
            Evidence("SO(3) residual", "≈ 0", sci(so3, 3), "orientation-matrix audit"),
            Evidence("Coordinate dimension", "file geometry", audit["coordinate_dimension"], ""),
        ),
        tone="good" if so3 < 1.0e-8 else "warn",
    )


def am_existence_finding(checks: Mapping[str, Any], *, exact_ct_habits: int, bj_am: int, approx_ct_habits: int) -> Finding:
    ct_exact = bool(checks.get("ct_am_exact_compatible"))
    agree = bool(checks.get("ct_vs_ball_james_am_existence_agreement"))
    conclusion = (
        "CT and Ball–James independently agree that an exact single-variant A/M interface exists."
        if agree and ct_exact
        else "CT and Ball–James independently agree that no exact single-variant A/M interface exists."
        if agree
        else "CT and Ball–James do not currently give the same A/M existence classification."
    )
    return Finding(
        title="Single-variant A/M interface",
        conclusion=conclusion,
        rationale=(
            f"CT exact-compatible = {ct_exact}; exact CT habit branches = {exact_ct_habits}; "
            f"Ball–James exact A/M branches = {bj_am}."
        ),
        how="CT evaluates the CMC degeneracy independently; Ball–James evaluates the stretch/rank-one condition independently. Only the final physical existence classification is compared here.",
        physical_meaning=(
            "A homogeneous martensite variant can meet austenite through an exact compatible interface."
            if agree and ct_exact
            else "A homogeneous martensite variant does not satisfy the exact interface condition for this lattice state."
            if agree
            else "The cross-theory classification needs investigation before a physical conclusion is drawn."
        ),
        limitation=(
            f"CT also has {approx_ct_habits} nearest-degeneracy diagnostic habit branch(es); these are not exact solutions. "
            "PTMC may still obtain an exact macroscopic interface by introducing lattice-invariant shear."
        ),
        evidence=(
            Evidence("CT exact A/M", "CT native criterion", "yes" if ct_exact else "no", ""),
            Evidence("Exact CT habit branches", "exact only", exact_ct_habits, ""),
            Evidence("Ball–James exact A/M branches", "exact only", bj_am, ""),
            Evidence("Existence classification", "same physical yes/no question", "agree" if agree else "differ", ""),
        ),
        tone="good" if agree else "bad",
    )


def mm_audit_finding(audit: Mapping[str, Any] | None) -> Finding | None:
    if not audit:
        return None
    success = bool(audit.get("success"))
    count = int(audit.get("relation_count", 0))
    return Finding(
        title="Transformation twins — CT ↔ Mallard ↔ Ball–James",
        conclusion=(
            f"All {count} independently checked physical M/M relations agree to numerical precision."
            if success
            else "The independent CT ↔ Mallard ↔ Ball–James M/M audit did not pass."
        ),
        rationale=(
            "The authoritative audit performs the branch-specific configuration push-forward before comparing geometry, shear, rank-one reconstruction and rotation."
        ),
        how="CT, Mallard and the generic Ball–James M/M rank-one problem are evaluated as independent routes; the audit compares the resulting physical relations rather than raw source-native vector fields.",
        physical_meaning=(
            "For the checked transformation-twin relations, CT reproduces the classical physical twin geometry and shear."
            if success
            else "No equivalence claim should be made until the failed residual or coverage condition is resolved."
        ),
        limitation="Generic row-level M/M assignments can remain 'partial' because CT parent-reference directions and current-configuration Ball–James/PTMC shear directions are not directly comparable without the required push-forward.",
        evidence=(
            Evidence("Relations checked", "complete independent relation set", count, ""),
            Evidence("CT coverage", "all CT relations represented", "complete" if audit.get("ct_coverage_complete") else "incomplete", ""),
            Evidence("Max CT geometry angle", "≈ 0°", sci(audit.get("max_ct_geometry_angle_deg"), 3), ""),
            Evidence("Max Ball–James geometry angle", "≈ 0°", sci(audit.get("max_ball_james_geometry_angle_deg"), 3), ""),
            Evidence("Max shear relative residual", "≈ 0", sci(audit.get("max_shear_relative_residual"), 3), ""),
            Evidence("Max rank-one residual", "≈ 0", sci(audit.get("max_rank_one_residual"), 3), ""),
            Evidence("Max rotation residual", "≈ 0", sci(audit.get("max_rotation_residual"), 3), ""),
        ),
        tone="good" if success else "bad",
    )


def ptmc_finding(unified: Any, *, enabled: bool) -> Finding:
    if not enabled:
        return Finding(
            title="PTMC laminate compatibility",
            conclusion="PTMC was not evaluated in this run.",
            rationale="The deep twinning-LIS branch calculation was explicitly disabled.",
            how="Enable 'All twinning LIS — deep run' when a laminate compatibility calculation is required.",
            physical_meaning="No PTMC conclusion is being made from this run.",
            limitation="Not attempted is distinct from no solution.",
            tone="neutral",
        )
    rows = [row for row in _rows(unified, theory="ptmc", kind="ptmc_habit") if getattr(row, "exact", None) is True]
    true_ips = sum(bool(getattr(row, "metadata", {}).get("true_invariant_plane")) for row in rows)
    if rows:
        rank_residuals = [
            value
            for row in rows
            if (value := _abs(getattr(row, "residuals", {}).get("rank_one"))) is not None
        ]
        middle_residuals = [
            value
            for row in rows
            if (value := _abs(getattr(row, "residuals", {}).get("middle_stretch"))) is not None
        ]
        min_rank = min(rank_residuals) if rank_residuals else None
        min_middle = min(middle_residuals) if middle_residuals else None
        return Finding(
            title="PTMC laminate compatibility",
            conclusion=f"PTMC found {len(rows)} exact twinned-laminate habit branch(es).",
            rationale=(
                "This does not contradict a failed single-variant CT/Ball–James A/M test: PTMC changes the macroscopic deformation by introducing a lattice-invariant shear between martensite variants."
            ),
            how="For each admissible twinning LIS relation, PTMC solves the laminate parameter and habit rotation so that the macroscopic deformation satisfies an invariant-plane/rank-one condition.",
            physical_meaning="An internally twinned martensitic mixture can form an exact macroscopic A/M interface even when one homogeneous martensite variant cannot.",
            limitation="The result depends on the selected LIS hypothesis and branch set; it is not a statement that CT and PTMC are the same theory.",
            evidence=(
                Evidence("Exact PTMC habit branches", "true invariant-plane/rank-one solutions", len(rows), ""),
                Evidence("Rows marked true invariant plane", "metadata audit", true_ips, ""),
                Evidence("Best middle-stretch residual", "≈ 0", sci(min_middle, 3), ""),
                Evidence("Best rank-one residual", "≈ 0", sci(min_rank, 3), ""),
            ),
            tone="good",
        )
    return Finding(
        title="PTMC laminate compatibility",
        conclusion="PTMC was evaluated but no exact twinned-laminate habit branch was found.",
        rationale="The requested LIS branch set did not produce a macroscopic invariant-plane solution for this state.",
        how="The full requested twinning-LIS branch set was solved and filtered by the native PTMC exact-solution criteria.",
        physical_meaning="Within the tested PTMC hypothesis, no exact laminate interface was obtained.",
        limitation="This does not exclude other LIS mechanisms or a different physically justified PTMC model.",
        tone="warn",
    )


def or_comparison_finding(equivalence: Any, *, enabled: bool, tolerance_deg: float) -> Finding:
    if not enabled:
        return Finding(
            title="CT closing-gap ↔ PTMC orientation relationship",
            conclusion="CT/PTMC OR comparison was not requested in this run.",
            rationale="CT closing-gap OR generation or PTMC was disabled.",
            how="Both sources must be enabled before symmetry-reduced physical OR disorientation can be compared.",
            physical_meaning="No OR-equivalence conclusion is being made.",
            limitation="Not attempted is distinct from disagreement.",
            tone="neutral",
        )
    candidates = []
    for item in getattr(equivalence, "matches", ()):
        if _value(getattr(item, "family", "")) != "orientation_relationship":
            continue
        theories = {_value(getattr(item, "left_theory", "")), _value(getattr(item, "right_theory", ""))}
        if theories != {"cayron_ct", "ptmc"}:
            continue
        value = getattr(getattr(item, "residuals", None), "or_disorientation_deg", None)
        if value is not None:
            candidates.append(float(value))
    if not candidates:
        return Finding(
            title="CT closing-gap ↔ PTMC orientation relationship",
            conclusion="No CT/PTMC OR pair exposed a comparable physical OR in this run.",
            rationale="The matcher did not receive both OR observables for an assignable branch pair.",
            how="The comparison is only made after both OR matrices are re-expressed into a common Cartesian convention and reduced by proper crystallographic symmetry.",
            physical_meaning="No numerical OR-equivalence statement can be made from this run.",
            limitation="Missing observables are not treated as disagreement.",
            tone="neutral",
        )
    best = min(candidates)
    exact = best <= tolerance_deg
    return Finding(
        title="CT closing-gap ↔ PTMC orientation relationship",
        conclusion=(
            f"An exact numerical CT/PTMC OR coincidence is found within the requested tolerance; best separation = {best:.6g}°."
            if exact
            else f"No exact CT/PTMC OR coincidence is found; the closest assigned pair differs by {best:.6g}°."
        ),
        rationale=f"The best symmetry-reduced disorientation is compared with the explicit OR tolerance {tolerance_deg:.6g}°.",
        how="Each physical OR is first re-expressed into the same symmetric-metric Cartesian frame; proper parent/product crystal symmetries are then minimized over before the disorientation is reported.",
        physical_meaning=(
            "At least one CT closing-gap and PTMC OR prediction is numerically indistinguishable at the requested tolerance."
            if exact
            else "Some OR predictions may be geometrically close, but closeness is not being promoted to mathematical equivalence."
        ),
        limitation="This compares physical OR predictions only; it does not imply that CT and PTMC use the same derivation or mechanism.",
        evidence=(
            Evidence("Closest symmetry-reduced OR separation", f"≤ {tolerance_deg:.6g}° for numerical equivalence", f"{best:.9g}°", "within tolerance" if exact else "outside tolerance"),
            Evidence("Comparable assigned OR pairs", "descriptive", len(candidates), ""),
        ),
        tone="good" if exact else "warn",
    )


def cofactor_finding(unified: Any) -> Finding:
    rows = _rows(unified, theory="ball_james", kind="ball_james_mm")
    if not rows:
        return Finding(
            title="Cofactor conditions",
            conclusion="No Ball–James M/M rows were available for a cofactor summary.",
            rationale="Cofactor conditions are attached to the Ball–James martensite-pair branches.",
            how="CC1, CC2 and CC3 are retained separately for every M/M branch.",
            physical_meaning="No cofactor conclusion is available from this run.",
            tone="neutral",
        )
    cc1 = sum(bool(getattr(row, "metadata", {}).get("cofactor_cc1_satisfied")) for row in rows)
    cc2 = sum(bool(getattr(row, "metadata", {}).get("cofactor_cc2_satisfied")) for row in rows)
    cc3 = sum(bool(getattr(row, "metadata", {}).get("cofactor_cc3_satisfied")) for row in rows)
    all3 = sum(bool(getattr(row, "metadata", {}).get("cofactor_all_satisfied")) for row in rows)
    return Finding(
        title="Ball–James cofactor conditions",
        conclusion=f"{all3}/{len(rows)} M/M branch rows satisfy CC1, CC2 and CC3 simultaneously.",
        rationale="The three cofactor conditions are independent requirements and are therefore reported separately rather than collapsed into one residual.",
        how="Each Ball–James M/M rank-one branch carries native CC1, CC2 and CC3 evaluations from the cofactor backend.",
        physical_meaning="The count shows how many tested martensite-pair branches meet the full cofactor criterion for the current lattice state.",
        limitation="A row count is not a probability or ranking; symmetry-related branches can represent related physical families.",
        evidence=(
            Evidence("CC1 satisfied", f"out of {len(rows)}", cc1, ""),
            Evidence("CC2 satisfied", f"out of {len(rows)}", cc2, ""),
            Evidence("CC3 satisfied", f"out of {len(rows)}", cc3, ""),
            Evidence("CC1 ∧ CC2 ∧ CC3", f"out of {len(rows)}", all3, ""),
        ),
        tone="good" if all3 else "warn",
    )


def supercompatibility_finding(unified: Any, *, requested: bool, algebraic_tolerance: float) -> Finding:
    if not requested:
        return Finding(
            title="CT supercompatibility",
            conclusion="CT supercompatibility was not requested in this run.",
            rationale="The supercompatibility extension was disabled.",
            how="Enable the extension only when the A/M/M shear–shear condition is part of the scientific question.",
            physical_meaning="No supercompatibility claim is being made.",
            limitation="Not attempted is distinct from not satisfied.",
            tone="neutral",
        )
    exact_habits = [row for row in _rows(unified, theory="cayron_ct", kind="ct_am_habit") if getattr(row, "exact", None) is True]
    rows = _rows(unified, theory="cayron_ct", kind="ct_supercompatibility")
    if not exact_habits:
        return Finding(
            title="CT supercompatibility",
            conclusion="CT supercompatibility is not evaluable for this state because no exact CT A/M habit branch exists.",
            rationale="The A/M/M shear–shear residual is defined here from an exact A/M habit branch together with a CT M/M twin; there is no exact A/M branch to seed that calculation.",
            how="The implementation intentionally does not promote nearest-degeneracy A/M diagnostic planes into exact supercompatibility inputs.",
            physical_meaning="The current lattice state fails earlier at the exact A/M compatibility stage, so a positive exact A/M/M supercompatibility claim cannot be formed from these inputs.",
            limitation="This is 'not evaluable from an exact A/M branch', not a fabricated numerical failure value.",
            tone="warn",
        )
    if not rows:
        return Finding(
            title="CT supercompatibility",
            conclusion="Supercompatibility was requested, exact A/M branches exist, but no supercompatibility result row was produced.",
            rationale="This is a reporting/implementation condition that should be investigated rather than silently interpreted as failure.",
            how="The requested extension should generate one residual for each exact A/M habit × CT twin combination.",
            physical_meaning="No scientific supercompatibility conclusion should be made until the missing result is explained.",
            tone="bad",
        )
    residuals = [
        abs(float(getattr(row, "residuals", {}).get("ct_supercompatibility_dimensionless")))
        for row in rows
        if getattr(row, "residuals", {}).get("ct_supercompatibility_dimensionless") is not None
    ]
    best = min(residuals) if residuals else None
    satisfied = sum(value <= algebraic_tolerance for value in residuals)
    return Finding(
        title="CT supercompatibility",
        conclusion=(
            f"{satisfied}/{len(residuals)} evaluated A/M/M branch combinations satisfy the shear–shear residual within the algebraic tolerance."
            if residuals
            else "CT supercompatibility rows exist but expose no numerical residual."
        ),
        rationale=(
            f"The best |shear–shear residual| is {sci(best, 3)} and the project algebraic tolerance is {sci(algebraic_tolerance, 3)}."
            if best is not None
            else "No residual was available for classification."
        ),
        how="For every exact CT A/M habit and CT M/M twin, the native CT shear–shear compatibility residual is evaluated without replacing CT source-native vectors by Ball–James or PTMC objects.",
        physical_meaning="This tests the stronger CT A/M/M compatibility condition for the current state.",
        limitation="The residual is a CT-native condition; comparison with cofactor conditions is reported separately and must not be treated as an automatic identity unless the matched physical relation supports it.",
        evidence=(
            Evidence("Evaluated combinations", "exact A/M habit × CT twin", len(rows), ""),
            Evidence("Within algebraic tolerance", f"≤ {sci(algebraic_tolerance, 3)}", satisfied, ""),
            Evidence("Best residual", "0", sci(best, 3), ""),
        ),
        tone="good" if satisfied else "warn",
    )


def experiment_finding(equivalence: Any) -> Finding | None:
    rows = list(getattr(equivalence, "experiment_residuals", ()) or ())
    if not rows:
        return None
    return Finding(
        title="Theory ↔ experiment",
        conclusion=f"{len(rows)} theory/observation residual record(s) are available for experimental validation.",
        rationale="Experimental comparisons are kept component-wise; no overall theory winner score is constructed.",
        how="Observed ORs, habit planes, twin planes/directions, shears or shape strains are compared only when the same physical observable and configuration are available on both sides.",
        physical_meaning="This is the stage that determines which predicted branch actually approaches the measured specimen rather than merely agreeing with another theory.",
        limitation="Missing or configuration-incompatible observables remain N/A and are not converted into penalties.",
        tone="neutral",
    )
