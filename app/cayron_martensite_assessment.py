from __future__ import annotations

"""Pure presentation contract for a Cayron-CT martensitic assessment.

This module intentionally contains no crystallographic solver.  It only reads
an already calculated application response plus an already calculated unified
CT report and turns those authoritative outputs into a question-first,
professor-facing assessment.

No Ball--James or PTMC quantity is used to decide any Cayron-CT statement.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AssessmentStep:
    step_id: str
    question: str
    answer: str
    status: str
    formulae: tuple[str, ...]
    reasoning: str
    physical_meaning: str
    limitation: str
    evidence: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class CayronMartensiteAssessment:
    title: str
    overall_answer: str
    overall_status: str
    overall_reasoning: str
    steps: tuple[AssessmentStep, ...]



def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _result_from_response(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return _mapping(response.get("result", response))
    return _mapping(getattr(response, "result", {}))


def _project_from_response(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return _mapping(response.get("project_payload", response.get("project", {})))
    return _mapping(getattr(response, "project_payload", {}))


def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _row_kind(row: Any) -> str:
    if isinstance(row, Mapping):
        return _enum_text(row.get("prediction_kind", row.get("kind")))
    return _enum_text(getattr(row, "prediction_kind", ""))


def _row_theory(row: Any) -> str:
    if isinstance(row, Mapping):
        return _enum_text(row.get("theory"))
    return _enum_text(getattr(row, "theory", ""))


def _row_exact(row: Any) -> bool | None:
    if isinstance(row, Mapping):
        value = row.get("exact")
    else:
        value = getattr(row, "exact", None)
    return value if isinstance(value, bool) else None


def _row_attr(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _rows(unified: Any) -> tuple[Any, ...]:
    if unified is None:
        return ()
    if isinstance(unified, Mapping):
        raw = unified.get("rows", ())
    else:
        raw = getattr(unified, "rows", ())
    return tuple(raw or ())


def _ct_rows(unified: Any, kind: str | None = None) -> tuple[Any, ...]:
    selected = tuple(row for row in _rows(unified) if _row_theory(row) == "cayron_ct")
    if kind is None:
        return selected
    return tuple(row for row in selected if _row_kind(row) == kind)


def _first_transformation(project: Mapping[str, Any]) -> Mapping[str, Any]:
    transformations = project.get("transformations", ())
    if isinstance(transformations, Sequence) and transformations:
        item = transformations[0]
        return _mapping(item)
    return {}


def _correspondence_matrix(project: Mapping[str, Any]) -> Any:
    transformation = _first_transformation(project)
    return transformation.get(
        "correspondence_M_from_A",
        transformation.get("correspondence", None),
    )


def _compact_vector(value: Any) -> Any:
    if value is None:
        return None
    try:
        return [float(x) for x in value]
    except Exception:
        return value


def build_cayron_martensite_assessment(
    response: Any,
    unified: Any | None,
    *,
    closing_gap_requested: bool,
    supercompatibility_requested: bool,
) -> CayronMartensiteAssessment:
    """Build a CT-only martensitic assessment from already-computed outputs.

    The function is deliberately read-only.  It never invokes CT, group theory,
    twinning, orientation or compatibility solvers.
    """

    result = _result_from_response(response)
    project = _project_from_response(response)
    summary = _mapping(result.get("summary"))
    topology = _mapping(result.get("topology"))
    metric = _mapping(result.get("metric"))
    ct = _mapping(result.get("ct_detail"))

    topology_variants = int(topology.get("n_variants", summary.get("variant_count", 0)) or 0)
    stretch_variants = int(summary.get("variant_count", topology_variants) or 0)
    operators = int(topology.get("n_operators", summary.get("operator_count", 0)) or 0)
    subgroup_order = topology.get("subgroup_order")
    parent_group_order = topology.get("parent_group_order")
    product_group_order = topology.get("product_group_order")

    exact_am = bool(ct.get("exact_compatible", summary.get("ct_exact_compatible", False)))
    degeneracy_order = int(ct.get("degeneracy_order", summary.get("degeneracy_order", 0)) or 0)
    ct_reason = str(ct.get("reason", summary.get("ct_reason", "")) or "")
    eta = ct.get("eta_eigenvalues")
    mu = ct.get("generalized_mu")
    inertia = ct.get("inertia")
    nearest_index = ct.get("nearest_zero_index")
    nearest_residual = ct.get("nearest_zero_residual")
    exact_planes = tuple(ct.get("exact_habit_planes_parent_covectors", ()) or ())
    approx = _mapping(ct.get("approximate_diagnostic"))
    approx_planes = tuple(approx.get("candidate_planes_parent_covectors", ()) or ())
    approx_admissible = bool(approx.get("admissible_signature", False))

    mm_rows = _ct_rows(unified, "ct_mm_twin")
    habit_rows = _ct_rows(unified, "ct_am_habit")
    exact_habit_rows = tuple(row for row in habit_rows if _row_exact(row) is True)
    diagnostic_habit_rows = tuple(row for row in habit_rows if _row_exact(row) is False)
    closing_rows = _ct_rows(unified, "ct_closing_gap_or")
    super_rows = _ct_rows(unified, "ct_supercompatibility")

    correspondence = _correspondence_matrix(project)

    steps: list[AssessmentStep] = []

    steps.append(
        AssessmentStep(
            step_id="state",
            question="1. Can Cayron CT accept the supplied parent/product lattices and correspondence as a crystallographic transformation state?",
            answer=(
                "Yes. The active state passed project/metric/correspondence validation and produced a CT calculation."
                if result
                else "No active calculated CT state is available."
            ),
            status="validated" if result else "not evaluated",
            formulae=(
                r"u_M=C_{M\leftarrow A}u_A",
                r"p_M=C_{M\leftarrow A}^{-T}p_A",
                r"C\neq R",
            ),
            reasoning=(
                "CT starts from two crystallographic metrics and an explicit lattice correspondence. "
                "A successful calculated state means those inputs survived the application's domain checks; "
                "the correspondence is never relabelled as an orientation relationship."
            ),
            physical_meaning=(
                "The proposed parent→product lattice mapping is a valid object on which CT can operate. "
                "This is the starting point for a martensitic crystallographic description, not proof that a specimen transformed martensitically."
            ),
            limitation="Crystallographic admissibility alone does not establish transformation mechanism, kinetics, reversibility or shape-memory behavior.",
            evidence=(
                ("Exact correspondence C(M←A)", correspondence),
                ("Parent metric available", "yes" if metric.get("parent_metric") is not None else "no"),
                ("Product metric available", "yes" if metric.get("product_metric") is not None else "no"),
            ),
        )
    )

    topology_ok = topology_variants > 0 and operators > 0
    steps.append(
        AssessmentStep(
            step_id="topology",
            question="2. Does the correspondence generate a discrete symmetry-related martensitic variant/operator topology?",
            answer=(
                f"Yes. The exact CT correspondence topology contains {topology_variants} variant(s) and {operators} correspondence-operator class(es); the metric stretch layer exposes {stretch_variants} distinct stretch variant(s)."
                if topology_ok
                else "No discrete CT variant/operator topology was produced for the active state."
            ),
            status="reached" if topology_ok else "not reached",
            formulae=(
                r"H=G_A\cap C^{-1}G_M C",
                r"N_{variants}=|G_A|/|H|",
            ),
            reasoning=(
                "The common subgroup H contains the parent symmetries preserved by the correspondence into the product symmetry. "
                "The exact coset/groupoid construction then generates distinct variants and correspondence operators."
            ),
            physical_meaning=(
                "A non-trivial discrete family of symmetry-related product states is the CT crystallographic structure expected for a parent→martensite variant system."
            ),
            limitation="A variant topology is crystallographic evidence; it does not by itself prove that all variants occur experimentally.",
            evidence=(
                ("Parent point-group order", parent_group_order),
                ("Product point-group order", product_group_order),
                ("Common subgroup order", subgroup_order),
                ("CT correspondence variants", topology_variants),
                ("Metric stretch variants", stretch_variants),
                ("Operator classes", operators),
            ),
        )
    )

    if unified is None:
        mm_answer = "Not yet evaluated in the current CT theory run. Calculate/update the theory comparison to expose CT M/M constructions."
        mm_status = "not evaluated"
    elif mm_rows:
        mm_answer = f"Yes. The CT-native inventory contains {len(mm_rows)} martensite/martensite twin branch row(s)."
        mm_status = "reached"
    else:
        mm_answer = "No CT martensite/martensite twin branch was produced in the current CT inventory."
        mm_status = "not reached"
    steps.append(
        AssessmentStep(
            step_id="mm_twins",
            question="3. Can CT relate product variants to one another through its martensite/martensite twin constructions?",
            answer=mm_answer,
            status=mm_status,
            formulae=(
                r"C_{int}=C\,G\,C^{-1}",
                r"G^2=I,\quad (\det G,\operatorname{tr}G)=(-1,+1)\;\text{for Type I}",
                r"G^2=I,\quad (\det G,\operatorname{tr}G)=(+1,-1)\;\text{for Type II}",
                r"s_I^2=\operatorname{tr}(C_{int}^T M_M C_{int}M_M^{-1})-3",
                r"s_{II}^2=\operatorname{tr}(C_{int}M_M^{-1}C_{int}^TM_M)-3",
            ),
            reasoning=(
                "The CT twin engine follows Cayron Type-I Eqs. (17)–(20) and Type-II Eqs. (21)–(24): it starts from exact order-two parent symmetries, builds the intercorrespondence, "
                "and obtains the plane, direction and shear quantities in the real product metric."
            ),
            physical_meaning=(
                "Reachable CT M/M relations show that the product states are not merely unrelated lattice mappings: CT can connect variants by transformation-twin geometry."
            ),
            limitation=(
                "Native generator/sign rows are not automatically the number of unique physical twin systems; geometric deduplication and experimental occurrence remain separate questions."
            ),
            evidence=(
                ("Native CT M/M rows", len(mm_rows) if unified is not None else None),
                ("Example native labels", tuple(str(_row_attr(row, "branch_label", "")) for row in mm_rows[:4])),
            ),
        )
    )

    am_status = "reached" if exact_am else "not reached"
    am_answer = (
        f"Yes. Exact CT A/M compatibility is reached with degeneracy order {degeneracy_order}."
        if exact_am
        else f"No. Exact CT A/M compatibility is not reached. {ct_reason or 'No exact CMC degeneracy was found.'}"
    )
    steps.append(
        AssessmentStep(
            step_id="am_exact",
            question="4. Can one martensite variant meet the parent exactly according to Cayron's CMC criterion?",
            answer=am_answer,
            status=am_status,
            formulae=(
                r"CMC=C^T M_M C-M_A",
                r"(C^T M_M C)v_i=\mu_i M_Av_i",
                r"\eta_i=\mu_i-1",
            ),
            reasoning=(
                "The dimensional CMC is Cayron 2026 Eq. (32). CT requires an exact CMC degeneracy with the correct remaining-sign structure. "
                "The generalized eigenproblem is an exactly equivalent metric-native evaluation of the normalized CMC spectrum."
            ),
            physical_meaning=(
                "If reached, CT can construct an exact single-variant parent/martensite interface for the supplied lattice state. "
                "If not reached, the state can still possess a martensitic variant/twin topology; it is simply outside exact A/M compatibility."
            ),
            limitation="Failure of exact A/M compatibility is not, by itself, evidence that the product is not martensite.",
            evidence=(
                ("η spectrum", eta),
                ("Generalized μ spectrum", mu),
                ("CMC inertia (−,0,+)", inertia),
                ("Degeneracy order", degeneracy_order),
                ("Backend reason", ct_reason),
            ),
        )
    )

    if exact_am:
        near_answer = "The exact compatibility condition is already reached; no approximate replacement is needed."
        near_status = "exact reached"
    elif nearest_residual is not None:
        near_answer = (
            f"The nearest CT degeneracy residual is {float(nearest_residual):.10g}. "
            + ("A diagnostic plane construction is algebraically admissible." if approx_admissible else "The nearest-zero state does not have the sign structure required for a diagnostic habit-plane construction.")
        )
        near_status = "diagnostic only"
    else:
        near_answer = "No nearest-degeneracy diagnostic is available in the active result."
        near_status = "not available"
    steps.append(
        AssessmentStep(
            step_id="nearest",
            question="5. If exact A/M compatibility is absent, how close is this lattice state to the CT degeneracy condition?",
            answer=near_answer,
            status=near_status,
            formulae=(r"r_{CT}=\min_i|\eta_i|",),
            reasoning=(
                "The nearest generalized CMC eigenvalue is used only as a distance-to-degeneracy diagnostic. "
                "For diagnostic plane generation it may be projected to zero, while the original nonzero residual remains reported."
            ),
            physical_meaning="This measures closeness to exact CT A/M compatibility; it does not convert a near-compatible state into an exact one.",
            limitation="The diagnostic residual is not a probability of martensite and is not interchangeable with |λ₂−1| from another theory.",
            evidence=(
                ("Nearest η index", nearest_index),
                ("Nearest-degeneracy residual", nearest_residual),
                ("Diagnostic signature admissible", "yes" if approx_admissible else "no"),
                ("Diagnostic candidate plane count", len(approx_planes)),
            ),
        )
    )

    if exact_planes:
        hp_answer = f"Yes. CT reaches {len(exact_planes)} exact A/M habit-plane covector branch(es)."
        hp_status = "reached"
    elif exact_am and degeneracy_order == 3:
        hp_answer = (
            "Exact third-order CMC compatibility is reached, but no unique habit-plane branch is defined: "
            "the pulled-back product metric coincides with the parent metric under the correspondence."
        )
        hp_status = "not uniquely defined"
    elif exact_am:
        hp_answer = "Exact CT A/M compatibility is reached, but the active result exposes no explicit habit-plane branch; inspect the numerical audit before making an interface claim."
        hp_status = "not available"
    else:
        hp_answer = "No exact CT A/M habit plane is reachable because the exact CMC compatibility condition is absent for this state."
        hp_status = "not reached"
    steps.append(
        AssessmentStep(
            step_id="habit",
            question="6. Can CT reach an exact parent/martensite habit plane for this state?",
            answer=hp_answer,
            status=hp_status,
            formulae=(
                r"p_A=M_A\left(\sqrt{\eta_+}\,v_+\pm\sqrt{-\eta_-}\,v_-\right)\quad\text{(first-order exact degeneracy)}",
            ),
            reasoning=(
                "Habit-plane construction is downstream of exact CMC degeneracy. "
                "Approximate nearest-degeneracy planes, when available, remain explicitly diagnostic and are never promoted to exact branches."
            ),
            physical_meaning="An exact CT habit plane is the crystallographic plane on which a single martensite variant can meet the parent under the CT metric condition.",
            limitation="A predicted crystallographic habit plane still requires experimental validation against observed interfaces/traces.",
            evidence=(
                ("Exact CT habit planes", tuple(_compact_vector(item) for item in exact_planes)),
                ("Approximate diagnostic planes", tuple(_compact_vector(item) for item in approx_planes)),
            ),
        )
    )

    exact_shear_rows = tuple(row for row in exact_habit_rows if _row_attr(row, "shape_vector_parent_crystal") is not None)
    if exact_planes and exact_shear_rows:
        shear_answer = f"Yes. Exact CT A/M shear/displacement data are exposed for {len(exact_shear_rows)} exact habit branch(es)."
        shear_status = "reached"
    elif exact_planes:
        shear_answer = "The exact habit plane exists, but the current unified CT view has not exposed the corresponding native d-vector row yet."
        shear_status = "not evaluated"
    else:
        shear_answer = "Not exactly. Without an exact CT A/M habit seed, an exact A/M d-vector is not claimed."
        shear_status = "not reachable exactly"
    steps.append(
        AssessmentStep(
            step_id="smc",
            question="7. Can CT construct the native parent/martensite shear associated with an exact habit plane?",
            answer=shear_answer,
            status=shear_status,
            formulae=(
                r"SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}",
                r"d_A=SMC\,m_A",
            ),
            reasoning=(
                "Cayron's SMC is the dimensional construction of Eq. (41), computed directly from the two metrics and the correspondence. "
                "The physically interpreted CT IPS displacement/shear d_A is attached to an eligible habit-plane covector m_A."
            ),
            physical_meaning="This is Cayron's native d-vector construction for the A/M interface; it is not relabelled as a shape vector from another theory.",
            limitation="A diagnostic near-degeneracy d-vector, if exposed, remains approximate and cannot seed exact CT supercompatibility.",
            evidence=(
                ("SMC matrix available", "yes" if metric.get("smc_dimensional") is not None else "no"),
                ("Exact CT A/M rows with d", len(exact_shear_rows)),
                ("Approximate CT A/M diagnostic rows", len(diagnostic_habit_rows)),
            ),
        )
    )

    if not closing_gap_requested:
        or_answer = "Not requested in the current CT run. Enable CT closing-gap ORs to evaluate this construction."
        or_status = "not requested"
    elif closing_rows:
        or_answer = f"Yes. CT reaches {len(closing_rows)} closing-gap orientation candidate row(s) from its twin parallelism constraints."
        or_status = "reached"
    else:
        or_answer = "Closing-gap ORs were requested, but no CT closing-gap orientation candidate was produced for the current state."
        or_status = "not reached"
    steps.append(
        AssessmentStep(
            step_id="closing_gap",
            question="8. Can CT reach parent/martensite orientation relationships through its closing-gap construction?",
            answer=or_answer,
            status=or_status,
            formulae=(
                r"K_{1,A}\parallel K_{1,M},\qquad \eta_{1,A}\parallel\eta_{1,M}\quad\text{(Type I)}",
                r"\eta_{2,A}\parallel\eta_{2,M},\qquad K_{2,A}\parallel K_{2,M}\quad\text{(Type II)}",
            ),
            reasoning=(
                "The CT orientation adapter constructs proper rotations satisfying the native twin plane/direction parallelisms. "
                "A unique natural OR is never inferred from metrics+correspondence alone."
            ),
            physical_meaning="Closing-gap candidates show which parent/product orientations CT can geometrically reach while closing the twin-element mismatch.",
            limitation="Multiple candidates are retained. None is called the preferred/natural OR unless an external natural-OR hypothesis is explicitly supplied.",
            evidence=(
                ("Closing-gap requested", "yes" if closing_gap_requested else "no"),
                ("CT closing-gap candidate rows", len(closing_rows)),
                ("Example native labels", tuple(str(_row_attr(row, "branch_label", "")) for row in closing_rows[:4])),
            ),
        )
    )

    exact_super_seed = bool(exact_am and exact_planes)
    if not exact_super_seed:
        if exact_am and degeneracy_order == 3:
            super_answer = "Not evaluable as a unique A/M/M branch condition: third-order CMC compatibility does not provide a unique exact A/M habit-plane seed."
        else:
            super_answer = "Not evaluable. CT supercompatibility requires an exact CT A/M habit/shear seed, which this state does not provide."
        super_status = "not evaluable"
    elif not supercompatibility_requested:
        super_answer = "Not requested in the current CT run. Enable CT supercompatibility to evaluate the exact A/M/M condition."
        super_status = "not requested"
    elif super_rows:
        satisfied = sum(1 for row in super_rows if _row_exact(row) is True)
        if satisfied:
            super_answer = f"Yes for {satisfied} branch row(s): the CT supercompatibility residual satisfies the exact tolerance."
            super_status = "reached"
        else:
            super_answer = f"Evaluated on {len(super_rows)} branch row(s), but no branch satisfies the exact CT supercompatibility condition."
            super_status = "not reached"
    else:
        super_answer = "The exact A/M seed exists and supercompatibility was requested, but no eligible CT A/M/M branch row was produced."
        super_status = "not reached"
    steps.append(
        AssessmentStep(
            step_id="supercompatibility",
            question="9. Can CT reach its A/M/M supercompatibility condition for this state?",
            answer=super_answer,
            status=super_status,
            formulae=(r"2(m_A^T n)d_A=a",),
            reasoning=(
                "The CT shear/shear condition is downstream of an exact A/M habit/shear seed and an eligible M/M twin. "
                "The application deliberately refuses to substitute the approximate nearest-degeneracy construction."
            ),
            physical_meaning="When satisfied, the A/M and M/M shear constructions meet the additional CT supercompatibility condition.",
            limitation="Not evaluable is scientifically different from failed: absence of an exact A/M seed prevents the question from being posed exactly.",
            evidence=(
                ("Exact CT A/M habit seed", "yes" if exact_super_seed else "no"),
                ("Supercompatibility requested", "yes" if supercompatibility_requested else "no"),
                ("CT supercompatibility rows", len(super_rows)),
            ),
        )
    )

    if unified is None:
        overall_status = "incomplete CT inventory"
        overall_answer = (
            "CT already constructs a valid parent→product state and a discrete variant/operator topology, "
            "but the CT M/M and closing-gap inventory has not yet been calculated in this theory workspace."
        )
        overall_reasoning = (
            "The app can already answer the A/M CMC question from the calculated transformation state, but it should not claim a complete CT martensitic microstructure assessment until the CT-native branch inventory is available."
        )
    elif topology_ok and mm_rows:
        overall_status = "CT-consistent martensitic crystallography"
        compatibility_clause = (
            "Exact parent/martensite compatibility is also reached."
            if exact_am
            else "Exact parent/martensite compatibility is not reached for the supplied lattice parameters."
        )
        overall_answer = (
            "Yes at the crystallographic-theory level: CT constructs a symmetry-related parent→product variant system and reaches CT-native martensite/martensite relations. "
            + compatibility_clause
        )
        overall_reasoning = (
            "The positive statement is intentionally limited to crystallographic consistency within Cayron CT. "
            "A missing exact A/M degeneracy changes interface compatibility, not the existence of the CT variant/twin construction."
        )
    elif topology_ok:
        overall_status = "partial CT martensitic construction"
        overall_answer = (
            "CT constructs a discrete parent→product variant/operator topology, but the current CT inventory does not reach an M/M twin branch. "
            "The app therefore does not claim a complete twinned martensitic microstructure from CT alone."
        )
        overall_reasoning = (
            "Variant topology and exact A/M compatibility answer different questions. The absence of an M/M row in the current CT inventory is reported directly rather than hidden by an overall score."
        )
    else:
        overall_status = "CT martensitic construction not established"
        overall_answer = "The active result does not establish the discrete CT variant/operator construction needed for a CT martensitic crystallographic description."
        overall_reasoning = "The app does not infer martensite from an alloy name or from a single residual; it reports only the structures actually produced by the CT calculation."

    overall_reasoning += (
        " CT alone cannot experimentally prove that a specimen transformed by a martensitic mechanism, and it cannot prove shape-memory functionality; "
        "those require experimental/thermomechanical evidence in addition to crystallography."
    )

    steps.append(
        AssessmentStep(
            step_id="overall",
            question="10. So, does Cayron CT support a martensitic crystallographic description of the supplied state?",
            answer=overall_answer,
            status=overall_status,
            formulae=(),
            reasoning=overall_reasoning,
            physical_meaning=(
                "This final statement separates three issues that must not be conflated: CT martensitic crystallography, exact interface compatibility, and experimentally demonstrated SMA functionality."
            ),
            limitation="The conclusion is theory-level crystallographic evidence, not an experimental phase-identification or functional-material verdict.",
            evidence=(
                ("Variant/operator topology", "reached" if topology_ok else "not reached"),
                ("CT M/M construction", "reached" if mm_rows else ("not evaluated" if unified is None else "not reached")),
                ("Exact CT A/M compatibility", "reached" if exact_am else "not reached"),
                ("Exact CT A/M habit plane", hp_status),
                ("CT supercompatibility", super_status),
            ),
        )
    )

    return CayronMartensiteAssessment(
        title="Cayron CT — martensitic crystallography assessment",
        overall_answer=overall_answer,
        overall_status=overall_status,
        overall_reasoning=overall_reasoning,
        steps=tuple(steps),
    )
