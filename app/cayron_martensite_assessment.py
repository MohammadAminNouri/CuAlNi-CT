from __future__ import annotations

"""Read-only, source-disciplined Cayron-CT martensitic assessment.

This module contains no crystallographic solver.  It only interprets an
already-calculated application response and an already-calculated unified CT
report.  CT, Ball--James, PTMC and experiment remain separate theories/data
sources.

Notation used by the app:
    C := C_(M<-A), so u_M = C u_A.

Cayron's 2026 paper denotes this SAME numerical correspondence matrix by
C^(M->A), even though it acts on coordinates from A to M:
    u_M = C_Cayron^(M->A) u_A.

Therefore
    C_app_(M<-A) = C_Cayron^(M->A)
and Cayron's opposite correspondence is
    C_Cayron^(A->M) = C_app_(M<-A)^(-1).
"""

from dataclasses import dataclass
import math
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


def _row_residual(row: Any, name: str) -> float | None:
    residuals = _row_attr(row, "residuals", {})
    if not isinstance(residuals, Mapping):
        return None
    value = residuals.get(name)
    if value is None:
        return None
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
        return _mapping(transformations[0])
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


def _algebraic_tolerance(project: Mapping[str, Any]) -> float:
    policy = _mapping(project.get("numerical_policy"))
    for key in ("algebraic", "algebraic_tolerance"):
        if key not in policy:
            continue
        try:
            value = float(policy[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return 1.0e-10


def build_cayron_martensite_assessment(
    response: Any,
    unified: Any | None,
    *,
    closing_gap_requested: bool,
    supercompatibility_requested: bool,
) -> CayronMartensiteAssessment:
    """Build a CT-only assessment from already-computed outputs."""

    result = _result_from_response(response)
    project = _project_from_response(response)
    summary = _mapping(result.get("summary"))
    topology = _mapping(result.get("topology"))
    metric = _mapping(result.get("metric"))
    ct = _mapping(result.get("ct_detail"))

    topology_variants = int(
        topology.get("n_variants", summary.get("variant_count", 0)) or 0
    )
    stretch_variants = int(summary.get("variant_count", topology_variants) or 0)
    operators = int(
        topology.get("n_operators", summary.get("operator_count", 0)) or 0
    )
    subgroup_order = topology.get("subgroup_order")
    parent_group_order = topology.get("parent_group_order")
    product_group_order = topology.get("product_group_order")

    exact_am = bool(
        ct.get("exact_compatible", summary.get("ct_exact_compatible", False))
    )
    degeneracy_order = int(
        ct.get("degeneracy_order", summary.get("degeneracy_order", 0)) or 0
    )
    ct_reason = str(ct.get("reason", summary.get("ct_reason", "")) or "")
    eta = ct.get("eta_eigenvalues")
    mu = ct.get("generalized_mu")
    inertia = ct.get("inertia")
    nearest_index = ct.get("nearest_zero_index")
    nearest_residual = ct.get("nearest_zero_residual")
    exact_planes = tuple(
        ct.get("exact_habit_planes_parent_covectors", ()) or ()
    )
    approx = _mapping(ct.get("approximate_diagnostic"))
    approx_planes = tuple(
        approx.get("candidate_planes_parent_covectors", ()) or ()
    )
    approx_admissible = bool(approx.get("admissible_signature", False))

    mm_rows = _ct_rows(unified, "ct_mm_twin")
    habit_rows = _ct_rows(unified, "ct_am_habit")
    exact_habit_rows = tuple(
        row for row in habit_rows if _row_exact(row) is True
    )
    diagnostic_habit_rows = tuple(
        row for row in habit_rows if _row_exact(row) is False
    )
    closing_rows = _ct_rows(unified, "ct_closing_gap_or")
    super_rows = _ct_rows(unified, "ct_supercompatibility")

    correspondence = _correspondence_matrix(project)
    tolerance = _algebraic_tolerance(project)

    steps: list[AssessmentStep] = []

    # 1 ------------------------------------------------------------------
    steps.append(
        AssessmentStep(
            step_id="state",
            question=(
                "1. What lattice correspondence is being supplied to CT, "
                "and is the resulting crystallographic state valid?"
            ),
            answer=(
                "The active parent/product metrics and correspondence passed "
                "the application's state validation and produced a CT result."
                if result
                else "No active calculated CT state is available."
            ),
            status="validated" if result else "not evaluated",
            formulae=(
                r"u_M=C_{M\leftarrow A}u_A",
                r"C_{M\leftarrow A}^{\mathrm{app}}"
                r"\equiv C_{\mathrm{Cayron}}^{M\to A}",
                r"C_{\mathrm{Cayron}}^{A\to M}"
                r"=(C_{M\leftarrow A}^{\mathrm{app}})^{-1}",
                r"p_M=C_{M\leftarrow A}^{-T}p_A",
            ),
            reasoning=(
                "The app stores the correspondence by its coordinate action: "
                "parent direct coordinates enter and product direct coordinates leave. "
                "Cayron uses the superscript M→A for that same numerical matrix. "
                "This notation crosswalk is explicit so the correspondence cannot be "
                "mistaken for its inverse, for an orientation matrix, or for a distortion."
            ),
            physical_meaning=(
                "The supplied metrics and correspondence define a valid crystallographic "
                "state on which correspondence theory can operate. That is an input/state "
                "statement, not experimental proof of a martensitic transformation."
            ),
            limitation=(
                "A valid crystallographic state does not establish transformation "
                "mechanism, kinetics, reversibility or shape-memory functionality."
            ),
            evidence=(
                ("Exact correspondence C(M←A)", correspondence),
                (
                    "Parent metric available",
                    "yes" if metric.get("parent_metric") is not None else "no",
                ),
                (
                    "Product metric available",
                    "yes" if metric.get("product_metric") is not None else "no",
                ),
            ),
        )
    )

    # 2 ------------------------------------------------------------------
    topology_ok = topology_variants > 0 and operators > 0
    steps.append(
        AssessmentStep(
            step_id="topology",
            question=(
                "2. How many correspondence variants and intercorrespondence "
                "operator classes does this exact state generate?"
            ),
            answer=(
                f"The exact correspondence topology contains {topology_variants} "
                f"variant(s) and {operators} double-coset operator class(es); "
                f"the metric stretch layer exposes {stretch_variants} stretch variant(s)."
                if topology_ok
                else "No discrete CT correspondence topology was produced."
            ),
            status="reached" if topology_ok else "not reached",
            formulae=(
                r"H_C^A=G_A\cap C^{-1}G_MC",
                r"N_{\mathrm{corr.\,variants}}=\frac{|G_A|}{|H_C^A|}",
                r"\mathrm{variants}:~G_A/H_C^A",
                r"\mathrm{operators}:~H_C^A\backslash G_A/H_C^A",
            ),
            reasoning=(
                "Correspondence variants are left cosets; intercorrespondence operators "
                "are double cosets. Their counts answer different group-theoretic questions. "
                "The operator count is therefore determined by the current symmetry groups, "
                "subgroup embedding and correspondence; it is not a universal constant. "
                "For the published B2→B19′ NiTi correspondence the validated benchmark "
                "has 12 correspondence variants and 7 operator classes, but another valid "
                "correspondence can legitimately give a different double-coset count."
            ),
            physical_meaning=(
                "This is the discrete symmetry structure from which CT organizes "
                "product variants and the relations between them."
            ),
            limitation=(
                "A calculated variant/operator topology does not imply that every "
                "variant or operator is experimentally populated."
            ),
            evidence=(
                ("Parent point-group order", parent_group_order),
                ("Product point-group order", product_group_order),
                ("Correspondence subgroup order", subgroup_order),
                ("CT correspondence variants", topology_variants),
                ("CT double-coset operator classes", operators),
                ("Metric stretch variants", stretch_variants),
            ),
        )
    )

    # 3 ------------------------------------------------------------------
    if unified is None:
        mm_answer = (
            "The unified CT branch inventory has not yet been calculated for "
            "the current theory-comparison request."
        )
        mm_status = "not evaluated"
    elif mm_rows:
        mm_answer = (
            f"The current CT inventory contains {len(mm_rows)} native M/M "
            "twin branch row(s)."
        )
        mm_status = "reached"
    else:
        mm_answer = (
            "The unified CT inventory was evaluated, but it produced no native "
            "M/M twin branch for this state."
        )
        mm_status = "not reached"

    steps.append(
        AssessmentStep(
            step_id="mm_twins",
            question=(
                "3. Which martensite/martensite transformation-twin "
                "geometries are actually reachable from the parent symmetries?"
            ),
            answer=mm_answer,
            status=mm_status,
            formulae=(
                r"C_{\mathrm{int}}=C\,G_A\,C^{-1}",
                r"G_A^2=I,\quad(\det G_A,\operatorname{tr}G_A)=(-1,+1)"
                r"\quad\mathrm{Type~I}",
                r"p_M=C^{-T}p_A,\qquad "
                r"n_M=\frac{M_M^{-1}p_M}{\sqrt{p_M^TM_M^{-1}p_M}}",
                r"s_I^2=\operatorname{tr}"
                r"(C_{\mathrm{int}}^TM_MC_{\mathrm{int}}M_M^{-1})-3",
                r"a_M=-(C_{\mathrm{int}}+I)n_M",
                r"\mathrm{Type~I~twin~system}:\quad(p_M,a_M)",
                r"G_A^2=I,\quad(\det G_A,\operatorname{tr}G_A)=(+1,-1)"
                r"\quad\mathrm{Type~II}",
                r"a_M=C\,a_A,\qquad p_M=M_Ma_M",
                r"C_{\mathrm{int}}^{\ast}=C_{\mathrm{int}}^{-T}"
                r"=M_MC_{\mathrm{int}}M_M^{-1}",
                r"s_{II}^2=\operatorname{tr}"
                r"(C_{\mathrm{int}}M_M^{-1}C_{\mathrm{int}}^TM_M)-3",
                r"jp_M=-(C_{\mathrm{int}}^{\ast}-I)p_M",
                r"\mathrm{Type~II~twin~system}:\quad(jp_M,a_M)",
            ),
            reasoning=(
                "CT starts from exact order-two parent symmetries. A parent reflection "
                "generates a Type-I construction whose rational twin plane is obtained "
                "directly by reciprocal correspondence; the product metric then supplies "
                "the unit normal, shear magnitude and complementary shear direction. "
                "A parent 180° rotation generates a Type-II construction whose rational "
                "twin direction is obtained by direct correspondence. Its metric-dual "
                "covector p_M=M_Ma_M is acted on by the reciprocal intercorrespondence "
                "operator C_int* = C_int^(-T) to obtain the generally irrational junction "
                "plane jp_M. Thus the full Type-I system is (p_M,a_M) and the full "
                "Type-II system is (jp_M,a_M). The complete relations needed for the "
                "interpretation are written explicitly here rather than referenced only "
                "by paper equation numbers."
            ),
            physical_meaning=(
                "A reachable CT M/M branch gives an actual transformation-twin geometry "
                "(plane, direction and shear), rather than merely saying that two product "
                "variants exist."
            ),
            limitation=(
                "Native generator/sign rows are not automatically the number of unique "
                "physical twin systems. Projective/sign equivalence and experimental "
                "occurrence are separate issues."
            ),
            evidence=(
                ("Native CT M/M rows", len(mm_rows) if unified is not None else None),
                (
                    "Example native labels",
                    tuple(
                        str(_row_attr(row, "branch_label", ""))
                        for row in mm_rows[:4]
                    ),
                ),
            ),
        )
    )

    # 4 ------------------------------------------------------------------
    am_answer = (
        f"Exact single-variant A/M compatibility is reached with CMC "
        f"degeneracy order {degeneracy_order}."
        if exact_am
        else (
            "Exact single-variant A/M compatibility is not reached. "
            + (ct_reason or "No exact CMC degeneracy was found.")
        )
    )
    steps.append(
        AssessmentStep(
            step_id="am_exact",
            question=(
                "4. Does one martensite variant satisfy exact "
                "austenite/martensite metric compatibility?"
            ),
            answer=am_answer,
            status="reached" if exact_am else "not reached",
            formulae=(
                r"CMC=C^T M_M C-M_A",
                r"u_A^TCMC\,u_A=0",
                r"(C^TM_MC)v_i=\mu_iM_Av_i,\qquad\eta_i=\mu_i-1",
                r"\widehat{CMC}=M_A^{-1/2}CMC\,M_A^{-1/2}",
                r"\mathrm{first~order}:~\eta_i=0,\quad\eta_j\eta_k<0",
                r"\mathrm{second~order}:~\eta_i=\eta_j=0",
                r"\mathrm{third~order}:~\eta_1=\eta_2=\eta_3=0",
            ),
            reasoning=(
                "Cayron's dimensional CMC is the difference between the product metric "
                "pulled back by correspondence and the parent metric. Cayron denotes the "
                "eigenvalues of the dimensional CMC in an orthonormal eigenbasis by q_i. "
                "The app evaluates the same degeneracy/inertia question with the "
                "generalized problem above. The η_i are eigenvalues of a metric-normalized "
                "congruent CMC, not numerically the same quantities as Cayron's dimensional "
                "q_i in general. Congruence preserves rank and inertia, so the zero "
                "multiplicity and sign structure used to classify exact CMC degeneracy are "
                "equivalent."
            ),
            physical_meaning=(
                "First-order degeneracy gives two exact A/M habit planes; second-order "
                "degeneracy gives one; third-order degeneracy means complete metric match "
                "under the correspondence."
            ),
            limitation=(
                "Failure of exact A/M compatibility does not imply that the product is "
                "not martensite and does not invalidate the CT variant/twin topology."
            ),
            evidence=(
                ("Normalized generalized η spectrum", eta),
                ("Generalized μ spectrum", mu),
                ("CMC inertia (negative, zero, positive)", inertia),
                ("Degeneracy order", degeneracy_order),
                ("Backend reason", ct_reason),
            ),
        )
    )

    # 5 ------------------------------------------------------------------
    if exact_am:
        near_answer = (
            f"Yes. The current CMC is exactly degenerate at order {degeneracy_order}; "
            "no approximate distance-to-degeneracy diagnostic is needed."
        )
        near_status = "exact reached"
        near_formulae: tuple[str, ...] = ()
        near_reasoning = (
            "Exact compatibility has already been established by the CMC zero-eigenvalue "
            "multiplicity and inertia classification. Reporting an approximate nearest-zero "
            "distance here would add no scientific information and could incorrectly make an "
            "exact state look approximate."
        )
        near_physical_meaning = (
            "The state lies on an exact CMC compatibility manifold for the active numerical "
            "policy; the appropriate exact habit-plane construction is handled in the next step."
        )
        near_limitation = (
            "Exact algebraic compatibility remains a theory-level crystallographic statement; "
            "it is not experimental evidence of transformation kinetics or functionality."
        )
        near_evidence: tuple[tuple[str, Any], ...] = (
            ("CMC degeneracy order", degeneracy_order),
            ("CMC inertia (negative, zero, positive)", inertia),
        )
    elif nearest_residual is not None:
        near_answer = (
            f"No exact CMC degeneracy is present. The app's normalized CMC "
            f"degeneracy residual is {float(nearest_residual):.10g}. "
            + (
                "The normalized sign structure also permits a diagnostic plane construction."
                if approx_admissible
                else "The normalized sign structure does not permit a diagnostic habit-plane construction."
            )
        )
        near_status = "diagnostic only"
        near_formulae = (r"r_{\mathrm{app}}=\min_i|\eta_i|",)
        near_reasoning = (
            "This is an app-defined, dimensionless diagnostic based on the normalized "
            "generalized CMC spectrum. It reports how close one normalized eigenvalue "
            "is to zero while retaining the original sign information. It is not "
            "Cayron's lattice-parameter distance to the C1/C2/C3 equality/inequality "
            "boundaries, and it is never substituted for an exact zero."
        )
        near_physical_meaning = (
            "It is useful for numerical proximity and parameter exploration; it is not "
            "an exact compatibility condition."
        )
        near_limitation = (
            "The residual is not a probability, energy, hysteresis measure or experimental "
            "confidence, and it must not be compared numerically with |λ₂−1| as though the "
            "two theories used the same native scalar."
        )
        near_evidence = (
            ("Nearest η index", nearest_index),
            ("App normalized CMC degeneracy residual", nearest_residual),
            (
                "Diagnostic sign structure admissible",
                "yes" if approx_admissible else "no",
            ),
            ("Diagnostic candidate plane count", len(approx_planes)),
        )
    else:
        near_answer = "No exact CMC degeneracy and no normalized proximity diagnostic are available."
        near_status = "not available"
        near_formulae = ()
        near_reasoning = (
            "The assessment does not manufacture a distance measure when the backend has not "
            "exposed the normalized spectral diagnostic."
        )
        near_physical_meaning = "No proximity statement is made."
        near_limitation = "Absence of a diagnostic value is not itself a compatibility verdict."
        near_evidence = ()

    steps.append(
        AssessmentStep(
            step_id="nearest",
            question=(
                "5. Is the current state exactly CMC-degenerate, and if not, what is its "
                "normalized distance from degeneracy?"
            ),
            answer=near_answer,
            status=near_status,
            formulae=near_formulae,
            reasoning=near_reasoning,
            physical_meaning=near_physical_meaning,
            limitation=near_limitation,
            evidence=near_evidence,
        )
    )

    # 6 ------------------------------------------------------------------
    if exact_planes:
        hp_answer = (
            f"CT reaches {len(exact_planes)} exact A/M habit-plane "
            "covector branch(es)."
        )
        hp_status = "reached"
    elif exact_am and degeneracy_order == 3:
        hp_answer = (
            "Third-order CMC compatibility is exact, but no unique habit plane exists "
            "because the pulled-back product metric matches the parent metric completely."
        )
        hp_status = "not uniquely defined"
    elif exact_am:
        hp_answer = (
            "Exact CMC compatibility is reached, but the active result exposes no "
            "explicit habit-plane branch; no interface direction is invented."
        )
        hp_status = "not available"
    else:
        hp_answer = (
            "No exact A/M habit plane is claimed because exact CMC degeneracy is absent."
        )
        hp_status = "not reached"

    if exact_am and degeneracy_order == 1:
        habit_formulae = (
            r"q_-X_-^2+q_+X_+^2=0,\qquad q_-<0<q_+",
            r"m_d^\pm\propto"
            r"\sqrt{q_+}\,e_+^\ast\pm\sqrt{-q_-}\,e_-^\ast",
            r"m_A^\pm=P^{-T}m_d^\pm",
            r"m_A^\pm\propto M_A"
            r"\left(\sqrt{\eta_+}v_+\pm\sqrt{-\eta_-}v_-\right)",
        )
        habit_reasoning = (
            "First-order degeneracy has exactly one zero CMC eigenvalue and two nonzero "
            "eigenvalues of opposite sign. The quadratic compatibility cone therefore "
            "factorizes into two linear factors, producing two projectively distinct exact "
            "habit-plane covectors. The final expression is the app's metric-native form "
            "for M_A-orthonormal generalized eigenvectors."
        )
    elif exact_am and degeneracy_order == 2:
        habit_formulae = (
            r"\eta_i=\eta_j=0,\qquad\eta_k\neq0",
            r"u_A^TCMC\,u_A="
            r"\eta_k\left(v_k^TM_Au_A\right)^2=0",
            r"v_k^TM_Au_A=0",
            r"m_A\propto M_Av_k",
        )
        habit_reasoning = (
            "Second-order degeneracy has two zero normalized generalized CMC eigenvalues "
            "and one nonzero eigenvalue. In an M_A-orthonormal generalized eigenbasis, the "
            "quadratic compatibility condition therefore contains only the square of the "
            "coordinate along the nonzero-eigenvalue direction. The double plane collapses "
            "to one unique projective plane. Its reciprocal covector is m_A ∝ M_A v_k. "
            "This is the correct construction for Cayron's D1/D2 second-order cases; the "
            "first-order ± two-plane formula is deliberately not displayed here."
        )
    elif exact_am and degeneracy_order == 3:
        habit_formulae = (
            r"CMC=0",
            r"u_A^TCMC\,u_A=0\quad\mathrm{for~every}\ u_A",
        )
        habit_reasoning = (
            "Third-order degeneracy means the pulled-back product metric and parent metric "
            "coincide completely. Every direction satisfies the metric equality, so CT does "
            "not select a unique habit plane from CMC degeneracy alone."
        )
    else:
        habit_formulae = ()
        habit_reasoning = (
            "An exact habit-plane factorization is only performed after exact CMC degeneracy "
            "has been established. Approximate diagnostic planes are kept separate and are "
            "never promoted to exact A/M solutions."
        )

    habit_evidence: tuple[tuple[str, Any], ...] = (
        (
            "Exact CT habit planes m_A",
            tuple(_compact_vector(item) for item in exact_planes),
        ),
    )
    if not exact_am and approx_planes:
        habit_evidence += (
            (
                "Approximate diagnostic planes (not exact)",
                tuple(_compact_vector(item) for item in approx_planes),
            ),
        )

    steps.append(
        AssessmentStep(
            step_id="habit",
            question=(
                "6. Given the detected CMC degeneracy order, what exact A/M habit-plane "
                "geometry follows from that specific order?"
            ),
            answer=hp_answer,
            status=hp_status,
            formulae=habit_formulae,
            reasoning=(
                habit_reasoning
                + " The A/M habit-plane symbol is m_A; p_A is reserved for an M/M twin plane."
            ),
            physical_meaning=(
                "Each reported m_A is an exact parent-crystal reciprocal covector defining "
                "a single-variant invariant-plane interface. Plane covectors are projective: "
                "m_A and -m_A describe the same physical plane."
            ),
            limitation=(
                "Diagnostic planes from a nonzero normalized residual remain approximate "
                "and cannot be promoted to exact A/M branches."
            ),
            evidence=habit_evidence,
        )
    )

    # 7 ------------------------------------------------------------------
    exact_shear_rows = tuple(
        row
        for row in exact_habit_rows
        if _row_attr(row, "shape_vector_parent_crystal") is not None
    )
    if exact_planes and exact_shear_rows:
        shear_answer = (
            f"Exact CT A/M displacement/shear data are exposed for "
            f"{len(exact_shear_rows)} exact habit branch(es)."
        )
        shear_status = "reached"
    elif exact_planes:
        shear_answer = (
            "An exact habit plane exists, but the current unified inventory has not "
            "exposed its native d_A row; no d-vector is fabricated."
        )
        shear_status = "not evaluated"
    else:
        shear_answer = (
            "Without an exact A/M habit-plane seed, an exact CT d_A vector is not claimed."
        )
        shear_status = "not reachable exactly"

    steps.append(
        AssessmentStep(
            step_id="smc",
            question=(
                "7. What A/M IPS displacement/shear vector follows from each "
                "exact habit plane?"
            ),
            answer=shear_answer,
            status=shear_status,
            formulae=(
                r"m_M=C^{-T}m_A",
                r"SMC=M_A^{-1}-C^{-1}M_M^{-1}C^{-T}",
                r"d_A=SMC\,m_A",
                r"\|m_A\|_\ast^2=m_A^TM_A^{-1}m_A=1",
            ),
            reasoning=(
                "The SMC construction uses the parent and product metrics plus the same "
                "correspondence C. For a physically interpreted magnitude, the habit-plane "
                "covector must use the reciprocal-metric unit normalization. The resulting "
                "d_A is Cayron's A/M IPS displacement/shear vector and is kept distinct "
                "from a Ball–James shape vector."
            ),
            physical_meaning=(
                "Together, m_A and d_A define the CT single-variant invariant-plane "
                "shear/displacement construction used downstream in the A/M/M test."
            ),
            limitation=(
                "A d-vector generated from an approximate diagnostic plane is still "
                "approximate and cannot seed exact supercompatibility."
            ),
            evidence=(
                (
                    "SMC matrix available",
                    "yes" if metric.get("smc_dimensional") is not None else "no",
                ),
                ("Exact CT A/M rows with d_A", len(exact_shear_rows)),
                (
                    "Approximate CT A/M diagnostic rows",
                    len(diagnostic_habit_rows),
                ),
            ),
        )
    )

    # 8 ------------------------------------------------------------------
    if unified is None:
        if closing_gap_requested:
            or_answer = (
                "Closing-gap ORs are requested, but the unified theory run has not yet "
                "been calculated for this request."
            )
        else:
            or_answer = (
                "Closing-gap ORs are not requested and no unified theory inventory is active."
            )
        or_status = "not evaluated" if closing_gap_requested else "not requested"
    elif not closing_gap_requested:
        or_answer = "Closing-gap ORs were not requested in this CT run."
        or_status = "not requested"
    elif closing_rows:
        or_answer = (
            f"CT generated {len(closing_rows)} closing-gap orientation candidate "
            "row(s) from twin-element parallelism constraints."
        )
        or_status = "reached"
    else:
        or_answer = (
            "Closing-gap ORs were requested and evaluated, but no candidate was "
            "produced for the current state."
        )
        or_status = "not reached"

    steps.append(
        AssessmentStep(
            step_id="closing_gap",
            question=(
                "8. Which parent/product orientation candidates are generated "
                "by CT's closing-gap twin parallelisms?"
            ),
            answer=or_answer,
            status=or_status,
            formulae=(
                r"\mathrm{Type~I}:~~p_A\parallel p_M,\qquad a_A\parallel a_M",
                r"\mathrm{Type~II}:~~a_A\parallel a_M,\qquad jp_A\parallel jp_M",
                r"a_A=C^{-1}a_M\quad\mathrm{(Type~I~direction~pullback)}",
            ),
            reasoning=(
                "A closing-gap OR is obtained by enforcing the corresponding twin-plane "
                "and twin-direction parallelisms in physical space. Cayron's T is a passive "
                "crystallographic coordinate-transformation representation of the OR. "
                "A proper Cartesian rotation R is another representation of the same "
                "physical relative orientation only after the crystal bases/conventions "
                "are accounted for. T and R_F from a polar decomposition are therefore "
                "not silently identified."
            ),
            physical_meaning=(
                "These candidates are the relative orientations compatible with the "
                "chosen CT twin construction; they are not automatically a unique "
                "experimental 'natural OR'."
            ),
            limitation=(
                "Multiple closing-gap candidates may exist. A preferred/natural OR "
                "requires an explicit external hypothesis or experimental comparison."
            ),
            evidence=(
                ("Closing-gap requested", "yes" if closing_gap_requested else "no"),
                (
                    "Unified CT inventory available",
                    "yes" if unified is not None else "no",
                ),
                ("CT closing-gap candidate rows", len(closing_rows)),
                (
                    "Example native labels",
                    tuple(
                        str(_row_attr(row, "branch_label", ""))
                        for row in closing_rows[:4]
                    ),
                ),
            ),
        )
    )

    # 9 ------------------------------------------------------------------
    exact_super_seed = bool(exact_am and exact_planes)
    residuals = tuple(
        value
        for value in (
            _row_residual(row, "ct_supercompatibility_dimensionless")
            for row in super_rows
        )
        if value is not None
    )
    satisfied = sum(value <= tolerance for value in residuals)

    if not exact_super_seed:
        if exact_am and degeneracy_order == 3:
            super_answer = (
                "The state is third-order CMC compatible, but a unique exact A/M "
                "habit-plane seed is not defined, so a branchwise A/M/M shear–shear "
                "test is not posed."
            )
        else:
            super_answer = (
                "The exact A/M habit/shear prerequisite is absent, so exact CT "
                "A/M/M supercompatibility is not evaluable."
            )
        super_status = "not evaluable"
    elif unified is None:
        super_answer = (
            "An exact A/M seed exists, but the unified theory inventory has not "
            "yet been calculated for the current request."
        )
        super_status = "not evaluated"
    elif not supercompatibility_requested:
        super_answer = (
            "An exact A/M seed exists, but CT supercompatibility was not requested."
        )
        super_status = "not requested"
    elif not super_rows:
        super_answer = (
            "CT supercompatibility was requested, but no eligible A/M/M branch row "
            "was produced."
        )
        super_status = "not reached"
    elif residuals and satisfied:
        super_answer = (
            f"{satisfied}/{len(residuals)} evaluated A/M/M residual(s) satisfy "
            f"the exact algebraic tolerance ({tolerance:.3e})."
        )
        super_status = "reached"
    elif residuals:
        best = min(residuals)
        super_answer = (
            f"0/{len(residuals)} evaluated A/M/M residual(s) satisfy the exact "
            f"algebraic tolerance. Best ε = {best:.10g}; tolerance = {tolerance:.3e}."
        )
        super_status = "not reached"
    else:
        super_answer = (
            "A/M/M rows exist, but no native dimensionless shear–shear residual "
            "is exposed; the assessment will not infer pass/fail from row.exact."
        )
        super_status = "not available"

    steps.append(
        AssessmentStep(
            step_id="supercompatibility",
            question=(
                "9. Do any exact A/M habit/shear branches and M/M twin branches "
                "satisfy Cayron's shear–shear supercompatibility condition?"
            ),
            answer=super_answer,
            status=super_status,
            formulae=(
                r"2(m_A^Tn)d_A=a",
                r"\varepsilon="
                r"\frac{\left\|2(m_A^Tn)d_A-a\right\|}{s}",
                r"\varepsilon=0\quad\Longleftrightarrow\quad"
                r"\mathrm{exact~shear/shear~compatibility}",
                r"\|m_A\|_\ast=1,\qquad\|n\|_{M_A}=1,\qquad\|a\|_{M_A}=s",
            ),
            reasoning=(
                "The exact CT A/M habit plane and its d_A vector are combined with an "
                "eligible CT M/M twin. The final condition is tested through the native "
                "dimensionless shear–shear incompatibility ε. Crucially, the unified "
                "row.exact flag is not used as the supercompatibility verdict; only the "
                "actual residual is compared with the project's algebraic tolerance. "
                "The metric normalization of m_A, n and a is part of the definition."
            ),
            physical_meaning=(
                "ε=0 means the exact single-variant A/M IPS shear and the M/M twin shear "
                "are mutually compatible in Cayron's A/M/M construction. A nonzero ε "
                "quantifies incompatibility for that branch."
            ),
            limitation=(
                "Exact CT shear–shear supercompatibility is distinct from merely having "
                "an exact A/M habit plane, from merely having an M/M twin, and from the "
                "Ball–James/cofactor criteria."
            ),
            evidence=(
                ("Exact CT A/M habit seed", "yes" if exact_super_seed else "no"),
                (
                    "Supercompatibility requested",
                    "yes" if supercompatibility_requested else "no",
                ),
                (
                    "Unified CT inventory available",
                    "yes" if unified is not None else "no",
                ),
                ("CT supercompatibility rows", len(super_rows)),
            ),
        )
    )

    # 10 -----------------------------------------------------------------
    if not result:
        overall_status = "CT martensitic construction not established"
        overall_answer = "No active calculated CT state is available."
        overall_reasoning = (
            "Without a calculated state, no CT crystallographic conclusion is made."
        )
    elif unified is None:
        overall_status = "incomplete CT inventory"
        overall_answer = (
            "The base CT state and its correspondence topology are available, but the "
            "unified CT branch inventory has not yet been calculated for the current "
            "theory-comparison request."
        )
        overall_reasoning = (
            "The base state can answer metric/CMS compatibility questions, but M/M "
            "twins and optional closing-gap/supercompatibility branches must not be "
            "described as evaluated until a unified theory run actually exists."
        )
    elif topology_ok and mm_rows:
        overall_status = "CT-consistent martensitic crystallography"
        compatibility_clause = (
            "Exact single-variant A/M compatibility is also reached."
            if exact_am
            else "Exact single-variant A/M compatibility is not reached for this metric state."
        )
        overall_answer = (
            "At the CT crystallographic-theory level, the state has a discrete "
            "correspondence-variant topology and CT-native M/M twin relations. "
            + compatibility_clause
        )
        overall_reasoning = (
            "The positive statement is limited to the structures actually produced by CT. "
            "Exact A/M interface compatibility, closing-gap orientation candidates and "
            "A/M/M supercompatibility are separate downstream questions and are reported "
            "with their own statuses."
        )
    elif topology_ok:
        overall_status = "partial CT martensitic construction"
        overall_answer = (
            "The correspondence produces a discrete CT variant/operator topology, but "
            "the evaluated unified CT inventory contains no native M/M twin row."
        )
        overall_reasoning = (
            "A correspondence topology and an M/M twin construction are different claims; "
            "the latter is not inferred merely from the former."
        )
    else:
        overall_status = "CT martensitic construction not established"
        overall_answer = (
            "The active result does not establish the discrete CT correspondence "
            "topology required for this martensitic crystallographic construction."
        )
        overall_reasoning = (
            "The assessment does not infer martensite from an alloy name or from a "
            "single compatibility residual."
        )

    overall_reasoning += (
        " CT alone is not experimental proof of transformation mechanism, phase identity, "
        "reversibility, hysteresis, fatigue performance or shape-memory functionality."
    )

    steps.append(
        AssessmentStep(
            step_id="overall",
            question=(
                "10. What can CT conclude about this crystallographic state, "
                "and what still requires experiment?"
            ),
            answer=overall_answer,
            status=overall_status,
            formulae=(),
            reasoning=overall_reasoning,
            physical_meaning=(
                "The final statement keeps four levels separate: correspondence topology, "
                "M/M twinning, exact A/M interface compatibility, and full A/M/M "
                "supercompatibility. Experimental martensite/SMA functionality is a fifth, "
                "independent evidential level."
            ),
            limitation=(
                "This is a theory-level crystallographic assessment, not an experimental "
                "phase-identification or functional-material verdict."
            ),
            evidence=(
                (
                    "Variant/operator topology",
                    "reached" if topology_ok else "not reached",
                ),
                (
                    "CT M/M construction",
                    (
                        "reached"
                        if mm_rows
                        else ("not evaluated" if unified is None else "not reached")
                    ),
                ),
                (
                    "Exact CT A/M compatibility",
                    "reached" if exact_am else "not reached",
                ),
                ("Exact CT A/M habit plane", hp_status),
                ("CT closing-gap ORs", or_status),
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
