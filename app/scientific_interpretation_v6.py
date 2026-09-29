from __future__ import annotations

"""Reconstructive professor-facing interpretation layered on the frozen solvers."""

from dataclasses import replace
from typing import Any, Mapping

import app.scientific_interpretation as base
from app.scientific_types import Evidence, Finding, sci


def _with(f: Finding, **kwargs: Any) -> Finding:
    return replace(f, **kwargs)


def transformation_findings(result: Mapping[str, Any]) -> tuple[Finding, ...]:
    ct, bj, agreement = base.transformation_findings(result)
    summary = result["summary"]
    ctd = result["ct_detail"]
    ct_ok = bool(summary["ct_exact_compatible"])
    bj_ok = bool(result["ball_james_detail"]["lambda2_exact"])
    ct = _with(
        ct,
        theory="Cayron Correspondence Theory — CMC degeneracy",
        equations=(
            r"\mathrm{CMC}=C^{T}M_M C-M_A",
            r"\widehat G=M_A^{-1/2}C^TM_MC\,M_A^{-1/2},\qquad \widehat{\mathrm{CMC}}=\widehat G-I",
            r"(C^TM_MC)v_i=\mu_i M_Av_i,\qquad \eta_i=\mu_i-1",
        ),
        symbols=(
            ("M_A, M_M", "parent and product lattice metric tensors"),
            ("C", "lattice correspondence C_{M←A}; not an orientation rotation"),
            ("μ_i, η_i", "generalized metric eigenvalues and their offsets from unity"),
        ),
        assumptions=("M_A and M_M are symmetric positive-definite metrics.", "The registered correspondence and crystal settings are used without inference."),
        backend_mapping=(
            "cualni_cryst.ct.cmc evaluates CᵀM_MC−M_A.",
            "analyze_cmc solves the symmetric-definite generalized eigenproblem and classifies the exact zero structure without enlarging tolerance.",
            "Nearest-degeneracy planes are generated separately and never promoted to exact habits.",
        ),
        provenance=("Cayron Correspondence Theory; dimensional CMC is implemented as Cayron 2026 Eq. 32.",),
        verbal="CT asks whether the correspondence pulls the product metric onto the parent metric with the exact degeneracy required for a compatible interface.",
        verification=(
            f"exact A/M status = {'satisfied' if ct_ok else 'not satisfied'}",
            f"nearest zero residual = {sci(ctd['nearest_zero_residual'], 6)}",
            "approximate diagnostic substituted for exact branch = no",
        ),
    )
    bj = _with(
        bj,
        theory="Ball–James nonlinear-elastic single-variant compatibility",
        equations=(
            r"U=(F^TF)^{1/2},\qquad \lambda_1\leq\lambda_2\leq\lambda_3",
            r"\lambda_2=1",
            r"RU-I=b\otimes m",
        ),
        symbols=(("U", "right stretch"), ("R∈SO(3)", "rigid rotation"), ("b", "rank-one shape vector"), ("m", "interface normal")),
        assumptions=("The test is for one homogeneous martensite stretch variant against austenite.",),
        backend_mapping=("The metric core computes U and its ordered principal stretches.", "The Ball–James backend constructs rank-one branches only when the λ₂ criterion is met."),
        provenance=("Ball–James nonlinear-elastic compatibility / crystallographic rank-one connection.",),
        verbal="A single variant can meet austenite exactly only if its middle principal stretch is one, so a rotation can make RU−I rank one.",
        verification=(f"exact A/M status = {'satisfied' if bj_ok else 'not satisfied'}",),
    )
    agreement = _with(
        agreement,
        rationale=f"CT exact A/M: {'satisfied' if ct_ok else 'not satisfied'}; Ball–James exact A/M: {'satisfied' if bj_ok else 'not satisfied'}. The native residuals are not treated as the same scalar.",
        theory="Independent existence comparison at the physical-question boundary",
        backend_mapping=("CT CMC degeneracy and Ball–James λ₂/rank-one compatibility are solved independently; only the final existence classification is compared.",),
        verbal="The two theories answer the same yes/no interface-existence question through different native mathematics.",
        verification=(f"CT status = {'satisfied' if ct_ok else 'not satisfied'}", f"Ball–James status = {'satisfied' if bj_ok else 'not satisfied'}"),
    )
    return ct, bj, agreement


def topology_finding(result: Mapping[str, Any]) -> Finding:
    f = base.topology_finding(result)
    return _with(
        f,
        theory="Correspondence groupoid / coset and double-coset topology",
        equations=(r"N_{\mathrm{variants}}=[G_A:H]", r"\mathrm{operators}\sim H\backslash G_A/H"),
        symbols=(("G_A", "registered parent point group"), ("H", "common stabilizer subgroup induced by the correspondence")),
        backend_mapping=("Exact rationalized symmetry operators are used for the groupoid calculation.", "Variant and operator counts are deliberately distinct topological objects."),
        verbal="Variants count symmetry-distinct transformation states; operator classes count symmetry-distinct relations between those states.",
        verification=("stretch-variant count and operator-class count are reported separately",),
    )


def orientation_finding(analysis: Mapping[str, Any]) -> Finding:
    f = base.orientation_finding(analysis)
    note = str(analysis.get("origin_note", ""))
    return _with(
        f,
        theory="Physical Cartesian orientation relationship with explicit lattice correspondence kept separate",
        equations=(r"x_A=R_{A\leftarrow M}x_M", r"R_{M\leftarrow A}=R_{A\leftarrow M}^{T}", r"u_M=C_{M\leftarrow A}u_A,\qquad p_M=C_{M\leftarrow A}^{-T}p_A,\qquad C\neq R"),
        symbols=(("R", "proper physical Cartesian rotation"), ("C", "lattice correspondence acting on lattice coefficients")),
        assumptions=("Mapping direction is product→parent for R_{A←M}.", "Any near-rotation repair is an explicit projection to SO(3), never silent."),
        backend_mapping=("The orientation service audits det(R)=+1 and orthogonality.", "Correspondence and orientation objects use separate APIs and are never substituted."),
        provenance=("Polar decomposition, manual matrix, Euler, parallelism, CT closing-gap, PTMC or experiment provenance remains attached to the source.",),
        verbal="R rotates physical Cartesian vectors; C maps lattice coordinates. They can be related in a model but they are not the same object.",
        verification=(f"OR provenance = {note or 'reported by the active source'}", "zero-angle rotation axis = conventional/undefined, never a physically preferred axis"),
    )


def martensite_pair_finding(pair: Mapping[str, Any], *, unique_systems: int) -> Finding:
    f = base.martensite_pair_finding(pair, unique_systems=unique_systems)
    return _with(
        f,
        theory="Mallard / Ball–James martensite–martensite rank-one compatibility with local PTMC follow-up",
        equations=(r"Q\,U_j-U_i=a\otimes n", r"R\,F(f)=I+b\otimes m"),
        symbols=(("U_i,U_j", "selected martensite stretch variants"), ("Q∈SO(3)", "relative twin rotation"), ("a,n", "twin shear/shape vector and interface normal"), ("f", "selected laminate parameter")),
        assumptions=("All counts in this finding are local to the selected variant pair/twin system.",),
        backend_mapping=("Mallard generator routes are grouped by reconstructed rank-one dyad before reporting unique physical systems.", "PTMC roots shown here are local branches, not the global deep-run count."),
        verbal="First the selected variants must form a rank-one twin; PTMC then asks whether a laminate of that twin can form a macroscopic invariant plane.",
        verification=(
            f"unique physical systems for selected pair = {unique_systems}",
            "scope = selected variant pair/twin system only",
            "PTMC branch count in this finding = local selected-system branches, never the global deep-run count",
        ),
    )


def ebsd_audit_finding(audit: Mapping[str, Any]) -> Finding:
    f = base.ebsd_audit_finding(audit)
    return _with(
        f,
        theory="Convention-explicit EBSD orientation/disorientation audit",
        equations=(r"v_{sample}=g_{sample\leftarrow crystal}v_{crystal}", r"\Delta g=g_2S_2(g_1S_1)^T,\qquad \theta=\cos^{-1}((\operatorname{tr}\Delta g-1)/2)"),
        symbols=(("g", "crystal-to-sample orientation matrix under the declared convention"), ("S_i", "proper phase-symmetry operators used in disorientation minimization")),
        assumptions=("Euler convention, angle units, specimen frame, phase mapping and filtering must be explicit before reconstruction.", "Trace validation is disabled unless an explicit sample-surface normal is supplied."),
        backend_mapping=("Imported rotations are checked for SO(3) consistency before segmentation or reconstruction.", "Phase masks, quality thresholds, ambiguity margins, reconstruction residuals and bounded OR-refinement settings remain explicit inputs/outputs."),
        provenance=("EBSD orientation convention: crystal Cartesian → sample Cartesian; vendor conventions are converted only at the I/O boundary.",),
        verbal="The EBSD layer first proves what coordinate convention and phase each orientation belongs to, then compares symmetry-reduced measured rotations to predictions without silently correcting the specimen frame.",
        verification=("no Euler convention, unit or specimen-frame correction is guessed by this finding", "minority/raw phase labels are retained before user-selected filtering"),
    )


def am_existence_finding(checks: Mapping[str, Any], *, exact_ct_habits: int, bj_am: int, approx_ct_habits: int) -> Finding:
    f = base.am_existence_finding(checks, exact_ct_habits=exact_ct_habits, bj_am=bj_am, approx_ct_habits=approx_ct_habits)
    return _with(
        f,
        theory="Independent Cayron-CT CMC and Ball–James A/M existence tests",
        equations=(r"\mathrm{CMC}=C^TM_MC-M_A", r"\lambda_2(U)=1", r"RU-I=b\otimes m"),
        backend_mapping=("Exact CT habits and exact Ball–James branches are counted independently.", "Nearest-degeneracy CT diagnostic planes are excluded from exact existence claims."),
        verbal="The same physical existence question is answered independently by CT metric degeneracy and Ball–James rank-one compatibility.",
        verification=(f"exact CT habits = {exact_ct_habits}", f"approximate CT diagnostic habits = {approx_ct_habits}", f"exact Ball–James branches = {bj_am}"),
    )


def mm_audit_finding(audit: Mapping[str, Any] | None) -> Finding | None:
    f = base.mm_audit_finding(audit)
    if f is None:
        return None
    relation_count = audit.get("relation_count") if isinstance(audit, Mapping) else None
    success = audit.get("success") if isinstance(audit, Mapping) else None
    verification = [
        "generic assignment ≠ physical agreement",
        "partial generic matches are never counted as theory failures",
        "authoritative comparison = independent CT ↔ Mallard ↔ Ball–James push-forward audit",
    ]
    if relation_count is not None:
        verification.append(f"authoritative physical relation count = {relation_count}")
    if success is not None:
        verification.append(f"authoritative audit success = {'yes' if bool(success) else 'no'}")
    return _with(
        f,
        theory="Independent CT ↔ Mallard ↔ Ball–James M/M physical-relation audit",
        equations=(r"Q\,U_j-U_i=a\otimes n", r"d_{current}\propto U_j d_{reference}"),
        assumptions=(
            "CT source-native parent-reference twin directions are not directly equated to current-configuration Ball–James/PTMC shear directions.",
            "Multiple Mallard/generator routes may reconstruct the same physical rank-one relation and are grouped before physical counting.",
        ),
        backend_mapping=(
            "The authoritative audit performs the required branch-specific U_j push-forward before comparing geometry, shear, rank-one reconstruction and rotation.",
            "Raw generator provenance remains an audit layer; the physical-relation count is deduplicated independently.",
        ),
        verbal="More candidate assignments than physical relations is bookkeeping, not a contradiction; the authoritative audit first puts native objects into one physical configuration and then deduplicates the rank-one relations.",
        verification=tuple(verification),
    )


def ptmc_finding(unified: Any, *, enabled: bool) -> Finding:
    f = base.ptmc_finding(unified, enabled=enabled)
    if not enabled:
        return _with(f, theory="WLR/BM-style phenomenological theory of martensite crystallography (PTMC)", verification=("requested = no", "calculation executed = no", "status = not requested"))
    count = len([r for r in getattr(unified, "rows", ()) if str(getattr(getattr(r, "theory", None), "value", "")) == "ptmc" and str(getattr(getattr(r, "prediction_kind", None), "value", "")) == "ptmc_habit" and getattr(r, "exact", None) is True])
    conclusion = f"PTMC found {count} exact native solver branch(es) for the requested twinning-LIS deep run." if count else f.conclusion
    return _with(
        f,
        conclusion=conclusion,
        theory="Phenomenological Theory of Martensite Crystallography — twinned laminate",
        equations=(r"F(f)=\text{macroscopic laminate deformation}", r"R\,F(f)=I+b\otimes m"),
        symbols=(("f", "native laminate/twin parameter"), ("R", "habit rotation"), ("b,m", "PTMC-native shape vector and habit-plane normal")),
        assumptions=("A native solver branch is not automatically a unique physical habit system; physical deduplication requires explicit symmetry/projective/dyad rules.",),
        backend_mapping=("The enabled twinning-LIS branch set is solved for an invariant-plane/rank-one macroscopic deformation.",),
        verbal="PTMC can recover exact macroscopic compatibility by twinning even when one homogeneous martensite variant fails CT/Ball–James A/M compatibility.",
        verification=(f"exact native PTMC solver branches = {count}", "physical unique-system count = not inferred from this row count"),
    )


def or_comparison_finding(equivalence: Any, *, enabled: bool, tolerance_deg: float) -> Finding:
    f = base.or_comparison_finding(equivalence, enabled=enabled, tolerance_deg=tolerance_deg)
    if not enabled:
        return _with(f, verification=("requested = no", "calculation executed = no", "status = not requested"))
    return _with(
        f,
        theory="Symmetry-reduced physical OR disorientation comparison",
        equations=(r"\Delta_{\mathrm{OR}}=\min_{S_A,S_M\in G^+}\operatorname{angle}\!\left(S_A R_1 S_M R_2^T\right)",),
        symbols=(("G⁺", "proper rotational subgroup of the registered crystal point group"), ("R_1,R_2", "physical OR matrices expressed in the same Cartesian convention")),
        assumptions=("The reported minimum is over eligible assigned branch pairs unless an explicit all-pairs search is run.",),
        backend_mapping=("Both ORs are re-expressed in a common Cartesian frame before proper parent/product symmetry minimization.",),
        verbal="A small disorientation means the physical rotations are geometrically close after symmetry; it does not make their derivations identical.",
        verification=(f"requested OR tolerance = {tolerance_deg:.9g}°", "scope = eligible assigned pairs; not asserted to be the global all-pairs minimum"),
    )


def cofactor_finding(unified: Any) -> Finding:
    f = base.cofactor_finding(unified)
    return _with(
        f,
        theory="Chen–Srivastava–Dabade–James cofactor conditions",
        equations=(
            r"\mathrm{CC1}:\ \lambda_2=1",
            r"\mathrm{CC2}:\ a\cdot U\,\operatorname{cof}(U^2-I)\,n=0",
            r"\mathrm{CC3}:\ \operatorname{tr}(U^2)-\det(U^2)-\frac{|a|^2|n|^2}{4}-2\geq0",
        ),
        symbols=(("U", "martensite stretch of the branch"), ("a,n", "native twin shear vector and normal")),
        backend_mapping=("cualni_cryst.cofactor.evaluate_cofactor_conditions evaluates CC1, CC2 and CC3 separately, including the minus sign before det(U²).",),
        provenance=("Chen, Srivastava, Dabade & James (2013), Theorem 2.",),
        verbal="All three cofactor conditions must hold on the same native twin branch; the three tests are not collapsed into one score.",
        verification=("row totals are native Ball–James branch rows, not automatically independent physical twins",),
    )


def supercompatibility_finding(unified: Any, *, requested: bool, algebraic_tolerance: float) -> Finding:
    f = base.supercompatibility_finding(unified, requested=requested, algebraic_tolerance=algebraic_tolerance)
    exact = [r for r in getattr(unified, "rows", ()) if str(getattr(getattr(r, "theory", None), "value", "")) == "cayron_ct" and str(getattr(getattr(r, "prediction_kind", None), "value", "")) == "ct_am_habit" and getattr(r, "exact", None) is True]
    rows = [r for r in getattr(unified, "rows", ()) if str(getattr(getattr(r, "prediction_kind", None), "value", "")) == "ct_supercompatibility"]
    if not requested:
        verification = ("requested = no", "calculation executed = no", "status = not requested")
    elif not exact:
        verification = ("requested = yes", "exact A/M seed required = yes", "exact A/M seeds = 0", "approximate seed substituted = no", "calculation executed = no", "status = not evaluable")
    else:
        verification = (f"requested = yes", f"exact A/M seeds = {len(exact)}", f"calculation rows = {len(rows)}", "approximate seed substituted = no")
    return _with(
        f,
        theory="Cayron CT A/M/M shear–shear supercompatibility",
        equations=(r"2\,(m_A^T n)\,d_A=a", r"r_{\mathrm{SC}}=2\,(m_A^T n)\,d_A-a"),
        symbols=(("m_A", "exact CT A/M habit-plane covector/normal object in the implemented parent frame"), ("d_A", "CT IPS shear vector from SMC"), ("n", "CT M/M twin-plane unit normal"), ("a", "CT M/M twin shear vector")),
        assumptions=("Only exact CT A/M habits may seed this exact condition.", "Nearest-degeneracy diagnostic planes are never substituted."),
        backend_mapping=("cualni_cryst.ct.ct_supercompatibility_vector evaluates the native residual vector; the reported dimensionless residual is classified against the project algebraic tolerance."),
        verbal="Supercompatibility is only a meaningful exact test after an exact CT A/M habit exists; without that seed the correct status is not evaluable, not failed.",
        verification=verification,
    )


def experiment_finding(equivalence: Any) -> Finding | None:
    f = base.experiment_finding(equivalence)
    if f is None:
        return None
    return _with(
        f,
        theory="Observable-by-observable theory ↔ experiment residual comparison",
        assumptions=("Reference frame, active/passive convention and observable type must match before a residual is interpreted.",),
        backend_mapping=("No aggregate theory-winner score is constructed; each measurable component remains explicit.",),
        verbal="The experiment layer asks which exact predicted branch approaches the measured observable, not which theory has the prettiest internal agreement.",
        verification=("component-wise residuals only; no winner score",),
    )
