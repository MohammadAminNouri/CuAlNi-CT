from __future__ import annotations

"""Human-readable reports designed for discussion with supervisors/professors."""

from pathlib import Path
import numpy as np
import sympy as sp

from .cualni_models import do3_to_6m_branch, do3_to_2h_branch, james_hane_6m_example_lattices
from .group_theory import correspondence_groupoid
from .symmetry import classify_symmetry
from .symbolic import do3_to_6m_pulled_metric, do3_to_2h_pulled_metric
from .stretch import stretch_from_metrics, principal_stretches, generate_stretch_variants, match_matrix_family
from .james_hane import cube_edge_6m_variants
from .symmetry import cubic_proper_rotations
from .ct import analyze_austenite_martensite


def _m(M): return sp.pretty(sp.Matrix(M))


def branch_report(branch) -> str:
    g=correspondence_groupoid(list(branch.parent_point_group),list(branch.product_point_group),branch.correspondence)
    lines=[f"# {branch.name}: exact discrete CT report","",f"Correspondence convention: `u_M = C_m_from_a u_A`.","","## C_m_from_a","```text",_m(branch.correspondence.C_m_from_a),"```","",
           f"Parent group order: {len(branch.parent_point_group)}",f"Product group order: {len(branch.product_point_group)}",f"|H_C| = {len(g.subgroup)}",f"N_C = {g.n_variants}",f"N_operators = {g.n_operators}",f"Double-coset sizes = {[len(x) for x in g.operators]}",f"Burnside cross-check = {g.burnside_count}","","## H_C"]
    for h in g.subgroup:
        info=classify_symmetry(h)
        lines += ["```text",_m(h),"```",f"kind={info.kind}, det={info.determinant}, order={info.order}",""]
    lines += ["## Operator summaries"]
    for s in g.summaries:
        lines.append(f"- O{s.index}: size={s.size}, inverse=O{s.inverse_index}, ambivalent={s.ambivalent}, contents={dict(s.symmetry_kinds)}")
    lines += ["","## Variant -> operator adjacency","```text"]
    lines.extend(" ".join(f"{x:2d}" for x in row) for row in g.adjacency)
    lines += ["```",""]
    return "\n".join(lines)


def verification_report() -> str:
    A,M=james_hane_6m_example_lattices(); branch=do3_to_6m_branch()
    U=stretch_from_metrics(A.metric(),M.metric(),branch.correspondence)
    lam,_=principal_stretches(U)
    derived=generate_stretch_variants(U,[np.array(q,float) for q in cubic_proper_rotations()])
    ref=cube_edge_6m_variants(A.a,M.a,M.b,M.c,M.beta_deg)
    ok,R=match_matrix_family(derived,ref,tol=1e-10)
    ct=analyze_austenite_martensite(A.metric(),M.metric(),branch.correspondence)
    return "\n".join([
        "# Independent verification report: DO3 -> 6M", "",
        "This report uses the James-Hane literature example only as a reproducibility benchmark; it is not a default specimen.","",
        "## Stretch from correspondence + metrics", "```text", np.array2string(U,precision=12), "```",
        f"Principal stretches: {np.array2string(lam,precision=12)}",
        f"|lambda2-1| = {abs(lam[1]-1):.12g}",
        f"Number of symmetry-generated stretch variants = {len(derived)}",
        f"Exact family match to independently transcribed James-Hane Eq.(10) = {ok}",
        f"Worst nearest Frobenius mismatch = {np.max(np.min(R,axis=1)):.3e}","",
        "## CMC",f"normalized CMC eigenvalues = {np.array2string(ct.analysis.eigenvalues,precision=12)}",
        f"exact CT A/M compatibility at rounded literature parameters = {ct.analysis.exact_compatible}",
        f"nearest normalized CMC eigenvalue magnitude = {ct.analysis.nearest_zero_residual:.12g}",
        "",
        "Interpretation: near-zero is expected for the published near-compatible Cu-Al-Ni example; exact equality must not be claimed from rounded table values.",
    ])


def write_all_reports(directory: str|Path) -> list[Path]:
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for name,txt in [
        ("DO3_6M_GROUPoid.md",branch_report(do3_to_6m_branch())),
        ("DO3_2H_GROUPoid.md",branch_report(do3_to_2h_branch())),
        ("DO3_6M_VERIFICATION.md",verification_report()),
    ]:
        p=d/name;p.write_text(txt,encoding='utf-8');outputs.append(p)
    return outputs
