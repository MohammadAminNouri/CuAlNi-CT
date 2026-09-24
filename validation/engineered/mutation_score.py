from __future__ import annotations

import numpy as np

from .mutants import *
from .physical import first_order_case, crystal_rebasis
from .rng import rng


def mutation_score() -> dict[str, bool]:
    killed: dict[str, bool] = {}

    case = first_order_case("mutation", 0)
    C, Ma, Mm = case.C_m_from_a, case.M_parent, case.M_product

    G = pullback_correct(C, Mm)
    killed["pullback_order"] = bool(
        np.linalg.norm(pullback_wrong_order(C, Mm) - G)
        / max(np.linalg.norm(G), 1e-15) > 1e-3
    )

    Ac = cmc_correct(Ma,Mm,C)
    killed["cmc_parent_sign"] = bool(
        np.linalg.norm(cmc_wrong_plus_parent(Ma,Mm,C)-Ac)
        / max(np.linalg.norm(Ac),1e-15) > 1e-3
    )

    g = rng("mutation-plane", 0)
    u = g.normal(size=3)
    p = np.cross(u, g.normal(size=3))
    um = C @ u
    pc = plane_map_correct(C, p)
    pw = plane_map_wrong_direct(C, p)
    correct_inc = abs(float(pc @ um))
    wrong_inc = abs(float(pw @ um)) / max(np.linalg.norm(pw)*np.linalg.norm(um), 1e-15)
    killed["plane_direct_instead_inverse_transpose"] = bool(
        correct_inc < 1e-8 and wrong_inc > 1e-4
    )

    Sc = smc_correct(Ma, Mm, C)
    killed["smc_no_inverse_C"] = bool(
        np.linalg.norm(smc_wrong_no_inverse_C(Ma, Mm, C)-Sc)
        / max(np.linalg.norm(Sc), 1e-15) > 1e-3
    )

    rebased = crystal_rebasis(case, "mutation-basis", 1)
    Pa = np.linalg.solve(case.B_parent, rebased.B_parent)
    Pm = np.linalg.solve(case.B_product, rebased.B_product)
    Cc = basis_C_correct(C, Pa, Pm)
    Cw = basis_C_wrong(C, Pa, Pm)
    killed["basis_change_direction"] = bool(
        np.linalg.norm(Cc - rebased.C_m_from_a) < 1e-8
        and np.linalg.norm(Cw - rebased.C_m_from_a) > 1e-3
    )

    U = case.truth.U_physical
    a = np.array([0.7, -0.2, 0.4])
    n = np.array([0.1, 0.9, -0.3])
    killed["cofactor_cc3_det_sign"] = bool(abs(
        cofactor_cc3_wrong_plus_det(U,a,n) - cofactor_cc3_correct(U,a,n)
    ) > 1e-2)

    F = U + 0.37*np.outer(a,n)
    killed["ptmc_wrong_singular_value"] = bool(abs(
        middle_stretch_wrong_smallest(F) - middle_stretch_correct(F)
    ) > 1e-3)

    p2 = np.array([1.0, -2.0, 0.7])
    u2 = np.array([-0.4, 1.1, 2.0])
    killed["plane_metric_not_reciprocal_metric"] = bool(abs(
        plane_norm_squared_correct(p2,Ma)-plane_norm_squared_wrong_direct(p2,Ma)
    ) > 1e-2)
    killed["direction_metric_not_reciprocal_metric"] = bool(abs(
        direction_norm_squared_correct(u2,Ma)-direction_norm_squared_wrong_reciprocal(u2,Ma)
    ) > 1e-2)

    killed["cmc_signature_requires_opposite_signs"] = bool(
        compatible_signature_correct(np.array([-0.3,0.0,-0.1])) is False
        and compatible_signature_wrong_any_zero(np.array([-0.3,0.0,-0.1])) is True
    )

    v = np.array([0.2,-0.5,0.8])
    killed["projective_antipodal_equivalence"] = bool(
        projective_parallel_correct(v,-v)
        and not projective_parallel_wrong_oriented(v,-v)
    )

    J = volume_ratio_correct(Ma,Mm,C)
    Jw = volume_ratio_wrong_without_sqrt(Ma,Mm,C)
    killed["volume_ratio_square_root"] = bool(abs(J-Jw) > 1e-3)

    return {name: bool(value) for name, value in killed.items()}
