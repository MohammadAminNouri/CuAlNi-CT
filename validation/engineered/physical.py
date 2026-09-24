from __future__ import annotations

"""Physical-first case construction.

This module does not import production crystallography.  It builds a physical
deformation F in Cartesian space and *then* encodes F into crystal bases using
F B_A = B_M C.
"""

import numpy as np

from .model import PhysicalTruth, CrystalEncoding, TruthClass
from .rng import rng


def random_rotation(g: np.random.Generator) -> np.ndarray:
    A = g.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q @ np.diag(d)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def basis_with_condition(g: np.random.Generator, condition: float = 10.0) -> np.ndarray:
    U = random_rotation(g)
    V = random_rotation(g)
    s = np.geomspace(1.0, max(1.0, float(condition)), 3)
    B = U @ np.diag(s) @ V.T
    if np.linalg.det(B) < 0:
        B[:, 0] *= -1.0
    return B


def random_sl3(g: np.random.Generator, steps: int = 7) -> np.ndarray:
    P = np.eye(3, dtype=np.int64)
    for _ in range(steps):
        i, j = g.choice(3, size=2, replace=False)
        k = int(g.choice([-2, -1, 1, 2]))
        E = np.eye(3, dtype=np.int64)
        E[i, j] = k
        candidate = P @ E
        if np.max(np.abs(candidate)) <= 40:
            P = candidate
    if round(float(np.linalg.det(P))) != 1:
        raise AssertionError("SL3 generator failed")
    return P


def random_positive_integer_correspondence(g: np.random.Generator) -> np.ndarray:
    P = random_sl3(g)
    D = np.diag([int(g.choice([1, 1, 1, 2, 3])), 1, 1])
    C = P @ D
    if round(float(np.linalg.det(C))) <= 0:
        raise AssertionError("orientation reversing generated C")
    return C.astype(float)


def classify_lambdas(lam: np.ndarray) -> tuple[TruthClass, bool, int]:
    lam = np.sort(np.asarray(lam, float))
    close = np.isclose(lam, 1.0, atol=0.0, rtol=0.0)
    n1 = int(np.sum(close))
    if n1 == 3:
        return TruthClass.THIRD_ORDER_IDENTITY, True, 3
    if n1 == 2:
        return TruthClass.SECOND_ORDER_DEGENERATE, True, 2
    if n1 == 1:
        idx = int(np.where(close)[0][0])
        others = [lam[j] for j in range(3) if j != idx]
        if (others[0] - 1.0) * (others[1] - 1.0) < 0.0:
            return TruthClass.FIRST_ORDER_COMPATIBLE, True, 1
        return TruthClass.ZERO_EIGENVALUE_SAME_SIGN, False, 0
    return TruthClass.INCOMPATIBLE, False, 0


def make_truth(
    namespace: str,
    index: int,
    *,
    lambdas: np.ndarray | None = None,
    exact_first_order: bool | None = None,
) -> PhysicalTruth:
    g = rng(namespace, index)
    if lambdas is None:
        if exact_first_order is True:
            lambdas = np.array([
                g.uniform(0.55, 0.93),
                1.0,
                g.uniform(1.07, 1.65),
            ])
        elif exact_first_order is False:
            middle = 1.0 + g.choice([-1.0, 1.0]) * g.uniform(2e-3, 0.15)
            lo = min(g.uniform(0.55, 0.9), middle - 0.03)
            hi = max(g.uniform(1.1, 1.65), middle + 0.03)
            lambdas = np.sort(np.array([lo, middle, hi]))
        else:
            raise ValueError("specify lambdas or exact_first_order")

    lambdas = np.asarray(lambdas, float)
    Q = random_rotation(g)
    R = random_rotation(g)
    U = Q @ np.diag(lambdas) @ Q.T
    F = R @ U
    cls, compat, order = classify_lambdas(lambdas)
    return PhysicalTruth(
        case_id=f"{namespace}/{index}",
        lambdas=np.sort(lambdas),
        U_physical=U,
        R_physical=R,
        F_physical=F,
        truth_class=cls,
        exact_compatible=compat,
        expected_degeneracy_order=order,
    )


def encode_truth(
    truth: PhysicalTruth,
    namespace: str,
    index: int,
    *,
    parent_condition: float = 8.0,
    C: np.ndarray | None = None,
    B_parent: np.ndarray | None = None,
    provenance: tuple[str, ...] = (),
) -> CrystalEncoding:
    g = rng(namespace, index)
    Ba = (
        basis_with_condition(g, parent_condition)
        if B_parent is None else np.asarray(B_parent, float)
    )
    Cmat = (
        random_positive_integer_correspondence(g)
        if C is None else np.asarray(C, float)
    )
    Bm = truth.F_physical @ Ba @ np.linalg.inv(Cmat)
    if np.linalg.det(Ba) <= 0 or np.linalg.det(Bm) <= 0 or np.linalg.det(Cmat) <= 0:
        raise AssertionError("physical encoding lost handedness")
    Ma = Ba.T @ Ba
    Mm = Bm.T @ Bm
    return CrystalEncoding(
        case_id=truth.case_id,
        B_parent=Ba,
        B_product=Bm,
        C_m_from_a=Cmat,
        M_parent=0.5 * (Ma + Ma.T),
        M_product=0.5 * (Mm + Mm.T),
        truth=truth,
        provenance=provenance + ("physical-first:F*B_A=B_M*C",),
    )


def first_order_case(namespace: str, index: int, *, parent_condition: float = 8.0) -> CrystalEncoding:
    truth = make_truth(namespace, index, exact_first_order=True)
    return encode_truth(truth, namespace + "-encoding", index, parent_condition=parent_condition)


def incompatible_case(namespace: str, index: int, *, parent_condition: float = 8.0) -> CrystalEncoding:
    truth = make_truth(namespace, index, exact_first_order=False)
    return encode_truth(truth, namespace + "-encoding", index, parent_condition=parent_condition)


def degeneracy_case(kind: str, index: int = 0) -> CrystalEncoding:
    if kind == "second_low":
        lam = np.array([0.73, 1.0, 1.0])
    elif kind == "second_high":
        lam = np.array([1.0, 1.0, 1.31])
    elif kind == "identity":
        lam = np.ones(3)
    elif kind == "same_sign_low":
        lam = np.array([0.72, 0.88, 1.0])
    elif kind == "same_sign_high":
        lam = np.array([1.0, 1.12, 1.34])
    else:
        raise ValueError(kind)
    truth = make_truth(f"degeneracy-{kind}", index, lambdas=lam)
    return encode_truth(truth, f"degeneracy-{kind}-encoding", index)


def reencode(
    case: CrystalEncoding,
    *,
    B_parent: np.ndarray | None = None,
    C: np.ndarray | None = None,
    tag: str,
) -> CrystalEncoding:
    Ba = case.B_parent if B_parent is None else np.asarray(B_parent, float)
    Cmat = case.C_m_from_a if C is None else np.asarray(C, float)
    Bm = case.truth.F_physical @ Ba @ np.linalg.inv(Cmat)
    return CrystalEncoding(
        case_id=case.case_id,
        B_parent=Ba,
        B_product=Bm,
        C_m_from_a=Cmat,
        M_parent=Ba.T @ Ba,
        M_product=Bm.T @ Bm,
        truth=case.truth,
        provenance=case.provenance + (tag,),
    )


def crystal_rebasis(case: CrystalEncoding, namespace: str, index: int) -> CrystalEncoding:
    g = rng(namespace, index)
    Pa = random_sl3(g).astype(float)
    Pm = random_sl3(g).astype(float)
    Ba = case.B_parent @ Pa
    Bm = case.B_product @ Pm
    Cnew = np.linalg.inv(Pm) @ case.C_m_from_a @ Pa
    rounded = np.rint(Cnew)
    if np.linalg.norm(Cnew - rounded) < 1e-10:
        Cnew = rounded
    return CrystalEncoding(
        case_id=case.case_id,
        B_parent=Ba,
        B_product=Bm,
        C_m_from_a=Cnew,
        M_parent=Ba.T @ Ba,
        M_product=Bm.T @ Bm,
        truth=case.truth,
        provenance=case.provenance + ("SL3 parent/product rebasis",),
    )


def common_length_scale(case: CrystalEncoding, scale: float) -> CrystalEncoding:
    return CrystalEncoding(
        case_id=case.case_id,
        B_parent=scale * case.B_parent,
        B_product=scale * case.B_product,
        C_m_from_a=case.C_m_from_a.copy(),
        M_parent=(scale * scale) * case.M_parent,
        M_product=(scale * scale) * case.M_product,
        truth=case.truth,
        provenance=case.provenance + (f"common length scale={scale:g}",),
    )


def reciprocal_swap(case: CrystalEncoding) -> CrystalEncoding:
    """Swap parent/product. Physical F becomes F^-1."""
    F = np.linalg.inv(case.truth.F_physical)
    U = np.linalg.eigh(F.T @ F)
    w, V = U
    Uinv = (V * np.sqrt(w)) @ V.T
    R = F @ np.linalg.inv(Uinv)
    lam = np.sort(np.linalg.svd(F, compute_uv=False))
    cls, compat, order = classify_lambdas(lam)
    truth = PhysicalTruth(
        case_id=case.case_id + "/swapped",
        lambdas=lam,
        U_physical=Uinv,
        R_physical=R,
        F_physical=F,
        truth_class=cls,
        exact_compatible=compat,
        expected_degeneracy_order=order,
        notes="parent/product reciprocal transformation",
    )
    C = np.linalg.inv(case.C_m_from_a)
    return CrystalEncoding(
        case_id=truth.case_id,
        B_parent=case.B_product,
        B_product=case.B_parent,
        C_m_from_a=C,
        M_parent=case.M_product,
        M_product=case.M_parent,
        truth=truth,
        provenance=case.provenance + ("parent/product swap",),
    )


def metric_to_cell(M: np.ndarray) -> dict[str, float]:
    M = np.asarray(M, float)
    a, b, c = np.sqrt(np.diag(M))
    ca = np.clip(M[1, 2] / (b*c), -1.0, 1.0)
    cb = np.clip(M[0, 2] / (a*c), -1.0, 1.0)
    cg = np.clip(M[0, 1] / (a*b), -1.0, 1.0)
    return {
        "a": float(a), "b": float(b), "c": float(c),
        "alpha_deg": float(np.degrees(np.arccos(ca))),
        "beta_deg": float(np.degrees(np.arccos(cb))),
        "gamma_deg": float(np.degrees(np.arccos(cg))),
    }
