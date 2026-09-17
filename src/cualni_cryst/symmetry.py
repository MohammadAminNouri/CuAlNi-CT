from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
import sympy as sp


def matrix_key(g: sp.Matrix) -> tuple:
    return tuple(sp.simplify(x) for x in list(sp.Matrix(g)))


def canonical_matrix_sort_key(g: sp.Matrix) -> tuple:
    return tuple(str(sp.simplify(x)) for x in list(sp.Matrix(g)))


def cubic_full_m3m() -> list[sp.Matrix]:
    """All 48 signed permutation matrices of cubic m-3m (O_h)."""
    out: dict[tuple, sp.Matrix] = {}
    for perm in permutations(range(3)):
        P = sp.zeros(3)
        for col, row in enumerate(perm):
            P[row, col] = 1
        for signs in product((-1, 1), repeat=3):
            G = P * sp.diag(*signs)
            out[matrix_key(G)] = G
    mats = sorted(out.values(), key=canonical_matrix_sort_key)
    if len(mats) != 48:
        raise AssertionError(f"Expected 48 cubic operations, found {len(mats)}")
    # identity first for human-readable tables
    I = sp.eye(3)
    mats.remove(I)
    return [I] + mats


def cubic_proper_rotations() -> list[sp.Matrix]:
    return [g for g in cubic_full_m3m() if int(g.det()) == 1]


def monoclinic_2_over_m_unique_b() -> list[sp.Matrix]:
    """Point group 2/m in conventional unique-b direct basis."""
    mats = [
        sp.eye(3),
        sp.diag(-1, 1, -1),      # 2-fold around b
        -sp.eye(3),              # inversion
        sp.diag(1, -1, 1),       # mirror normal to b
    ]
    return mats


def orthorhombic_mmm() -> list[sp.Matrix]:
    """Full orthorhombic mmm (D2h), order 8."""
    mats = [sp.diag(sx, sy, sz) for sx, sy, sz in product((-1, 1), repeat=3)]
    mats = sorted(mats, key=canonical_matrix_sort_key)
    I = sp.eye(3)
    mats.remove(I)
    return [I] + mats


def classify_cubic_operation(g: sp.Matrix, atol: float = 1e-9) -> dict[str, object]:
    """Human-readable geometric classification of an orthogonal cubic operation."""
    G = np.array(g, dtype=float)
    det = round(float(np.linalg.det(G)))
    tr = float(np.trace(G))
    info: dict[str, object] = {"det": det, "trace": tr, "order": matrix_order(g)}

    if np.allclose(G, np.eye(3), atol=atol):
        info.update(kind="identity", angle_deg=0.0)
        return info
    if np.allclose(G, -np.eye(3), atol=atol):
        info.update(kind="inversion")
        return info

    if det == 1:
        cosang = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cosang)))
        vals, vecs = np.linalg.eig(G)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        axis = np.real(vecs[:, idx])
        axis /= np.max(np.abs(axis))
        info.update(kind="rotation", angle_deg=angle, axis=_canonical_direction(axis))
        return info

    vals, vecs = np.linalg.eig(G)
    if np.isclose(tr, 1.0, atol=atol):
        idx = int(np.argmin(np.abs(vals + 1.0)))
        normal = np.real(vecs[:, idx])
        normal /= np.max(np.abs(normal))
        info.update(kind="reflection", plane_normal=_canonical_direction(normal))
        return info

    H = -G
    cosang = np.clip((np.trace(H) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosang)))
    vals2, vecs2 = np.linalg.eig(H)
    idx = int(np.argmin(np.abs(vals2 - 1.0)))
    axis = np.real(vecs2[:, idx])
    axis /= np.max(np.abs(axis))
    info.update(kind="rotoinversion", proper_partner_angle_deg=angle, axis=_canonical_direction(axis))
    return info


def _canonical_direction(v: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, dtype=float)
    nz = np.where(np.abs(v) > 1e-8)[0]
    if len(nz) and v[nz[0]] < 0:
        v = -v
    v[np.abs(v) < 1e-10] = 0.0
    return tuple(float(x) for x in np.round(v, 8))


def matrix_order(g: sp.Matrix, max_order: int = 24) -> int:
    I = sp.eye(g.rows)
    x = I
    for n in range(1, max_order + 1):
        x = sp.simplify(x * g)
        if x == I:
            return n
    raise ValueError("Matrix order exceeds search bound")


@dataclass(frozen=True)
class SymmetryInfo:
    kind: str
    determinant: int
    order: int
    angle_deg: float | None = None
    axis: tuple[float, float, float] | None = None
    plane_normal: tuple[float, float, float] | None = None


def classify_symmetry(g: sp.Matrix) -> SymmetryInfo:
    """Classify an exact signed-permutation crystallographic symmetry.

    This helper is intended for the cubic parent group used in this project.
    It keeps reflections separate from generic improper operations because
    Cayron's Type-I construction depends specifically on parent mirrors.
    """
    d = classify_cubic_operation(g)
    return SymmetryInfo(
        kind=str(d["kind"]),
        determinant=int(d["det"]),
        order=int(d["order"]),
        angle_deg=float(d["angle_deg"]) if "angle_deg" in d else None,
        axis=d.get("axis"),
        plane_normal=d.get("plane_normal"),
    )
