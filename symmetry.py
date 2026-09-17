from __future__ import annotations

from itertools import permutations, product
import numpy as np
import sympy as sp


def cubic_full_m3m() -> list[sp.Matrix]:
    """All 48 signed permutation matrices of cubic m-3m (O_h)."""
    out: list[sp.Matrix] = []
    for perm in permutations(range(3)):
        P = np.zeros((3, 3), dtype=int)
        for col, row in enumerate(perm):
            P[row, col] = 1
        for signs in product((-1, 1), repeat=3):
            G = P @ np.diag(signs)
            out.append(sp.Matrix(G.tolist()))
    # exact deduplication
    unique = {_key(g): g for g in out}
    mats = list(unique.values())
    if len(mats) != 48:
        raise AssertionError(f"Expected 48 cubic operations, found {len(mats)}")
    return mats


def cubic_proper_rotations() -> list[sp.Matrix]:
    return [g for g in cubic_full_m3m() if int(g.det()) == 1]


def monoclinic_2_over_m_unique_b() -> list[sp.Matrix]:
    """Point group 2/m in the conventional unique-b direct basis."""
    return [
        sp.eye(3),
        sp.diag(-1, 1, -1),      # 2-fold around b
        -sp.eye(3),              # inversion
        sp.diag(1, -1, 1),       # mirror normal to b
    ]


def orthorhombic_mmm() -> list[sp.Matrix]:
    """Full orthorhombic mmm (D2h), order 8."""
    return [sp.diag(sx, sy, sz) for sx, sy, sz in product((-1, 1), repeat=3)]


def _key(g: sp.Matrix) -> tuple:
    return tuple(sp.simplify(x) for x in list(g))


def matrix_key(g: sp.Matrix) -> tuple:
    return _key(g)


def classify_cubic_operation(g: sp.Matrix) -> dict[str, object]:
    """Basic geometric classification for an orthogonal cubic symmetry matrix."""
    G = np.array(g, dtype=float)
    det = round(float(np.linalg.det(G)))
    tr = round(float(np.trace(G)), 12)
    info: dict[str, object] = {"det": det, "trace": tr}

    if np.allclose(G, np.eye(3)):
        info["kind"] = "identity"
        info["angle_deg"] = 0.0
        return info
    if np.allclose(G, -np.eye(3)):
        info["kind"] = "inversion"
        return info

    if det == 1:
        cosang = np.clip((np.trace(G) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cosang)))
        vals, vecs = np.linalg.eig(G)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        axis = np.real(vecs[:, idx])
        axis = axis / np.linalg.norm(axis)
        info.update(kind="rotation", angle_deg=angle, axis=axis)
        return info

    # improper operation. A true reflection has eigenvalues {1,1,-1}, trace=1.
    vals, vecs = np.linalg.eig(G)
    if np.isclose(np.trace(G), 1.0):
        idx = int(np.argmin(np.abs(vals + 1.0)))
        normal = np.real(vecs[:, idx])
        normal = normal / np.linalg.norm(normal)
        info.update(kind="reflection", plane_normal=normal)
    else:
        # -G is a proper rotation; describe improper operation via its proper partner.
        H = -G
        cosang = np.clip((np.trace(H) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cosang)))
        vals2, vecs2 = np.linalg.eig(H)
        idx = int(np.argmin(np.abs(vals2 - 1.0)))
        axis = np.real(vecs2[:, idx])
        axis = axis / np.linalg.norm(axis)
        info.update(kind="rotoinversion", proper_partner_angle_deg=angle, axis=axis)
    return info
