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
    """All 48 crystallographic operations of cubic m-3m (O_h).

    These are exact signed-permutation matrices in the conventional cubic
    crystallographic basis.  The *full* group is intentionally returned:
    reflections and other improper operations are required by Cayron's
    correspondence/operator construction and, in particular, parent mirror
    operations are the generators of CT Type-I twins.
    """
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
    I = sp.eye(3)
    mats.remove(I)
    return [I] + mats


def proper_operations(
    operations: list[sp.Matrix] | tuple[sp.Matrix, ...],
) -> list[sp.Matrix]:
    """Return the determinant +1 subgroup/part of a crystallographic operation list."""
    return [sp.Matrix(g) for g in operations if sp.simplify(sp.det(g) - 1) == 0]


def improper_operations(
    operations: list[sp.Matrix] | tuple[sp.Matrix, ...],
) -> list[sp.Matrix]:
    """Return determinant -1 crystallographic operations."""
    return [sp.Matrix(g) for g in operations if sp.simplify(sp.det(g) + 1) == 0]


def cubic_proper_rotations() -> list[sp.Matrix]:
    """The 24 proper rotations of cubic m-3m."""
    mats = proper_operations(cubic_full_m3m())
    if len(mats) != 24:
        raise AssertionError(f"Expected 24 proper cubic rotations, found {len(mats)}")
    return mats


def monoclinic_2_over_m_unique_b() -> list[sp.Matrix]:
    """Point group 2/m in conventional unique-b direct basis."""
    return [
        sp.eye(3),
        sp.diag(-1, 1, -1),  # 2-fold around b
        -sp.eye(3),  # inversion
        sp.diag(1, -1, 1),  # mirror normal to b
    ]


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
    info.update(
        kind="rotoinversion",
        proper_partner_angle_deg=angle,
        axis=_canonical_direction(axis),
    )
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
    """Classify an exact signed-permutation symmetry of the cubic parent.

    This helper is deliberately cubic-specific.  Generic point-group support
    lives in :mod:`point_groups`; do not use this classifier as a substitute
    for metric validation in a non-cubic basis.
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


@dataclass(frozen=True)
class CubicM3MInventory:
    """Explicit inventory of the 48 operations of cubic m-3m.

    Keeping this object visible is scientifically useful: CT topology uses the
    full 48-element crystallographic group, whereas EBSD/physical orientation
    matrices live in SO(3) and therefore use only the 24 proper rotations.
    """

    total: int
    proper: int
    improper: int
    identity: int
    inversion: int
    reflections: int
    proper_twofold_rotations: int
    proper_threefold_rotations: int
    proper_fourfold_rotations: int
    rotoinversions_order4: int
    rotoinversions_order6: int


def cubic_m3m_inventory() -> CubicM3MInventory:
    """Return and internally cross-check the exact m-3m operation counts."""
    counts = {
        "identity": 0,
        "inversion": 0,
        "reflections": 0,
        "proper_twofold": 0,
        "proper_threefold": 0,
        "proper_fourfold": 0,
        "rotoinv4": 0,
        "rotoinv6": 0,
    }
    operations = cubic_full_m3m()
    for g in operations:
        info = classify_symmetry(g)
        if info.kind == "identity":
            counts["identity"] += 1
        elif info.kind == "inversion":
            counts["inversion"] += 1
        elif info.kind == "reflection":
            counts["reflections"] += 1
        elif info.kind == "rotation":
            if info.order == 2:
                counts["proper_twofold"] += 1
            elif info.order == 3:
                counts["proper_threefold"] += 1
            elif info.order == 4:
                counts["proper_fourfold"] += 1
        elif info.kind == "rotoinversion":
            if info.order == 4:
                counts["rotoinv4"] += 1
            elif info.order == 6:
                counts["rotoinv6"] += 1

    proper = len(proper_operations(operations))
    improper = len(improper_operations(operations))
    inventory = CubicM3MInventory(
        total=len(operations),
        proper=proper,
        improper=improper,
        identity=counts["identity"],
        inversion=counts["inversion"],
        reflections=counts["reflections"],
        proper_twofold_rotations=counts["proper_twofold"],
        proper_threefold_rotations=counts["proper_threefold"],
        proper_fourfold_rotations=counts["proper_fourfold"],
        rotoinversions_order4=counts["rotoinv4"],
        rotoinversions_order6=counts["rotoinv6"],
    )
    expected = CubicM3MInventory(
        total=48,
        proper=24,
        improper=24,
        identity=1,
        inversion=1,
        reflections=9,
        proper_twofold_rotations=9,
        proper_threefold_rotations=8,
        proper_fourfold_rotations=6,
        rotoinversions_order4=6,
        rotoinversions_order6=8,
    )
    if inventory != expected:
        raise AssertionError(f"Unexpected cubic m-3m inventory: {inventory!r}")
    return inventory
