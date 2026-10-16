"""P0 rigorous screening of non-conventional martensite rank-one connections.

A proper parent lattice twofold Q satisfying U_j=Q U_i Q^T is a necessary
condition for a conventional Mallard Type-I/Type-II/Compound twin relation.
Absence EXCLUDES conventional classification; presence does NOT certify a type.
Any rank-one connection excluded by this screening remains a valid mathematical
compatibility but must not be presented as a classical twinning mode.

Sources: Pitteri & Zanzotto (1998), Acta Materialia 46, 225-237,
DOI: 10.1016/S1359-6454(97)00203-6;
Tahseen & Dabade (2026), JMPS 206, 106409,
DOI: 10.1016/j.jmps.2025.106409.
This module does not encode material-specific results or known shears.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
import numpy as np

CONVENTIONAL = frozenset({"Type I", "Type II", "Compound"})
NONCONVENTIONAL = "Nonconventional rank-one (parent twofold excluded)"


def _rel(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, ord="fro") / max(
        float(np.linalg.norm(left, ord="fro")),
        float(np.linalg.norm(right, ord="fro")), 1e-14,
    ))


@dataclass(frozen=True)
class TwofoldScreen:
    operation_count: int
    proper_twofold_count: int
    mapping_twofold_indices: tuple[int, ...]
    nearest_mapping_error: float | None
    tolerance: float

    @property
    def conventional_twofold_excluded(self) -> bool:
        return not self.mapping_twofold_indices


def screen_parent_twofold(
    U_i: Any, U_j: Any, *, parent_metric: Any,
    parent_group: Sequence[Any], tolerance: float = 1e-8,
) -> TwofoldScreen:
    """Check EVERY supplied parent point-group operation in Cartesian space.

    This checks a necessary conventional-twin symmetry, NOT Type-I/II details.
    It fails closed for invalid/empty/non-metric-preserving parent groups.
    """
    metric = np.asarray(parent_metric, dtype=float)
    left, right = np.asarray(U_i, dtype=float), np.asarray(U_j, dtype=float)
    if not np.isfinite(tolerance) or not 1e-13 <= tolerance <= 1e-4:
        raise ValueError("A finite numerical tolerance in [1e-13, 1e-4] is required")
    for name, matrix in (("metric", metric), ("U_i", left), ("U_j", right)):
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite 3x3 matrix")
        if _rel(matrix, matrix.T) > tolerance:
            raise ValueError(f"{name} must be symmetric")
        if float(np.linalg.eigvalsh(matrix)[0]) <= 0:
            raise ValueError(f"{name} must be positive definite")
    if len(parent_group) < 1:
        raise ValueError("Cannot exhaust parent twofolds without its symmetry group")
    ev, v = np.linalg.eigh(metric)
    B = (v * np.sqrt(ev)) @ v.T
    Bi = (v * (1.0 / np.sqrt(ev))) @ v.T
    qualifying: list[int] = []
    errors: list[float] = []
    twofolds = 0
    for index, operation in enumerate(parent_group):
        g = np.asarray(operation, dtype=float)
        if g.shape != (3, 3) or not np.isfinite(g).all():
            raise ValueError(f"Invalid parent symmetry operation #{index}")
        if _rel(g.T @ metric @ g, metric) > tolerance:
            raise ValueError(f"Parent symmetry #{index} violates the lattice metric")
        Q = B @ g @ Bi
        if _rel(Q.T @ Q, np.eye(3)) > tolerance:
            raise ValueError(f"Parent symmetry #{index} is not Cartesian-orthogonal")
        if abs(float(np.linalg.det(Q)) - 1.0) > tolerance:
            continue
        if not (abs(float(np.trace(Q)) + 1.0) <= tolerance and
                _rel(Q @ Q, np.eye(3)) <= tolerance):
            continue
        twofolds += 1
        error = _rel(Q @ left @ Q.T, right)
        errors.append(error)
        if error <= tolerance:
            qualifying.append(index)
    return TwofoldScreen(
        operation_count=len(parent_group), proper_twofold_count=twofolds,
        mapping_twofold_indices=tuple(qualifying),
        nearest_mapping_error=min(errors, default=None), tolerance=tolerance,
    )


@dataclass(frozen=True)
class P0ClassicalAudit:
    conventional_count: int
    nonconventional_proved_count: int
    unresolved_with_twofold_count: int
    certificates: tuple[dict[str, Any], ...]


def screen_unclassified_report(
    families: Sequence[Any], *, parent_metric: Any, parent_group: Sequence[Any],
    stretch_variants: Mapping[int, Any], tolerance: float = 1e-8,
) -> tuple[tuple[Any, ...], P0ClassicalAudit]:
    """Preserve all interfaces and habits; add a proof for nonclassical records.

    Does NOT attempt to create new Type-I, Type-II or Compound assignments.
    Does NOT alter D=a⊗n, K1, eta1, lambda, b, m, branch or traceability.
    """
    rewritten = []
    proofs = []
    conventional = nonconventional = unresolved = 0
    for family in families:
        pairs = []
        for pair in family.pair_records:
            structures = []
            for branch in pair.constructions:
                if branch.classification in CONVENTIONAL:
                    conventional += 1
                    structures.append(branch)
                    continue
                if pair.stretch_i is None or pair.stretch_j is None:
                    raise ValueError("Rank-one branch without an indexed stretch pair")
                if pair.stretch_i not in stretch_variants or pair.stretch_j not in stretch_variants:
                    raise ValueError("Unknown stretch index in twin construction")
                if (not np.isfinite(branch.rank_one_residual) or
                        branch.rank_one_residual > max(1e-6, 100 * tolerance)):
                    raise ValueError("Cannot classify a rank-one branch with invalid residual")
                screen = screen_parent_twofold(
                    stretch_variants[pair.stretch_i], stretch_variants[pair.stretch_j],
                    parent_metric=parent_metric, parent_group=parent_group,
                    tolerance=tolerance,
                )
                witness = {
                    "pair": pair.pair_id, "branch": branch.branch,
                    "proper_parent_twofolds_examined": screen.proper_twofold_count,
                    "parent_symmetries_examined": screen.operation_count,
                    "matching_twofold_indices": list(screen.mapping_twofold_indices),
                    "nearest_twofold_mapping_error": screen.nearest_mapping_error,
                    "tolerance": tolerance,
                    "conclusion": "conventional_parent_twofold_excluded"
                    if screen.conventional_twofold_excluded else
                    "classical_type_requires_additional_cross_lock",
                }
                proofs.append(witness)
                if screen.conventional_twofold_excluded:
                    nonconventional += 1
                    structures.append(replace(
                        branch, classification=NONCONVENTIONAL,
                        classification_status=(
                            "PROVED: no proper 180-degree symmetry in the supplied "
                            f"parent group ({screen.operation_count} operations; "
                            f"{screen.proper_twofold_count} twofolds) maps the "
                            "two ordered stretch variants. Consequently, the classical "
                            "Type I/II/Compound parent-twofold prerequisite is absent. "
                            "Rank-one compatibility remains valid; genericity and "
                            "experimental occurrence are NOT established. "
                            "Previous audit: " + branch.classification_status
                        ),
                    ))
                else:
                    unresolved += 1
                    structures.append(replace(
                        branch, classification_status=(
                            "UNRESOLVED: a parent twofold maps these stretches, "
                            "but Type I/II/Compound geometry lacks a verified direct "
                            "cross-lock. No type inferred. Previous audit: "
                            + branch.classification_status
                        ),
                    ))
            pairs.append(replace(pair, constructions=tuple(structures)))
        rewritten.append(replace(family, pair_records=tuple(pairs)))
    return tuple(rewritten), P0ClassicalAudit(
        conventional_count=conventional,
        nonconventional_proved_count=nonconventional,
        unresolved_with_twofold_count=unresolved,
        certificates=tuple(proofs),
    )
