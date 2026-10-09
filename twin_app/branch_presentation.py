from __future__ import annotations

"""Stable *presentation* of computed rank-one branches without changing physics.

No classification is inferred from branch sign or shear magnitude. Physical
rank-one tensors are compared before two records are described as distinct.
"""

from dataclasses import dataclass
from typing import Sequence, Any
import numpy as np


@dataclass(frozen=True)
class PresentedTwinBranch:
    label: str
    representative: Any
    equivalent_source_indices: tuple[int, ...]
    source_branch_signs: tuple[int, ...]
    classifications: tuple[str, ...]
    all_habit_solutions: tuple[Any, ...]


def _tensor(construction: Any) -> np.ndarray:
    a = np.asarray(construction.a_parent_cartesian, dtype=float)
    n = np.asarray(construction.n_parent_cartesian, dtype=float)
    if a.shape != (3,) or n.shape != (3,) or not (np.isfinite(a).all() and np.isfinite(n).all()):
        raise ValueError("Twin branch lacks a finite physical rank-one tensor")
    if np.linalg.norm(a) <= 1e-15 or np.linalg.norm(n) <= 1e-15:
        raise ValueError("A physical twin branch needs nonzero shear and normal vectors")
    return np.outer(a, n)


def _fingerprint(construction: Any) -> tuple[float, ...]:
    """Sort independent of the solver's ±-branch enumeration order."""
    tensor = _tensor(construction)
    scale = max(float(np.linalg.norm(tensor)), 1e-15)
    normal = np.asarray(construction.n_parent_cartesian, float)
    normal = normal / np.linalg.norm(normal)
    axis = next((i for i in range(3) if abs(normal[i]) > 1e-12), 0)
    if normal[axis] < 0:
        normal = -normal
    return tuple(np.round(normal, 11)) + tuple(np.round((tensor / scale).ravel(), 11))


def distinct_twin_branches(
    constructions: Sequence[Any], *, relative_tolerance: float = 1e-8
) -> tuple[PresentedTwinBranch, ...]:
    """Group duplicate physical tensors but never average or invent their data.

    A duplicate tensor may still carry alternative *classification evidence*
    or habit solutions. All source indices are retained for expert inspection.
    """
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    ordered = sorted(range(len(constructions)), key=lambda i: (_fingerprint(constructions[i]), i))
    groups: list[list[int]] = []
    for index in ordered:
        candidate = _tensor(constructions[index])
        for group in groups:
            target = _tensor(constructions[group[0]])
            scale = max(float(np.linalg.norm(target)), float(np.linalg.norm(candidate)), 1e-15)
            if float(np.linalg.norm(candidate-target)) <= relative_tolerance * scale:
                group.append(index)
                break
        else:
            groups.append([index])
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def group_habits(indices: list[int]) -> tuple[Any, ...]:
        seen = set()
        unique = []
        for j in indices:
            for sol in getattr(constructions[j], "habit_solutions", ()):
                b = np.asarray(sol.shape_vector_parent_cartesian, float)
                m = np.asarray(sol.habit_normal_parent_cartesian, float)
                key = (
                    round(float(sol.other_variant_volume_fraction), 10),
                    tuple(np.round(np.outer(b, m).ravel(), 10)),
                )
                if key not in seen:
                    seen.add(key)
                    unique.append(sol)
        return tuple(unique)
    return tuple(
        PresentedTwinBranch(
            label=f"Interface {alphabet[i] if i < len(alphabet) else str(i+1)}",
            representative=constructions[sorted(indices, key=lambda j: (-len(getattr(constructions[j], "habit_solutions", ())), j))[0]],
            equivalent_source_indices=tuple(indices),
            source_branch_signs=tuple(int(constructions[j].branch) for j in indices),
            classifications=tuple(sorted({str(getattr(constructions[j], "classification", "unresolved")) for j in indices})),
            all_habit_solutions=group_habits(indices),
        )
        for i, indices in enumerate(groups)
    )
