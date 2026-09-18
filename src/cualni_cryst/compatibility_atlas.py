from __future__ import annotations

"""A/M and A/M/M compatibility engine for the truth-locked DO3 -> 6M branch.

The module is intentionally "backend first": it contains no GUI state and no
plotting. Every public result is a small dataclass with plain numerical fields,
so a future interactive crystallography application can call the same tested
physics engine and serialize the results without reimplementing the theory.

Three levels are kept separate:

1. measured/source state:
   evaluate what the supplied lattice parameters actually satisfy;
2. approximate CMC diagnostic:
   report proximity to compatibility, but never promote it to an exact habit;
3. exact-compatible projection:
   deliberately change a selected lattice parameter to a mathematically exact
   compatibility manifold, clearly labelled HYPOTHETICAL_TEST, and then test
   A/M/M supercompatibility, cofactor conditions and PTMC independently.

No projected result is presented as a measurement.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .ball_james import single_variant_austenite_habit_solutions
from .cofactor import CofactorResult, evaluate_cofactor_conditions
from .ct import (
    CTAMResult,
    analyze_austenite_martensite,
    ct_supercompatibility_residual,
    ips_shear_from_habit_plane,
)
from .cualni_models import (
    TransformationBranch,
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from .james_hane import six_m_exact_compatibility_angles
from .lattice import Lattice, metric_sqrt, plane_to_unit_normal
from .ptmc import PTMCSolution, solve_ptmc_laminate
from .stretch import (
    generate_stretch_variants,
    principal_stretches,
    stretch_from_metrics,
)
from .symmetry import cubic_proper_rotations
from .twin_compare import TwinAtlas, build_twin_atlas, projective_angle_deg


@dataclass(frozen=True)
class AMHabitComparison:
    """Independent CMC-vs-Ball-James check for one exact A/M habit branch."""

    cmc_habit_index: int
    ball_james_branch: int
    plane_angle_deg: float
    ball_james_rank_one_residual: float


@dataclass(frozen=True)
class AMStateSummary:
    """A/M compatibility summary for one specified physical or hypothetical state."""

    label: str
    status: str
    a0: float
    a: float
    b: float
    c: float
    beta_deg: float
    lambdas: tuple[float, float, float]
    lambda2_residual: float
    normalized_cmc_eigenvalues: tuple[float, float, float]
    exact_compatible: bool
    degeneracy_order: int
    nearest_cmc_residual: float
    n_exact_habit_planes: int
    n_approximate_habit_planes: int
    approximate_signature_admissible: bool


@dataclass(frozen=True)
class TwinSystemCompatibility:
    """One exact M/M twin system tested against exact A/M compatibility."""

    operator_index: int
    relation_index: int
    target_stretch_index: int
    twin_kind: str
    compound: bool
    twin_shear: float
    best_habit_index: int
    supercompatibility_residual: float
    shear_direction_angle_deg: float
    cc1_residual: float
    cc2_residual: float
    cc2_simplified: float
    cc3_margin: float
    cofactor_satisfied: bool
    ptmc_volume_fractions: tuple[float, ...]
    ptmc_max_middle_stretch_residual: float


@dataclass(frozen=True)
class CompatibilityAtlas:
    """Serializable backend result suitable for reports or a future GUI/API."""

    state: AMStateSummary
    am_habit_crosscheck: tuple[AMHabitComparison, ...]
    twin_systems: tuple[TwinSystemCompatibility, ...]
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary for a future interactive frontend."""

        return asdict(self)


@dataclass(frozen=True)
class BenchmarkProjection:
    """Observed literature benchmark plus a clearly labelled exact-compatible projection."""

    observed: CompatibilityAtlas
    projected: CompatibilityAtlas
    projected_beta_deg: float
    beta_shift_deg: float
    exact_beta_candidates_deg: tuple[float, float]


def _unit(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(x))
    if n <= 1e-15:
        raise ValueError("Cannot normalize a zero vector")
    return x / n


def _parent_plane_to_orthonormal_normal(
    p_a: np.ndarray, M_a: np.ndarray
) -> np.ndarray:
    """Map a parent reciprocal covector to a unit normal in the whitened basis."""

    n_crystal = plane_to_unit_normal(p_a, M_a)
    return _unit(metric_sqrt(M_a) @ n_crystal)


def _parent_direct_to_orthonormal(
    u_a: np.ndarray, M_a: np.ndarray
) -> np.ndarray:
    """Map a parent direct vector to a unit vector in the whitened basis."""

    return _unit(metric_sqrt(M_a) @ np.asarray(u_a, dtype=float).reshape(3))


def _projective_direct_angle_deg(
    u: np.ndarray, v: np.ndarray, M_a: np.ndarray
) -> float:
    return projective_angle_deg(
        _parent_direct_to_orthonormal(u, M_a),
        _parent_direct_to_orthonormal(v, M_a),
    )


def _stretch_family(
    branch: TransformationBranch, M_a: np.ndarray, M_m: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    U0 = stretch_from_metrics(M_a, M_m, branch.correspondence)
    variants = generate_stretch_variants(
        U0,
        [np.asarray(q, dtype=float) for q in cubic_proper_rotations()],
    )
    return U0, variants


def _state_summary(
    label: str,
    status: str,
    A: Lattice,
    M: Lattice,
    branch: TransformationBranch,
    am: CTAMResult,
) -> AMStateSummary:
    U = stretch_from_metrics(A.metric(), M.metric(), branch.correspondence)
    lambdas, _ = principal_stretches(U)
    return AMStateSummary(
        label=label,
        status=status,
        a0=float(A.a),
        a=float(M.a),
        b=float(M.b),
        c=float(M.c),
        beta_deg=float(M.beta_deg),
        lambdas=tuple(float(x) for x in lambdas),
        lambda2_residual=float(lambdas[1] - 1.0),
        normalized_cmc_eigenvalues=tuple(float(x) for x in am.analysis.eigenvalues),
        exact_compatible=bool(am.analysis.exact_compatible),
        degeneracy_order=int(am.analysis.degeneracy_order),
        nearest_cmc_residual=float(am.analysis.nearest_zero_residual),
        n_exact_habit_planes=len(am.exact_habit_planes),
        n_approximate_habit_planes=len(am.approximate.candidate_planes),
        approximate_signature_admissible=bool(am.approximate.admissible_signature),
    )


def _match_exact_habits_to_ball_james(
    A: Lattice,
    M: Lattice,
    branch: TransformationBranch,
    am: CTAMResult,
) -> tuple[AMHabitComparison, ...]:
    """Cross-check exact CMC habits with the independent R U - I = b⊗m theorem."""

    if not am.analysis.exact_compatible:
        return ()

    U = stretch_from_metrics(A.metric(), M.metric(), branch.correspondence)
    bj = single_variant_austenite_habit_solutions(U, tol=1e-8)
    if len(am.exact_habit_planes) != len(bj):
        raise AssertionError(
            "Exact CMC and Ball-James single-variant habit counts disagree: "
            f"{len(am.exact_habit_planes)} vs {len(bj)}"
        )

    cmc_normals = [
        _parent_plane_to_orthonormal_normal(p, A.metric())
        for p in am.exact_habit_planes
    ]
    available = set(range(len(bj)))
    rows: list[AMHabitComparison] = []

    for i, normal in enumerate(cmc_normals):
        best = min(
            available,
            key=lambda j: projective_angle_deg(normal, bj[j].n),
        )
        angle = projective_angle_deg(normal, bj[best].n)
        rows.append(
            AMHabitComparison(
                cmc_habit_index=i,
                ball_james_branch=int(bj[best].branch),
                plane_angle_deg=float(angle),
                ball_james_rank_one_residual=float(bj[best].residual),
            )
        )
        available.remove(best)

    return tuple(rows)


def _best_supercompatibility_branch(
    habit_planes: tuple[np.ndarray, ...],
    A: Lattice,
    M: Lattice,
    branch: TransformationBranch,
    twin,
) -> tuple[int, float, float]:
    """Return habit index, Cayron epsilon, and angle(d_A, twin direction)."""

    candidates: list[tuple[int, float, float]] = []
    for i, p in enumerate(habit_planes):
        d_a = ips_shear_from_habit_plane(
            p, A.metric(), M.metric(), branch.correspondence
        )
        eps = ct_supercompatibility_residual(
            p,
            d_a,
            twin.plane_a,
            twin.direction_a,
            twin.shear,
            A.metric(),
        )
        angle = _projective_direct_angle_deg(
            d_a, twin.direction_a, A.metric()
        )
        candidates.append((i, float(eps), float(angle)))

    if not candidates:
        raise ValueError("Exact A/M habit planes are required for supercompatibility")

    return min(candidates, key=lambda x: x[1])


def _cofactor_and_ptmc(
    Uj: np.ndarray,
    twin_solution,
) -> tuple[CofactorResult, tuple[PTMCSolution, ...]]:
    cofactor = evaluate_cofactor_conditions(
        Uj,
        twin_solution.a,
        twin_solution.n,
        tol=1e-8,
    )
    ptmc = tuple(
        solve_ptmc_laminate(
            Uj,
            twin_solution.a,
            twin_solution.n,
            tol=1e-9,
        )
    )
    return cofactor, ptmc


def analyze_do3_6m_state(
    A: Lattice,
    M: Lattice,
    *,
    label: str,
    status: str,
    branch: TransformationBranch | None = None,
    exact_tol: float = 1e-8,
) -> CompatibilityAtlas:
    """Run the A/M and, when exact A/M exists, A/M/M theory stack.

    Parameters
    ----------
    A, M
        Dimensional parent and martensite lattices. No silent normalization.
    label
        Human-readable state name.
    status
        Provenance label such as SOURCE_MEASURED or HYPOTHETICAL_TEST.
    exact_tol
        Numerical equality tolerance. It affects the classification of exact
        compatibility; the raw residuals are always returned.

    Returns
    -------
    CompatibilityAtlas
        If the supplied state is not exactly A/M compatible, ``twin_systems``
        is intentionally empty. Approximate CMC planes remain diagnostics only.
    """

    branch = branch or do3_to_6m_branch()
    am = analyze_austenite_martensite(
        A.metric(),
        M.metric(),
        branch.correspondence,
        tol=exact_tol,
    )
    summary = _state_summary(label, status, A, M, branch, am)
    crosscheck = _match_exact_habits_to_ball_james(A, M, branch, am)

    if not am.analysis.exact_compatible:
        return CompatibilityAtlas(
            state=summary,
            am_habit_crosscheck=crosscheck,
            twin_systems=(),
            note=(
                "No exact A/M/M supercompatibility values are reported because "
                "the supplied state does not satisfy exact CMC degeneracy. "
                "Approximate CMC candidates are diagnostics only."
            ),
        )

    twin_atlas: TwinAtlas = build_twin_atlas(
        branch, A.metric(), M.metric(), tol=1e-10
    )
    _, stretches = _stretch_family(branch, A.metric(), M.metric())

    rows: list[TwinSystemCompatibility] = []
    for op in twin_atlas.operators:
        for relation_index, relation in enumerate(op.relations):
            Uj = stretches[relation.target_stretch_index]

            for twin_kind, ct_twin, bj_twin in (
                ("I", relation.ct_type_i, relation.bj_type_i),
                ("II", relation.ct_type_ii, relation.bj_type_ii),
            ):
                habit_index, eps, angle = _best_supercompatibility_branch(
                    am.exact_habit_planes,
                    A,
                    M,
                    branch,
                    ct_twin,
                )
                cofactor, ptmc = _cofactor_and_ptmc(Uj, bj_twin)
                max_ptmc_residual = max(
                    (x.middle_stretch_residual for x in ptmc),
                    default=float("nan"),
                )
                rows.append(
                    TwinSystemCompatibility(
                        operator_index=op.operator_index,
                        relation_index=relation_index,
                        target_stretch_index=relation.target_stretch_index,
                        twin_kind=twin_kind,
                        compound=relation.compound,
                        twin_shear=float(ct_twin.shear),
                        best_habit_index=habit_index,
                        supercompatibility_residual=eps,
                        shear_direction_angle_deg=angle,
                        cc1_residual=float(cofactor.cc1_residual),
                        cc2_residual=float(cofactor.cc2_residual),
                        cc2_simplified=float(cofactor.cc2_simplified),
                        cc3_margin=float(cofactor.cc3_margin),
                        cofactor_satisfied=bool(cofactor.satisfied),
                        ptmc_volume_fractions=tuple(
                            float(x.volume_fraction) for x in ptmc
                        ),
                        ptmc_max_middle_stretch_residual=float(max_ptmc_residual),
                    )
                )

    return CompatibilityAtlas(
        state=summary,
        am_habit_crosscheck=crosscheck,
        twin_systems=tuple(rows),
        note=(
            "Exact A/M compatibility is satisfied for this state. Each exact "
            "M/M relation is tested by Cayron shear/shear epsilon, independently "
            "by Chen-James cofactor conditions and by PTMC laminate roots."
        ),
    )


def nearest_exact_6m_beta_projection(
    A: Lattice,
    M: Lattice,
) -> tuple[Lattice, float, tuple[float, float]]:
    """Project only beta to the nearest James-Hane Eq.(25) exact A/M value.

    This is a hypothetical mathematical control, not a fitted or measured state.
    The dimensional a, b, c and parent a0 are left unchanged.
    """

    candidates = six_m_exact_compatibility_angles(A.a, M.a, M.c)
    beta = min(candidates, key=lambda x: abs(x - M.beta_deg))
    projected = Lattice.monoclinic_unique_b(
        M.a,
        M.b,
        M.c,
        beta,
        label=(
            "HYPOTHETICAL_TEST: James-Hane Eq.(25) exact-compatible beta projection"
        ),
    )
    return projected, float(beta), tuple(float(x) for x in candidates)


def james_hane_benchmark_projection() -> BenchmarkProjection:
    """Analyze the rounded literature benchmark and an explicit exact-beta control."""

    A, M = james_hane_6m_example_lattices()
    observed = analyze_do3_6m_state(
        A,
        M,
        label="James-Hane Table-4 rounded Cu-Al-Ni benchmark",
        status="SOURCE_MEASURED_ROUNDED_BENCHMARK",
    )

    M_projected, beta, candidates = nearest_exact_6m_beta_projection(A, M)
    projected = analyze_do3_6m_state(
        A,
        M_projected,
        label="Nearest James-Hane Eq.(25) exact-beta projection",
        status="HYPOTHETICAL_TEST",
    )

    return BenchmarkProjection(
        observed=observed,
        projected=projected,
        projected_beta_deg=beta,
        beta_shift_deg=float(beta - M.beta_deg),
        exact_beta_candidates_deg=candidates,
    )
