from __future__ import annotations

"""Equation-preserving adaptive numerical kernels for metric compatibility.

The scientific eigenproblem is never changed::

    G v = mu M_a v,        G = C.T @ M_m @ C.

Binary64 is used when it is demonstrably safe for the requested decision.
Near the exact-compatibility boundary, under severe conditioning, or when two
algebraically equivalent binary64 routes disagree materially, the *same*
finite-precision input problem is re-evaluated with arbitrary precision.

Important distinction
---------------------
This module does not alter a scientific tolerance in order to stabilize a
classification.  It changes arithmetic precision.  Exact-rational root
certification is intentionally kept in the independent validation package,
not in production, so the production implementation and the oracle remain
separate.
"""

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction

import mpmath as mp
import numpy as np
from scipy.linalg import eigh


_FLOAT_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class AdaptiveMetricEigensystem:
    mu: np.ndarray
    eigenvectors_crystal: np.ndarray
    escalated: bool
    source: str
    condition_parent: float
    condition_product: float
    condition_correspondence: float
    condition_pulled_metric: float
    route_disagreement: float
    distance_to_compatibility_boundary: float
    metric_orthonormality_residual: float
    eigen_equation_residual: float


def _relative_symmetry_residual(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=float)
    return float(
        np.linalg.norm(A - A.T, ord="fro")
        / max(np.linalg.norm(A, ord="fro"), np.finfo(float).tiny)
    )


def _fraction(x: float) -> Fraction:
    num, den = float(x).as_integer_ratio()
    return Fraction(num, den)


def _canonical_metric_binary64(M: np.ndarray) -> np.ndarray:
    """Canonical binary64 representation of a symmetric metric tensor.

    Metric tensors are symmetric objects.  Small antisymmetric roundoff from an
    upstream matrix product is removed before exact-binary certification.  The
    resulting binary64 entries define the numerical metric used consistently
    by production and validation.
    """
    A = np.asarray(M, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(A)):
        raise ValueError("metric contains non-finite entries")
    return 0.5 * (A + A.T)


def _exact_spd_binary64(
    A: np.ndarray,
) -> tuple[bool, tuple[Fraction, Fraction, Fraction]]:
    """Exact Sylvester test for a canonical 3x3 binary64 symmetric matrix."""
    A = np.asarray(A, dtype=float).reshape(3, 3)
    a00, a01, a02 = map(_fraction, A[0])
    a10, a11, a12 = map(_fraction, A[1])
    a20, a21, a22 = map(_fraction, A[2])
    d1 = a00
    d2 = a00 * a11 - a01 * a10
    d3 = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    return bool(d1 > 0 and d2 > 0 and d3 > 0), (d1, d2, d3)


def _exact_det_binary64(A: np.ndarray) -> Fraction:
    A = np.asarray(A, dtype=float).reshape(3, 3)
    a00, a01, a02 = map(_fraction, A[0])
    a10, a11, a12 = map(_fraction, A[1])
    a20, a21, a22 = map(_fraction, A[2])
    return (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )


def _validate_metric(M: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(M, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} contains non-finite entries")
    asym = _relative_symmetry_residual(raw)
    if asym > 1.0e-12:
        raise ValueError(f"{name} must be symmetric; relative asymmetry={asym:g}")

    A = _canonical_metric_binary64(raw)
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} overflowed while canonicalizing symmetry")

    eig = np.linalg.eigvalsh(A)
    scale = max(float(np.linalg.norm(A, ord=2)), np.finfo(float).tiny)
    # A backward-stable binary64 eigensolver can lose the sign of a tiny
    # eigenvalue when the metric is severely conditioned.  Near that numerical
    # boundary, decide SPD-ness from the exact rational values of the canonical
    # binary64 matrix.  No eigenvalue is clipped or repaired.
    ambiguous = float(np.min(eig)) <= 256.0 * _FLOAT_EPS * scale
    if ambiguous:
        spd, minors = _exact_spd_binary64(A)
        if not spd:
            signs = tuple(int(x > 0) - int(x < 0) for x in minors)
            raise ValueError(
                f"{name} is not positive definite as an exact canonical "
                f"binary64 symmetric matrix; binary64 eigenvalues={eig}; "
                f"leading-principal-minor signs={signs}"
            )
    elif np.min(eig) <= 0.0:
        raise ValueError(f"{name} must be positive definite; eigenvalues={eig}")
    return A


def _validate_correspondence(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(C)):
        raise ValueError("correspondence contains non-finite entries")
    s = np.linalg.svd(C, compute_uv=False)
    if s[0] == 0.0:
        raise ValueError("correspondence is the zero matrix")
    if s[-1] <= 64.0 * _FLOAT_EPS * s[0]:
        # Near the binary64 SVD resolution limit, singular/nonsingular is
        # decided exactly from the supplied IEEE-754 entries.
        if _exact_det_binary64(C) == 0:
            raise ValueError(
                "correspondence is singular; "
                f"binary64 singular_values={s}"
            )
    return C


def validate_metric_pencil_inputs(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _validate_metric(M_a, "M_a"),
        _validate_metric(M_m, "M_m"),
        _validate_correspondence(C),
    )


def pulled_metric_literal(M_m: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Return exactly the binary64 expression ``C.T @ M_m @ C``.

    No post-symmetrization is performed here.  Algorithms that require a
    symmetric storage representation create it locally without redefining the
    crystallographic pull-back.
    """
    return np.asarray(C, float).T @ np.asarray(M_m, float) @ np.asarray(C, float)


def _symmetric_solver_view(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def _factorized_pulled_metric(M_m: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Algebraically identical binary64 assembly: (B_m C)^T (B_m C)."""
    B_m = np.linalg.cholesky(M_m).T
    X = B_m @ C
    G = X.T @ X
    return 0.5 * (G + G.T)


def _sort_eigensystem(mu: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(mu)
    mu = np.asarray(mu[order], dtype=float)
    V = np.asarray(V[:, order], dtype=float)
    return mu, V


def factorized_double_eigensystem(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast binary64 solution of the unchanged generalized metric equation."""
    M_a, M_m, C = validate_metric_pencil_inputs(M_a, M_m, C)
    G = _factorized_pulled_metric(M_m, C)
    mu, V = eigh(G, M_a, check_finite=False)
    mu, V = _sort_eigensystem(mu, V)
    if np.min(mu) <= 0.0:
        raise ArithmeticError(f"generalized metric eigenvalues must be positive; mu={mu}")
    return mu, V, G


def literal_double_eigenvalues(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """Binary64 spectrum using the literal matrix product, for diagnostics."""
    M_a, M_m, C = validate_metric_pencil_inputs(M_a, M_m, C)
    G = _symmetric_solver_view(pulled_metric_literal(M_m, C))
    mu = eigh(G, M_a, eigvals_only=True, check_finite=False)
    mu = np.sort(np.asarray(mu, dtype=float))
    return mu


def _array_key(A: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(A, dtype=np.float64)).tobytes(order="C")


def _array_from_key(key: bytes) -> np.ndarray:
    return np.frombuffer(key, dtype=np.float64).copy().reshape(3, 3)


def _mp_exact_float(x: float) -> mp.mpf:
    num, den = float(x).as_integer_ratio()
    return mp.mpf(num) / mp.mpf(den)


def _mp_matrix(A: np.ndarray) -> mp.matrix:
    A = np.asarray(A, dtype=float)
    return mp.matrix([[_mp_exact_float(x) for x in row] for row in A])


def _deterministic_column_signs(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float).copy()
    for j in range(V.shape[1]):
        k = int(np.argmax(np.abs(V[:, j])))
        if V[k, j] < 0.0:
            V[:, j] *= -1.0
    return V


@lru_cache(maxsize=512)
def _high_precision_cached(
    key_a: bytes,
    key_m: bytes,
    key_c: bytes,
    digits: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Cached arbitrary-precision solve of the exact binary64 input pencil."""
    M_a = _array_from_key(key_a)
    M_m = _array_from_key(key_m)
    C = _array_from_key(key_c)

    with mp.workdps(int(digits)):
        A = _mp_matrix(M_a)
        M = _mp_matrix(M_m)
        X = _mp_matrix(C)
        G = X.T * M * X

        # A = L L^T.  q=L^T v gives the standard symmetric problem
        # (L^-1 G L^-T) q = mu q.
        L = mp.cholesky(A)
        Linv = L ** -1
        H = Linv * G * Linv.T
        H = (H + H.T) / 2
        values, Q = mp.eigsy(H)
        Vmp = Linv.T * Q

        pairs = sorted(
            [
                (
                    mp.mpf(values[i]),
                    [mp.mpf(Vmp[j, i]) for j in range(3)],
                )
                for i in range(3)
            ],
            key=lambda item: item[0],
        )

        mu = tuple(float(item[0]) for item in pairs)
        V = np.column_stack(
            [np.array([float(x) for x in item[1]], dtype=float) for item in pairs]
        )
        V = _deterministic_column_signs(V)
        return mu, tuple(float(x) for x in V.ravel(order="C"))


def high_precision_metric_eigensystem(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    digits: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    """Arbitrary-precision eigensystem for the exact IEEE-754 input values."""
    M_a, M_m, C = validate_metric_pencil_inputs(M_a, M_m, C)
    if digits < 50:
        raise ValueError("digits must be >= 50")
    mu_t, V_t = _high_precision_cached(
        _array_key(M_a), _array_key(M_m), _array_key(C), int(digits)
    )
    mu = np.asarray(mu_t, dtype=float)
    V = np.asarray(V_t, dtype=float).reshape(3, 3)
    if np.min(mu) <= 0.0:
        raise ArithmeticError(f"high-precision metric eigenvalues are not positive: {mu}")
    return mu, V


def _residuals(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    mu: np.ndarray,
    V: np.ndarray,
) -> tuple[float, float]:
    G = pulled_metric_literal(M_m, C)
    R = G @ V - M_a @ V @ np.diag(mu)
    denom = (
        np.linalg.norm(G, ord="fro") * np.linalg.norm(V, ord="fro")
        + np.linalg.norm(M_a, ord="fro")
        * np.linalg.norm(V @ np.diag(mu), ord="fro")
    )
    eig_res = float(
        np.linalg.norm(R, ord="fro") / max(float(denom), np.finfo(float).tiny)
    )
    metric_res = float(np.linalg.norm(V.T @ M_a @ V - np.eye(3), ord="fro"))
    return eig_res, metric_res


def adaptive_metric_eigensystem(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    decision_tol: float = 1.0e-8,
    boundary_guard_factor: float = 1024.0,
    conditioning_guard: float = 1.0e8,
    route_disagreement_fraction: float = 0.125,
    precision_digits: int = 100,
) -> AdaptiveMetricEigensystem:
    """Solve the unchanged metric pencil with precision escalation when needed.

    ``decision_tol`` is the scientific/numerical compatibility threshold used
    by the caller.  It is never enlarged here.  The other parameters govern
    only *when arithmetic precision is escalated*.
    """
    if decision_tol <= 0.0:
        raise ValueError("decision_tol must be positive")
    if boundary_guard_factor <= 1.0:
        raise ValueError("boundary_guard_factor must be > 1")
    if conditioning_guard <= 1.0:
        raise ValueError("conditioning_guard must be > 1")
    if not (0.0 < route_disagreement_fraction < 1.0):
        raise ValueError("route_disagreement_fraction must lie in (0,1)")

    M_a, M_m, C = validate_metric_pencil_inputs(M_a, M_m, C)
    cond_a = float(np.linalg.cond(M_a))
    cond_m = float(np.linalg.cond(M_m))
    cond_c = float(np.linalg.cond(C))

    fast_failed = False
    try:
        mu_fast, V_fast, G_fast = factorized_double_eigensystem(M_a, M_m, C)
        mu_literal = literal_double_eigenvalues(M_a, M_m, C)
        distance = float(np.min(np.abs(mu_fast - 1.0)))
        disagreement = float(np.max(np.abs(mu_fast - mu_literal)))
        cond_g = float(np.linalg.cond(G_fast))
    except (np.linalg.LinAlgError, ValueError, ArithmeticError):
        # An algebraically valid exact-binary64 SPD pencil may be beyond a
        # reliable binary64 Cholesky/eigh route.  This is a reason to escalate,
        # not to alter the equation or reject prematurely.
        fast_failed = True
        mu_fast = V_fast = None
        distance = 0.0
        disagreement = float("inf")
        cond_g = float("inf")

    near_boundary = distance <= boundary_guard_factor * float(decision_tol)
    strongly_conditioned = max(cond_a, cond_m, cond_c, cond_g) >= conditioning_guard
    routes_disagree = disagreement >= route_disagreement_fraction * float(decision_tol)

    # A forward-accuracy guard complements the backward residual.  Because C
    # enters twice in G=C.T M_m C, correspondence conditioning enters this
    # conservative risk estimate quadratically.  The guard is tied to the
    # unchanged scientific decision tolerance rather than to an unrelated
    # hard-coded eigenvalue threshold.
    kappa_eff = max(cond_a, cond_m, cond_g, cond_c * cond_c)
    forward_risk = 64.0 * _FLOAT_EPS * kappa_eff
    forward_guard = float(decision_tol) / 1024.0
    forward_accuracy_unproven = (
        (not np.isfinite(forward_risk)) or forward_risk > forward_guard
    )

    escalated = bool(
        fast_failed
        or near_boundary
        or strongly_conditioned
        or routes_disagree
        or forward_accuracy_unproven
    )

    if escalated:
        if np.isfinite(kappa_eff) and kappa_eff > 1.0:
            condition_digits = int(np.ceil(np.log10(kappa_eff))) + 60
        elif not np.isfinite(kappa_eff):
            condition_digits = 180
        else:
            condition_digits = int(precision_digits)
        digits = max(int(precision_digits), condition_digits)
        mu, V = high_precision_metric_eigensystem(M_a, M_m, C, digits=digits)
        source = f"mpmath-exact-canonical-binary64-{digits}d"
    else:
        assert mu_fast is not None and V_fast is not None
        mu, V = mu_fast, V_fast
        source = "factorized-binary64-forward-guarded"

    eig_res, metric_res = _residuals(M_a, M_m, C, mu, V)
    return AdaptiveMetricEigensystem(
        mu=np.asarray(mu, dtype=float),
        eigenvectors_crystal=np.asarray(V, dtype=float),
        escalated=escalated,
        source=source,
        condition_parent=cond_a,
        condition_product=cond_m,
        condition_correspondence=cond_c,
        condition_pulled_metric=cond_g,
        route_disagreement=disagreement,
        distance_to_compatibility_boundary=distance,
        metric_orthonormality_residual=metric_res,
        eigen_equation_residual=eig_res,
    )


@lru_cache(maxsize=256)
def _high_precision_stretch_cached(
    key_a: bytes,
    key_m: bytes,
    key_c: bytes,
    digits: int,
) -> tuple[float, ...]:
    M_a = _array_from_key(key_a)
    M_m = _array_from_key(key_m)
    C = _array_from_key(key_c)
    with mp.workdps(int(digits)):
        A = _mp_matrix(M_a)
        M = _mp_matrix(M_m)
        X = _mp_matrix(C)
        G = X.T * M * X

        wa, Qa = mp.eigsy((A + A.T) / 2)
        if min(wa) <= 0:
            raise ArithmeticError("parent metric lost positive definiteness")
        W = Qa * mp.diag([1 / mp.sqrt(x) for x in wa]) * Qa.T
        H = W * G * W
        H = (H + H.T) / 2
        mu, Q = mp.eigsy(H)
        if min(mu) <= 0:
            raise ArithmeticError("normalized pulled metric lost positive definiteness")
        U = Q * mp.diag([mp.sqrt(x) for x in mu]) * Q.T
        out = tuple(float(U[i, j]) for i in range(3) for j in range(3))
        return out


def high_precision_symmetric_whitened_stretch(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    digits: int = 100,
) -> np.ndarray:
    """Principal stretch in the symmetric ``M_a^(1/2)`` orthonormal basis."""
    M_a, M_m, C = validate_metric_pencil_inputs(M_a, M_m, C)
    data = _high_precision_stretch_cached(
        _array_key(M_a), _array_key(M_m), _array_key(C), int(digits)
    )
    U = np.asarray(data, dtype=float).reshape(3, 3)
    return 0.5 * (U + U.T)


def adaptive_symmetric_whitened_stretch(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    decision_tol: float = 1.0e-8,
    precision_digits: int = 100,
) -> np.ndarray:
    """Return the same principal stretch, escalating arithmetic when required."""
    spec = adaptive_metric_eigensystem(
        M_a,
        M_m,
        C,
        decision_tol=decision_tol,
        precision_digits=precision_digits,
    )
    if spec.escalated:
        return high_precision_symmetric_whitened_stretch(
            M_a, M_m, C, digits=precision_digits
        )

    # Fast symmetric-whitening route, algebraically identical to the definition.
    M_a, M_m, C = validate_metric_pencil_inputs(M_a, M_m, C)
    w, Q = np.linalg.eigh(M_a)
    W = (Q * (1.0 / np.sqrt(w))) @ Q.T
    H = W @ pulled_metric_literal(M_m, C) @ W
    H = 0.5 * (H + H.T)
    mu, V = np.linalg.eigh(H)
    if np.min(mu) <= 0.0:
        raise ArithmeticError(f"normalized pulled metric is not positive definite: {mu}")
    U = (V * np.sqrt(mu)) @ V.T
    return 0.5 * (U + U.T)


@lru_cache(maxsize=256)
def _high_precision_smc_cached(
    key_a: bytes,
    key_m: bytes,
    key_c: bytes,
    digits: int,
) -> tuple[float, ...]:
    M_a = _array_from_key(key_a)
    M_m = _array_from_key(key_m)
    C = _array_from_key(key_c)
    with mp.workdps(int(digits)):
        A = _mp_matrix(M_a)
        M = _mp_matrix(M_m)
        X = _mp_matrix(C)
        G = X.T * M * X
        S = (A ** -1) - (G ** -1)
        return tuple(float(S[i, j]) for i in range(3) for j in range(3))


def high_precision_smc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    digits: int = 100,
) -> np.ndarray:
    """Evaluate ``M_a^-1 - (C.T M_m C)^-1`` at arbitrary precision."""
    M_a, M_m, C = validate_metric_pencil_inputs(M_a, M_m, C)
    data = _high_precision_smc_cached(
        _array_key(M_a), _array_key(M_m), _array_key(C), int(digits)
    )
    return np.asarray(data, dtype=float).reshape(3, 3)


def adaptive_smc(
    M_a: np.ndarray,
    M_m: np.ndarray,
    C: np.ndarray,
    *,
    precision_digits: int = 100,
) -> np.ndarray:
    r"""Evaluate Cayron SMC by the unchanged equation at high precision.

    The matrix is only 3x3, while SMC contains inverse operations and can suffer
    severe cancellation after hostile basis changes.  V3.1 therefore removes
    the unreliable binary64 acceptance shortcut entirely for SMC.  Production
    evaluates

        M_a^-1 - (C.T M_m C)^-1

    on the exact canonical binary64 inputs with arbitrary precision and only
    rounds once when returning the public ``float64`` matrix.  This is exactly
    the same scientific equation; only arithmetic precision changes.
    """
    M_a, M_m, C = validate_metric_pencil_inputs(M_a, M_m, C)
    kappa = max(
        float(np.linalg.cond(M_a)),
        float(np.linalg.cond(M_m)),
        float(np.linalg.cond(C)) ** 2,
    )
    if np.isfinite(kappa) and kappa > 1.0:
        condition_digits = int(np.ceil(np.log10(kappa))) + 60
    elif not np.isfinite(kappa):
        condition_digits = 180
    else:
        condition_digits = int(precision_digits)
    digits = max(int(precision_digits), condition_digits)
    return high_precision_smc(M_a, M_m, C, digits=digits)
