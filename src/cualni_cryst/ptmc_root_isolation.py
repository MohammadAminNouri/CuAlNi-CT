"""Binary64-input, high-precision PTMC habit-fraction root isolation.

Only invoked for numerically near-tangent finite-domain rank-one PTMC pencils.
The oracle solves the *actual encoded binary64* F0+xD determinant, not a
quadratic assembled in float64. A numeric rank-one increment may contain
extremely small higher-degree coefficients as exact binary64 data; we retain
all coefficients, rather than quietly treating it as exact rank one.

This searches for real determinant roots with derivative partitioning, detects
critical-point tangencies, then checks the MIDDLE singular value independently.
An uncertifiable near-zero extremum raises rather than inventing a root.
"""
from __future__ import annotations

import math
import numpy as np
from mpmath import mp


class UnresolvedPTMCRoot(ArithmeticError):
    """The encoded mathematical root topology cannot be certified."""


def _mp_number(x):
    """Exactly convert a finite binary64 float to arbitrary precision."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("Non-finite PTMC coefficient")
    numerator, denominator = x.as_integer_ratio()
    return mp.mpf(numerator) / mp.mpf(denominator)


def _add(a, b, sign=1):
    out = [mp.mpf(0)] * max(len(a), len(b))
    for k, x in enumerate(a):
        out[k] += x
    for k, x in enumerate(b):
        out[k] += sign * x
    return out


def _mul(a, b):
    out = [mp.mpf(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _determinant_polynomial(F0, D, delta):
    """det((F0+xD)^T(F0+xD)-delta^2 I), degree <= 6."""
    A = [[_mp_number(x) for x in row] for row in np.asarray(F0, dtype=float)]
    B = [[_mp_number(x) for x in row] for row in np.asarray(D, dtype=float)]
    d = _mp_number(delta)
    G = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append([
                mp.fsum(A[k][i] * A[k][j] for k in range(3)) - (d*d if i == j else 0),
                mp.fsum(A[k][i]*B[k][j] + B[k][i]*A[k][j] for k in range(3)),
                mp.fsum(B[k][i]*B[k][j] for k in range(3)),
            ])
        G.append(row)
    co = [mp.mpf(0)] * 7
    for perm, sign in (( (0,1,2),1),((1,2,0),1),((2,0,1),1),
                       ((2,1,0),-1),((1,0,2),-1),((0,2,1),-1)):
        term = _mul(_mul(G[0][perm[0]], G[1][perm[1]]), G[2][perm[2]])
        co = _add(co, term, sign)
    return co


def _value(coeffs, x):
    y = mp.mpf(0)
    for c in reversed(coeffs):
        y = y*x + c
    return y


def _derivative(coeffs):
    return [mp.mpf(k) * coeffs[k] for k in range(1, len(coeffs))]


def _root_bisection(coeffs, left, right, *, count=380):
    a, b = left, right
    fa = _value(coeffs, a)
    fb = _value(coeffs, b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa*fb >= 0:
        raise UnresolvedPTMCRoot("No certified sign change for a candidate root")
    for _ in range(count):
        mid = (a+b)/2
        if mid == a or mid == b:
            break
        fm = _value(coeffs, mid)
        if fm == 0:
            return mid
        if fa*fm < 0:
            b = mid
        else:
            a,fa = mid,fm
    return (a+b)/2


def _roots_on_interval(coeffs, low, high, *, relative_zero):
    """Derivative-root partition; does not depend on sign changes at tangencies."""
    while len(coeffs)>1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
    if len(coeffs)<=1:
        if coeffs[0] == 0:
            raise UnresolvedPTMCRoot("Determinant is identically zero: continuum needs separate proof")
        return []
    if len(coeffs)==2:
        root = -coeffs[0]/coeffs[1]
        return [root] if low<=root<=high else []
    turning = _roots_on_interval(_derivative(coeffs), low, high, relative_zero=relative_zero)
    knots = [low] + turning + [high]
    roots=[]
    coefficient_norm=max(abs(c) for c in coeffs)
    critical_resolution = relative_zero * coefficient_norm
    values=[_value(coeffs, z) for z in knots]
    for i,(z,v) in enumerate(zip(knots,values)):
        if abs(v)<=critical_resolution:
            if v != 0 and abs(v)>critical_resolution/100:
                raise UnresolvedPTMCRoot("Determinant extremum too near zero to classify")
            # A certified repeated root requires residual well below MP roundoff
            # for the chosen working precision, not just a small decimal value.
            if v != 0:
                raise UnresolvedPTMCRoot("Potential tangent indistinguishable from arithmetic residue")
            roots.append(z)
        if i and values[i-1]*v<0:
            roots.append(_root_bisection(coeffs,knots[i-1],z))
    ordered=[]
    for x in sorted(roots):
        if not ordered or abs(x-ordered[-1])>mp.mpf('1e-55'):
            ordered.append(x)
    return ordered


def resolve_bounded_habit_roots(F0, rank_one_increment, *, dilatational_factor=1.0,
                                 domain=(0.0,1.0), digits=100):
    """Return high-precision checked fractions for encoded F0,D, finite domain.

    Cannot assert experimental compatibility: binary64 input is a numerical
    representation of measured quantities with separate uncertainty.
    Raises UnresolvedPTMCRoot on numerical non-identifiability/continuum.
    """
    F0=np.asarray(F0,dtype=float)
    D=np.asarray(rank_one_increment,dtype=float)
    if F0.shape!=(3,3) or D.shape!=(3,3) or not np.isfinite(F0).all() or not np.isfinite(D).all():
        raise ValueError("Finite 3x3 inputs required")
    low,high = map(float,domain)
    if not (math.isfinite(low) and math.isfinite(high) and low<=high):
        raise ValueError("Finite ordered domain required")
    if not (35<=digits<=200):
        raise ValueError("High precision digits must be 35..200")
    with mp.workdps(digits):
        coeffs=_determinant_polynomial(F0,D,dilatational_factor)
        # Precision safety margin of 30+ decimal digits at default 100 dps.
        roots=_roots_on_interval(coeffs,_mp_number(low),_mp_number(high),
                                 relative_zero=mp.mpf(10)**(-(digits-30)))
        A=mp.matrix([[_mp_number(x) for x in row] for row in F0])
        B=mp.matrix([[_mp_number(x) for x in row] for row in D])
        target=_mp_number(dilatational_factor)
        valid=[]
        for f in roots:
            M=A+f*B
            eigenvalues,_=mp.eigsy(M.T*M)
            middle=mp.sqrt(eigenvalues[1])
            if abs(middle-target)>mp.mpf('1e-28')*max(1,abs(target)):
                # det=0 due to a different singular value, not a habit root
                continue
            f64=float(f)
            if not (low<=f64<=high):
                raise UnresolvedPTMCRoot("Root rounding exceeds domain")
            if any(f64==previous for previous in valid):
                raise UnresolvedPTMCRoot("Two distinct roots round to one binary64 fraction")
            valid.append(f64)
        if len(valid) > 2:
            raise UnresolvedPTMCRoot(
                "Encoded increment permits more than two roots; cannot certify a rank-one quadratic"
            )
        return tuple(sorted(valid))
