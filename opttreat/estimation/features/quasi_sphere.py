# random_features/quasi_sphere.py

from __future__ import annotations

from typing import Iterator, Callable
from itertools import count
from math import pi, sin, cos, sqrt, gamma
from typing import Tuple, Optional

import numpy as np


# -------------------------------------------------------
# Integral of sin^m for inverse CDF
# -------------------------------------------------------
def _int_sin_m(x: float, m: int) -> float:
    """
    ∫ sin^m(t) dt from 0 to x.
    Used for inverse-CDF mapping of hyperspherical coordinates.
    """
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * _int_sin_m(x, m - 2) - cos(x) * sin(x) ** (m - 1) / m


# -------------------------------------------------------
# Prime generator (infinite)
# -------------------------------------------------------
def _primes() -> Iterator[int]:
    yield from (2, 3, 5, 7)
    composites = {}
    ps = _primes()
    next(ps)
    p = next(ps)
    psq = p * p
    for i in count(9, 2):
        if i in composites:
            step = composites.pop(i)
        elif i < psq:
            yield i
            continue
        else:
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step


# -------------------------------------------------------
# Inverse CDF for monotone increasing function
# -------------------------------------------------------
def _inverse_increasing(
    func: Callable[[float], float],
    target: float,
    lower: float,
    upper: float,
    atol: float = 1e-10,
) -> float:
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid


# -------------------------------------------------------
# Quasi-random hypersphere point generator
# Returns: (dim, n_points) with columns as points
# -------------------------------------------------------
def quasi_random_hypersphere(dim: int, n_points: int) -> np.ndarray:
    """
    Low-discrepancy quasi-random points on S^{dim-1} ⊂ R^dim.

    Parameters
    ----------
    dim : int
        Ambient dimension (≥ 2).
    n_points : int
        Number of quasi-random points to generate.

    Returns
    -------
    points : np.ndarray of shape (dim, n_points)
        Each column is a unit vector on the sphere.
    """
    assert dim > 1 and n_points > 0

    # Points stored column-wise
    points = np.ones((dim, n_points), dtype=float)

    # First angle (theta_1) — evenly spaced circle
    for i in range(n_points):
        t = 2 * pi * i / n_points
        points[0, i] *= sin(t)
        points[1, i] *= cos(t)

    # Higher dimensions via deterministic Kronecker offsets
    for k, prime in zip(range(2, dim), _primes()):
        offset = sqrt(prime)
        mult = gamma(k / 2 + 0.5) / gamma(k / 2) / sqrt(pi)

        def dim_func(y):
            return mult * _int_sin_m(y, k - 1)

        for i in range(n_points):
            u = (i * offset) % 1.0
            theta = _inverse_increasing(dim_func, u, 0.0, pi)

            # Update 0..k-1 with sin, and k with cos
            for j in range(k):
                points[j, i] *= sin(theta)
            points[k, i] *= cos(theta)

    return points


# -------------------------------------------------------
# Convenience wrapper for RF ridge
# -------------------------------------------------------
def quasi_sphere_joint(
    d: int,
    K: int,
    joint: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Quasi-random sampling on the sphere for random features.

    Parameters
    ----------
    d : int
        Input dimension.
    K : int
        Number of features (neurons).
    joint : bool, default = True
        If True:
            sample (w, b) jointly from unit sphere in R^{d+1}.
            Returns W of shape (K, d) and b of shape (K,).
        If False:
            sample only weights w from unit sphere in R^d.
            Returns W of shape (K, d) and b = None.

    Returns
    -------
    W : np.ndarray, shape (K, d)
    b : np.ndarray of shape (K,) or None
    """
    if joint:
        # points on S^d in R^{d+1}
        pts = quasi_random_hypersphere(dim=d + 1, n_points=K)  # (d+1, K)
        W = pts[:d, :].T    # (K, d)
        b = pts[d, :]       # (K,)
        return W, b
    else:
        # points on S^{d-1} in R^d (weights only)
        pts = quasi_random_hypersphere(dim=d, n_points=K)      # (d, K)
        W = pts.T                                              # (K, d)
        return W, None
