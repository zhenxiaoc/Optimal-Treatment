"""Sobol integration grids and boundary-band helpers.

Shared by the target parameters (welfare/value) and the sieve variance
estimator so they build the integration grid and the near-boundary band the
same way.
"""

from __future__ import annotations

from typing import Callable
import warnings

import numpy as np
from scipy.stats.qmc import Sobol

from opttreat.data import ensure_2d_features, ensure_vector, trimmed_std


def boundary_band(h_vals: np.ndarray, delta: float, options: dict) -> tuple[np.ndarray, float]:
    """
    Return the near-zero boundary mask and fixed bandwidth for a function h.

    Uses options["loo_eps"] if provided; otherwise sets eps from delta times the
    outlier-trimmed spread of h_vals (:func:`opttreat.data.trimmed_std`, which
    drops 1.5*IQR Tukey outliers before taking the SD so heavy CATE tails do not
    inflate the band). The returned mask selects exactly |h_vals| < eps.
    """
    if "loo_eps" in options:
        eps = float(options["loo_eps"])
    else:
        eps = float(delta) * max(trimmed_std(h_vals), 1e-12)
    eps = max(eps, 1e-12)
    mask = np.abs(h_vals) < eps
    return mask, eps


def sobol_grid(
    dim: int,
    n: int,
    transform: Callable[[np.ndarray], np.ndarray],
    sobol_seed: int,
    scramble: bool,
) -> np.ndarray:
    """
    Generate n Sobol points in [0, 1]^dim and map them to the target support.
    """
    engine = Sobol(d=dim, scramble=scramble, seed=sobol_seed if scramble else None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The balance properties of Sobol")
        U = engine.random(n)
    return ensure_2d_features(transform(U), name="X_int")


def sobol_grid_with_boundary_points(
    h_func: Callable[[np.ndarray], np.ndarray],
    options: dict,
    *,
    delta: float,
    dim: int,
    n: int,
    transform: Callable[[np.ndarray], np.ndarray],
    sobol_seed: int,
    scramble: bool,
    min_band_option: str = "loo_min_band",
    max_sobol_option: str = "loo_max_sobol",
    expand_factor_option: str = "loo_sobol_expand_factor",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a Sobol grid with enough rows near the boundary h_func(X)=0.

    Keeps eps fixed, counts Sobol rows satisfying |h_func(X)| < eps, and
    increases the Sobol grid size until the boundary band is populated or the
    Sobol cap is reached.
    """
    min_band = int(options.get(min_band_option, options.get("loo_min_band", min(500, n))))
    max_sobol = int(options.get(max_sobol_option, options.get("loo_max_sobol", max(n, 10 * n))))
    max_sobol = max(max_sobol, n)
    expand_factor = max(
        float(options.get(expand_factor_option, options.get("loo_sobol_expand_factor", 2.0))),
        1.01,
    )

    while True:
        X = sobol_grid(dim, n, transform, sobol_seed, scramble)
        h_vals = ensure_vector(h_func(X), n=X.shape[0], name="h_boundary(X)")
        mask, _ = boundary_band(h_vals, delta, options)
        if int(mask.sum()) >= min_band or n >= max_sobol:
            return X, h_vals
        n = min(max_sobol, max(n + 1, int(np.ceil(n * expand_factor))))
