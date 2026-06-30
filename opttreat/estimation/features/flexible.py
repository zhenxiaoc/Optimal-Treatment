# Random-feature sampling helpers for rfg_type="flexible".

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


# ----------------------------------------------------------------------
# Requirements for numpy RNG distributions
# ----------------------------------------------------------------------
DISTRIBUTION_REQUIREMENTS = {
    "beta": 2,
    "binomial": 2,
    "chisquare": 1,
    "dirichlet": 1,
    "exponential": 1,
    "f": 2,
    "gamma": 1,
    "geometric": 1,
    "gumbel": 2,
    "hypergeometric": 3,
    "laplace": 2,
    "logistic": 2,
    "lognormal": 2,
    "logseries": 1,
    "multinomial": 2,
    "multivariate_normal": 2,
    "negative_binomial": 2,
    "noncentral_chisquare": 2,
    "noncentral_f": 3,
    "normal": 2,
    "pareto": 1,
    "poisson": 1,
    "power": 1,
    "rayleigh": 1,
    "standard_cauchy": 0,
    "standard_exponential": 0,
    "standard_gamma": 1,
    "standard_normal": 0,
    "standard_t": 1,
    "triangular": 3,
    "uniform": 2,
    "vonmises": 2,
    "wald": 2,
    "weibull": 1,
    "zipf": 1,
}


# ----------------------------------------------------------------------
# Distribution check
# ----------------------------------------------------------------------
def check_distribution(dist: str, params: List) -> None:
    if dist not in DISTRIBUTION_REQUIREMENTS:
        raise ValueError(f"Unknown distribution {dist!r}.")
    req = DISTRIBUTION_REQUIREMENTS[dist]
    if len(params) != req:
        raise ValueError(f"Distribution {dist!r} requires {req} parameters, got {len(params)}.")


# ----------------------------------------------------------------------
# Sampling wrapper
# ----------------------------------------------------------------------
def _sample_rng(
    dist: str,
    params: List,
    size: Tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    check_distribution(dist, params)

    # Try rng.<dist>
    if hasattr(rng, dist):
        sampler = getattr(rng, dist)
    # Try rng.standard_<dist>
    elif hasattr(rng, f"standard_{dist}"):
        sampler = getattr(rng, f"standard_{dist}")
    else:
        raise ValueError(f"Numpy Generator has no sampler for distribution {dist!r}.")

    return sampler(*params, size=size)


# ----------------------------------------------------------------------
# Main function: random_sample_joint
# ----------------------------------------------------------------------
def random_sample_joint(
    d: int,
    K: int,
    rng: np.random.Generator,
    weight_distribution: str,
    weight_params: List,
    bias_distribution: Optional[str] = None,
    bias_params: Optional[List] = None,
    joint: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random sampling of (W, b) for random-feature neurons.

    Semantics:
    ----------
    • joint = True:
        Draw weights from weight_distribution(*weight_params)
        Draw biases from bias_distribution(*bias_params)
    • joint = False:
        Draw only weights
        Set bias = 0  (ignore bias distribution)

    Parameters
    ----------
    d : int
        Dimension of weights.
    K : int
        Number of random features (neurons).
    rng : np.random.Generator
        RNG used for reproducibility.
    weight_distribution : str
        Name of numpy RNG distribution for weights.
    weight_params : list
        Parameters for the weight distribution.
    bias_distribution : str or None
        Name of numpy RNG bias distribution (ignored if joint=False).
    bias_params : list or None
        Parameters for bias distribution (ignored if joint=False).
    joint : bool
        True  -> sample both W and b from given distributions.
        False -> sample W only, set b = 0.

    Returns
    -------
    W : np.ndarray, shape (K, d)
        Weight matrix.
    b : np.ndarray, shape (K,)
        Bias vector.
    """
    # ------------------------------
    # Step 1: sample weights
    # ------------------------------
    W = _sample_rng(
        dist=weight_distribution,
        params=weight_params,
        size=(K, d),
        rng=rng,
    )

    # ------------------------------
    # Step 2: sample bias depending on joint flag
    # ------------------------------
    if not joint:
        # Ignore bias distribution entirely
        b = np.zeros(K)
        return W, b

    # joint = True → must sample bias
    if bias_distribution is None or bias_params is None:
        raise ValueError("joint=True requires both bias_distribution and bias_params.")

    b = _sample_rng(
        dist=bias_distribution,
        params=bias_params,
        size=(K,),
        rng=rng,
    )

    return W, b
