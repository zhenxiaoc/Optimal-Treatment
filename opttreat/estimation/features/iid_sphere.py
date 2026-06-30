# random_features/sphere_iid.py

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def iid_sphere_joint(
    d: int,
    K: int,
    rng: np.random.Generator,
    joint: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    IID sampling on the sphere for random features.

    joint=True:
        (w,b) on unit sphere in R^{d+1}.
    joint=False:
        w on unit sphere in R^d, no bias.
    d: 
        dimension of regressors
    K:
        targeted dimension of random features  
    """
    if joint:
        Z = rng.normal(size=(K, d + 1))   # R^{d+1}
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = Z / norms                     # rows on S^d
        W = Z[:, :d]                      # (K, d)
        b = Z[:, d]                       # (K,)
        return W, b
    else:
        Z = rng.normal(size=(K, d))       # R^d
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        W = Z / norms                     # rows on S^{d-1}
        return W, None