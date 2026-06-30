"""Random-feature map factory."""

from __future__ import annotations

from typing import Any, Dict, Callable
import numpy as np

from .iid_sphere import iid_sphere_joint          # (W, b) or (W, None)
from .quasi_sphere import quasi_sphere_joint      # (W, b) or (W, None)
from .activations import get_activation
from .linear_derivatives import attach_linear_feature_derivatives
from .flexible import random_sample_joint


def build_feature_map_from_options(
    options: Dict[str, Any],
    X_sample: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Construct a random feature map ψ: R^d → R^K based on `options`.

    There are two usage modes:

    1. User-provided feature_map:
        options["feature_map"] = callable X -> Psi(X)
        In this case we simply return that callable.

    2. Random feature generators via rfg_type:
        Required keys:
            - "rfg_type": {"iid_sphere", "quasi_sphere", "flexible"}
            - "n_features": int, number of random features K

        Optional / type-specific keys:
            - "activation": str or callable, default "cos"
            - "joint": bool, default True
            - "random_state": int or None

        For rfg_type == "iid_sphere" or "quasi_sphere":
            Uses iid_sphere_joint / quasi_sphere_joint.

        For rfg_type == "flexible":
            - "weight_distribution": str
            - "weight_params": list
            - If joint=True:
                "bias_distribution": str
                "bias_params": list

    The returned fmap(X_new) always returns an array of shape (n, K), where
    n is the number of rows of X_new. If X_new is a single x with shape (d,),
    we internally reshape it to (1, d).
    """

    # ------------------------------------------------------------------
    # 1. User-provided feature_map overrides everything
    # ------------------------------------------------------------------
    feature_map = options.get("feature_map")
    if feature_map is not None:
        if not callable(feature_map):
            raise ValueError("feature_map must be callable.")
        return feature_map

    # ------------------------------------------------------------------
    # 2. Retrieve core settings
    # ------------------------------------------------------------------
    rfg_type = options.get("rfg_type")
    if rfg_type is None:
        raise ValueError("Must specify either 'feature_map' or 'rfg_type' in options.")

    if "n_features" not in options:
        raise ValueError("When using rfg_type, 'n_features' must be specified in options.")

    K = int(options["n_features"])
    X_sample = np.asarray(X_sample)
    if X_sample.ndim != 2:
        raise ValueError("X_sample must be a 2D array (n_sample, d).")
    d = X_sample.shape[1]

    activation_opt = options.get("activation", "sigmoid")
    activation = get_activation(activation_opt)
    joint = bool(options.get("joint", True))
    seed = options.get("random_state", None)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Helper: ensure X_new is 2D (n, d)
    # ------------------------------------------------------------------
    def _ensure_2d(X_new: np.ndarray) -> np.ndarray:
        X_new = np.asarray(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        elif X_new.ndim != 2:
            raise ValueError("X_new must be 1D (d,) or 2D (n, d).")
        if X_new.shape[1] != d:
            raise ValueError(f"X_new has {X_new.shape[1]} columns, expected {d}.")
        return X_new

    # ------------------------------------------------------------------
    # 3. Sphere-based samplers
    # ------------------------------------------------------------------
    if rfg_type == "iid_sphere":
        # W: (K, d)
        # b: (K,)
        W, b = iid_sphere_joint(d=d, K=K, rng=rng, joint=joint)
        if b is None:
            b = np.zeros(K)

        def fmap(X_new: np.ndarray) -> np.ndarray:
            X_new_2d = _ensure_2d(X_new)              # (n, d)
            return activation(X_new_2d @ W.T + b)     # (n, K)

        return attach_linear_feature_derivatives(fmap, W, b, activation_opt)

    elif rfg_type == "quasi_sphere":
        # quasi_sphere_joint: (W, b) if joint=True, else (W, None)
        W, b = quasi_sphere_joint(d=d, K=K, joint=joint)
        if b is None:
            b = np.zeros(K)

        def fmap(X_new: np.ndarray) -> np.ndarray:
            X_new_2d = _ensure_2d(X_new)
            return activation(X_new_2d @ W.T + b)

        return attach_linear_feature_derivatives(fmap, W, b, activation_opt)

    elif rfg_type == "flexible":
        weight_dist = options.get("weight_distribution")
        weight_params = options.get("weight_params", [])
        bias_dist = options.get("bias_distribution")
        bias_params = options.get("bias_params", [])

        if weight_dist is None:
            raise ValueError("rfg_type='flexible' requires 'weight_distribution'.")
        if joint and bias_dist is None:
            raise ValueError("rfg_type='flexible' with joint=True requires 'bias_distribution'.")

        W, b = random_sample_joint(
            d=d,
            K=K,
            rng=rng,
            weight_distribution=weight_dist,
            weight_params=weight_params,
            bias_distribution=bias_dist,
            bias_params=bias_params,
            joint=joint,
        )
        if b is None:
            b = np.zeros(K)

        def fmap(X_new: np.ndarray) -> np.ndarray:
            X_new_2d = _ensure_2d(X_new)
            return activation(X_new_2d @ W.T + b)

        return attach_linear_feature_derivatives(fmap, W, b, activation_opt)

    # ------------------------------------------------------------------
    # 5. Unknown rfg_type
    # ------------------------------------------------------------------
    else:
        raise ValueError(f"Unknown rfg_type: {rfg_type!r}")
