"""Derivative helpers for linear-index random feature maps."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .activations import get_activation_derivatives


def attach_linear_feature_derivatives(
    feature_map: Callable[[np.ndarray], np.ndarray],
    W: np.ndarray,
    b: np.ndarray,
    activation: Any,
    *,
    gamma: float = 1.0,
    include_intercept: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Attach basis derivative methods to a map x -> [1,] act(gamma*(Wx+b))."""
    first, second = get_activation_derivatives(activation)
    if first is None or second is None:
        return feature_map

    W_arr = np.asarray(W, dtype=float)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    if W_arr.ndim != 2:
        raise ValueError("W must be a 2D array.")
    if b_arr.shape[0] != W_arr.shape[0]:
        raise ValueError("b must have one entry per row of W.")

    dim = W_arr.shape[1]
    gamma_float = float(gamma)
    offset = 1 if include_intercept else 0

    def _ensure_2d(x: np.ndarray) -> np.ndarray:
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError(f"feature_map expects 1D or 2D input, got shape {X.shape}.")
        if X.shape[1] != dim:
            raise ValueError(f"feature_map expected {dim} columns, got shape {X.shape}.")
        return X

    def _z(x: np.ndarray) -> np.ndarray:
        X = _ensure_2d(x)
        return gamma_float * (X @ W_arr.T + b_arr[None, :])

    def _pad(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not include_intercept:
            return values
        return np.hstack([np.zeros((values.shape[0], 1), dtype=float), values])

    def derivative(x: np.ndarray, axis: int, order: int = 1) -> np.ndarray:
        axis_int = int(axis)
        if axis_int < 0 or axis_int >= dim:
            raise ValueError(f"axis must be in [0, {dim}), got {axis}.")
        order_int = int(order)
        if order_int == 0:
            return np.asarray(feature_map(x), dtype=float)
        z = _z(x)
        if order_int == 1:
            values = first(z) * gamma_float * W_arr[:, axis_int][None, :]
        elif order_int == 2:
            values = second(z) * (gamma_float * W_arr[:, axis_int][None, :]) ** 2
        else:
            raise ValueError("Only derivative orders 0, 1, and 2 are supported.")
        return _pad(values)

    def gradient(x: np.ndarray) -> np.ndarray:
        z = _z(x)
        grad = first(z)[:, :, None] * gamma_float * W_arr[None, :, :]
        if not include_intercept:
            return grad
        return np.concatenate([np.zeros((grad.shape[0], 1, dim), dtype=float), grad], axis=1)

    def laplacian_values(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        beta_arr = np.asarray(beta, dtype=float).reshape(-1)
        beta_features = beta_arr[offset:]
        if beta_features.shape[0] != W_arr.shape[0]:
            raise ValueError("beta length does not match the feature map.")
        z = _z(x)
        w_norm2 = np.sum(W_arr * W_arr, axis=1)
        return np.sum(second(z) * beta_features[None, :] * (gamma_float**2) * w_norm2[None, :], axis=1)

    def hessian_quadratic_values(x: np.ndarray, beta: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        X = _ensure_2d(x)
        V = np.asarray(vectors, dtype=float)
        if V.shape != X.shape:
            raise ValueError(f"vectors must have shape {X.shape}, got {V.shape}.")
        beta_arr = np.asarray(beta, dtype=float).reshape(-1)
        beta_features = beta_arr[offset:]
        if beta_features.shape[0] != W_arr.shape[0]:
            raise ValueError("beta length does not match the feature map.")
        z = _z(X)
        dots = V @ W_arr.T
        return np.sum(second(z) * beta_features[None, :] * (gamma_float**2) * dots**2, axis=1)

    feature_map.derivative = derivative  # type: ignore[attr-defined]
    feature_map.gradient = gradient  # type: ignore[attr-defined]
    feature_map.laplacian_values = laplacian_values  # type: ignore[attr-defined]
    feature_map.hessian_quadratic_values = hessian_quadratic_values  # type: ignore[attr-defined]
    feature_map.domain_dim = dim  # type: ignore[attr-defined]
    feature_map.include_intercept = include_intercept  # type: ignore[attr-defined]
    return feature_map
