# random_features/activations.py

from __future__ import annotations
import numpy as np
from typing import Any, Callable


def get_activation(act_opt: Any) -> Callable[[np.ndarray], np.ndarray]:
    """
    Convert an activation option into an elementwise callable.

    Supported string activations are "cos"/"cosine", "relu", "sigmoid",
    "tanh", and "exp".  The smooth activations also have analytic first and
    second derivatives in get_activation_derivatives().
    """
    if callable(act_opt):
        return act_opt

    if act_opt is None or act_opt in ("cos", "cosine"):
        return np.cos

    if act_opt == "relu":
        return lambda z: np.maximum(z, 0.0)
    
    if act_opt == "sigmoid":
        return lambda z: 1 / (1 + np.exp(-z))
    
    if act_opt == "tanh":
        return lambda z: np.tanh(z)

    if act_opt == "exp":
        return lambda z: np.exp(z)

    raise ValueError(f"Unsupported activation option: {act_opt!r}")


def get_activation_derivatives(
    act_opt: Any,
) -> tuple[Callable[[np.ndarray], np.ndarray] | None, Callable[[np.ndarray], np.ndarray] | None]:
    """Return first and second elementwise derivatives for supported activations."""
    if callable(act_opt) and not isinstance(act_opt, str):
        first = getattr(act_opt, "derivative", None)
        second = getattr(act_opt, "second_derivative", None)
        return (first if callable(first) else None, second if callable(second) else None)

    if act_opt is None or act_opt in ("cos", "cosine"):
        return (lambda z: -np.sin(z), lambda z: -np.cos(z))

    if act_opt == "relu":
        return (lambda z: (z > 0.0).astype(float), lambda z: np.zeros_like(z, dtype=float))

    if act_opt == "sigmoid":
        def first(z: np.ndarray) -> np.ndarray:
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)

        def second(z: np.ndarray) -> np.ndarray:
            s = 1.0 / (1.0 + np.exp(-z))
            sp = s * (1.0 - s)
            return sp * (1.0 - 2.0 * s)

        return first, second

    if act_opt == "tanh":
        def first(z: np.ndarray) -> np.ndarray:
            t = np.tanh(z)
            return 1.0 - t**2

        def second(z: np.ndarray) -> np.ndarray:
            t = np.tanh(z)
            return -2.0 * t * (1.0 - t**2)

        return first, second

    if act_opt == "exp":
        return (lambda z: np.exp(z), lambda z: np.exp(z))

    raise ValueError(f"Unsupported activation option: {act_opt!r}")
