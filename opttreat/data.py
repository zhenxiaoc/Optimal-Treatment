"""Data parsing and array-shape helpers for OptTreat."""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def ensure_2d_features(x: Any, *, name: str = "X") -> np.ndarray:
    """Return features as a float array with shape (n, d)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}.")
    return arr


def ensure_vector(values: Any, n: int | None = None, *, name: str = "values") -> np.ndarray:
    """
    Return a 1D float vector, broadcasting scalars to length n when supplied.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        if n is None:
            return arr.reshape(1)
        return np.full(n, float(arr), dtype=float)
    arr = arr.reshape(-1)
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} has length {arr.shape[0]}, expected {n}.")
    return arr


def normalize_treatment(d: Any, n: int) -> np.ndarray:
    """Normalize treatment indicators to a length-n vector."""
    return ensure_vector(d, n=n, name="d")


def trimmed_std(values: Any, *, ddof: int = 1) -> float:
    """Sample standard deviation after removing 1.5*IQR Tukey outliers.

    Points outside ``[q1 - 1.5*IQR, q3 + 1.5*IQR]`` are dropped before the SD is
    computed. This gives a scale for the fitted CATE ``h`` that is robust to the
    heavy tails it can have on real data, which is the preferred way to set the
    near-indifference band half-width ``eps = delta * scale(h)``. Falls back to
    the untrimmed SD when fewer than two points survive the trim, and returns
    ``0.0`` when there are fewer than two points to begin with.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 2:
        return 0.0
    q1, q3 = np.quantile(arr, 0.25), np.quantile(arr, 0.75)
    iqr = q3 - q1
    keep = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
    if keep.size < 2:
        keep = arr
    spread = float(np.std(keep, ddof=ddof))
    if spread <= 0.0:
        # Trimming removed all spread (a point mass dominates the inlier core,
        # e.g. many exactly-zero CATEs). Fall back to the untrimmed spread so the
        # scale never collapses to zero while the raw data still varies -- a band
        # of width zero, or a finite-difference step of size zero, would be ill
        # defined for downstream callers.
        spread = float(np.std(arr, ddof=ddof))
    return spread


def common_support_mask(X: Any, d: Any, *, strict: bool = True) -> np.ndarray:
    """
    Return rows whose covariates lie in the coordinatewise treated/control overlap.

    Chen, Chen, and Gao (2025) use strict inequalities against each treatment
    group's sample minima and maxima in the unknown-distribution simulations.
    """
    X_arr = ensure_2d_features(X, name="X")
    d_vec = normalize_treatment(d, X_arr.shape[0])

    X_t = X_arr[d_vec == 1]
    X_c = X_arr[d_vec == 0]
    if X_t.size == 0 or X_c.size == 0:
        raise ValueError("common_support_mask requires at least one treated and one control observation.")

    lower = np.maximum(X_t.min(axis=0), X_c.min(axis=0))
    upper = np.minimum(X_t.max(axis=0), X_c.max(axis=0))

    if strict:
        return np.all((X_arr > lower) & (X_arr < upper), axis=1)
    return np.all((X_arr >= lower) & (X_arr <= upper), axis=1)


def split_treated_control(data: Any) -> Dict[str, np.ndarray]:
    """
    Parse input data and split into treated and control groups.

    Supported formats
    -----------------
    1. Dict with pre-split groups:
        {
            "X_t": (n_t, d),
            "Y_t": (n_t,),
            "X_c": (n_c, d),
            "Y_c": (n_c,),
        }

    2. Dict with pooled X, Y, d:
        {
            "X": (n, d),
            "Y": (n,),
            "d": (n,),
        }

    3. pandas DataFrame with columns:
        - "y" : outcome
        - "d" : treatment indicator (1 = treated, 0 = control)
        - "X*" : covariates (e.g. X1, X2, ...)

    Returns
    -------
    dict with keys "X_t", "Y_t", "X_c", "Y_c".
    """
    # --- case 1: dict with pre-split groups -------------------------
    if isinstance(data, dict) and all(k in data for k in ["X_t", "Y_t", "X_c", "Y_c"]):
        X_t = ensure_2d_features(data["X_t"], name="X_t")
        Y_t = np.asarray(data["Y_t"]).ravel()
        X_c = ensure_2d_features(data["X_c"], name="X_c")
        Y_c = np.asarray(data["Y_c"]).ravel()
        return {"X_t": X_t, "Y_t": Y_t, "X_c": X_c, "Y_c": Y_c}

    # --- case 2: dict with pooled X, Y, d --------------------------
    if isinstance(data, dict) and all(k in data for k in ["X", "Y", "d"]):
        X = ensure_2d_features(data["X"], name="X")
        Y = np.asarray(data["Y"]).ravel()
        d = normalize_treatment(data["d"], X.shape[0])

        mask_t = (d == 1)
        mask_c = (d == 0)

        return {
            "X_t": X[mask_t],
            "Y_t": Y[mask_t],
            "X_c": X[mask_c],
            "Y_c": Y[mask_c],
        }

    # --- case 3: pandas DataFrame ----------------------------------
    if isinstance(data, pd.DataFrame):
        df_t = data[data["d"] == 1]
        df_c = data[data["d"] == 0]

        Y_t = df_t["y"].to_numpy().ravel()
        X_t = df_t.filter(like="X").to_numpy()

        Y_c = df_c["y"].to_numpy().ravel()
        X_c = df_c.filter(like="X").to_numpy()

        return {"X_t": X_t, "Y_t": Y_t, "X_c": X_c, "Y_c": Y_c}

    # --- otherwise: unsupported format ------------------------------
    raise TypeError(
        "split_treated_control expects either:\n"
        "  - dict with keys 'X_t','Y_t','X_c','Y_c', or\n"
        "  - dict with keys 'X','Y','d', or\n"
        "  - pandas DataFrame with columns 'y','d','X*'."
    )
