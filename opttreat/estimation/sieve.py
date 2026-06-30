"""B-spline sieve first-stage estimator."""

from __future__ import annotations

from typing import Any, Dict, Callable

import numpy as np
from sklearn.linear_model import Ridge

from .base import Estimator, EstimatorOutput
from opttreat.estimation.splines.spline_factory import build_spline_basis_from_options


class SieveEstimator(Estimator):
    """
    Linear sieve estimator using B-spline features.

    Fits separate models for treated and control:
        mu_t(x) = E[Y | X=x, D=1]
        mu_c(x) = E[Y | X=x, D=0]
    and returns h_hat(x) = mu_t(x) - mu_c(x).

    Options (in self.options)
    -------------------------
    - "share_features": bool, default True
        If True, build one common spline basis from all X.
        If False, allow group-specific J_x_segments, etc.
    - "J_x_degree": int or array-like
    - "J_x_segments": int or array-like
    - "knots": "uniform" or "quantiles"
    - "basis": "additive", "tensor", or "glp"
    - "X_min", "X_max": optional bounds (see spline_factory)
    - "solver": "ridge" or "pinv", default "ridge"
        "pinv" matches the generalized-inverse OLS formula used in the
        Chen, Chen, and Gao (2025) R simulation code.
    - "alpha": float, ridge penalty used only when solver="ridge"
    - "trim_common_support": bool, default True
        If True, restrict to the coordinatewise intersection of treated
        and control supports before fitting.
    """

    def __init__(self, name: str = "sieve", options: Dict[str, Any] | None = None):
        super().__init__(name=name, options=options)
        self.h_hat: Callable[[np.ndarray], np.ndarray] | None = None
        self.mu_hat_t: Callable[[np.ndarray], np.ndarray] | None = None
        self.mu_hat_c: Callable[[np.ndarray], np.ndarray] | None = None

        self.e_t_: np.ndarray | None = None
        self.e_c_: np.ndarray | None = None
        self.Psi_t_: np.ndarray | None = None
        self.Psi_c_: np.ndarray | None = None
        self.X_all_: np.ndarray | None = None

        self.feature_map_t: Callable[[np.ndarray], np.ndarray] | None = None
        self.feature_map_c: Callable[[np.ndarray], np.ndarray] | None = None
        self.alpha_: float | None = None
        self.beta_t_: np.ndarray | None = None
        self.beta_c_: np.ndarray | None = None
        self.solver_: str | None = None

    # ------------------------------------------------------------------
    # Core API: fit
    # ------------------------------------------------------------------
    def fit(self, parsed_data: Dict[str, np.ndarray]) -> EstimatorOutput:
        # Unpack and standardize shapes
        X_t = np.asarray(parsed_data["X_t"], dtype=float)
        Y_t = np.asarray(parsed_data["Y_t"], dtype=float).ravel()
        X_c = np.asarray(parsed_data["X_c"], dtype=float)
        Y_c = np.asarray(parsed_data["Y_c"], dtype=float).ravel()

        if X_t.ndim == 1:
            X_t = X_t.reshape(-1, 1)
        if X_c.ndim == 1:
            X_c = X_c.reshape(-1, 1)

        # -----------------------------------------------------------
        # Optional: trim to common support
        # -----------------------------------------------------------
        # trim_common = bool(self.options.get("trim_common_support", True))
        # if trim_common:
        #     min_t = X_t.min(axis=0)   # shape (d,)
        #     max_t = X_t.max(axis=0)

        #     min_c = X_c.min(axis=0)
        #     max_c = X_c.max(axis=0)

        #     lower = np.maximum(min_t, min_c)
        #     upper = np.minimum(max_t, max_c)

        #     mask_t = np.all((X_t >= lower) & (X_t <= upper), axis=1)
        #     mask_c = np.all((X_c >= lower) & (X_c <= upper), axis=1)

        #     X_t = X_t[mask_t]
        #     Y_t = Y_t[mask_t]
        #     X_c = X_c[mask_c]
        #     Y_c = Y_c[mask_c]

        share_features = bool(self.options.get("share_features", True))
        solver = str(self.options.get("solver", "ridge")).lower()
        if solver not in ("ridge", "pinv"):
            raise ValueError("SieveEstimator.options['solver'] must be 'ridge' or 'pinv'.")

        alpha = 0.0 if solver == "pinv" else float(self.options.get("alpha", 1e-3))
        self.alpha_ = alpha
        self.solver_ = solver

        # -----------------------------------------------------------
        # Build feature maps
        # -----------------------------------------------------------
        if share_features:
            if "J_x_segments" not in self.options and (
                "J_x_segments_t" not in self.options or "J_x_segments_c" not in self.options
            ):
                raise ValueError(
                    "When share_features=True, specify 'J_x_segments' in options."
                )

            # Use all X (after trimming) to define knots and domain
            X_all = np.vstack([X_t, X_c])
            feature_map = build_spline_basis_from_options(self.options, X_all)

            feature_map_t = feature_map
            feature_map_c = feature_map

        else:
            # group-specific K_t, K_c, falling back to J_x_segments if needed
            if "J_x_segments" not in self.options and (
                "J_x_segments_t" not in self.options or "J_x_segments_c" not in self.options
            ):
                raise ValueError(
                    "When share_features=False, specify either 'J_x_segments' "
                    "or both 'J_x_segments_t' and 'J_x_segments_c' in options."
                )

            K_t = self.options.get("J_x_segments_t", self.options.get("J_x_segments"))
            K_c = self.options.get("J_x_segments_c", self.options.get("J_x_segments"))

            # Treat options separately for t and c
            opts_t = dict(self.options)
            opts_c = dict(self.options)

            opts_t["J_x_segments"] = int(K_t)
            opts_c["J_x_segments"] = int(K_c)

            feature_map_t = build_spline_basis_from_options(opts_t, X_t)
            feature_map_c = build_spline_basis_from_options(opts_c, X_c)

            X_all = np.vstack([X_t, X_c])

        Psi_t = feature_map_t(X_t)  # (n_t, K_t)
        Psi_c = feature_map_c(X_c)  # (n_c, K_c)

        # -----------------------------------------------------------
        # Fit separate linear models in the spline basis.
        # -----------------------------------------------------------
        pinv_rcond = float(self.options.get("pinv_rcond", np.sqrt(np.finfo(float).eps)))
        if solver == "pinv":
            beta_t = np.linalg.pinv(Psi_t.T @ Psi_t, rcond=pinv_rcond) @ Psi_t.T @ Y_t
            beta_c = np.linalg.pinv(Psi_c.T @ Psi_c, rcond=pinv_rcond) @ Psi_c.T @ Y_c
        else:
            beta_t = Ridge(alpha=alpha, fit_intercept=False).fit(Psi_t, Y_t).coef_.reshape(-1)
            beta_c = Ridge(alpha=alpha, fit_intercept=False).fit(Psi_c, Y_c).coef_.reshape(-1)

        fitted_t = Psi_t @ beta_t
        fitted_c = Psi_c @ beta_c

        # Store internals
        self.e_t_ = Y_t - fitted_t
        self.e_c_ = Y_c - fitted_c
        self.beta_t_ = beta_t
        self.beta_c_ = beta_c

        self.Psi_t_ = Psi_t
        self.Psi_c_ = Psi_c

        self.feature_map_t = feature_map_t
        self.feature_map_c = feature_map_c

        self.X_all_ = X_all

        # -----------------------------------------------------------
        # Define mu_hat_t, mu_hat_c, and h_hat
        # -----------------------------------------------------------
        def mu_hat_t(x: np.ndarray):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            Psi_x = feature_map_t(x)
            Psi_x = np.asarray(Psi_x, dtype=float)
            y_pred = Psi_x @ beta_t
            return float(y_pred[0]) if y_pred.shape[0] == 1 else y_pred

        def mu_hat_c(x: np.ndarray):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            Psi_x = feature_map_c(x)
            Psi_x = np.asarray(Psi_x, dtype=float)
            y_pred = Psi_x @ beta_c
            return float(y_pred[0]) if y_pred.shape[0] == 1 else y_pred

        def h_hat(x: np.ndarray):
            return mu_hat_t(x) - mu_hat_c(x)

        self.mu_hat_t = mu_hat_t
        self.mu_hat_c = mu_hat_c
        self.h_hat = h_hat

        return self.get_output()
