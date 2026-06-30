"""Random-feature linear first-stage estimator."""

from __future__ import annotations

from typing import Any, Dict, Callable

import numpy as np
from sklearn.linear_model import Ridge

from .base import Estimator, EstimatorOutput
from opttreat.estimation.features.feature_factory import build_feature_map_from_options


class RFRidgeEstimator(Estimator):
    """
    Random-feature linear estimator of h(x) = mu_t(x) - mu_c(x).

    - Fits separate ridge or pinv regressions on random features for treated and control.
    - Can share one common feature map or use group-specific random features.

    Attributes:
    name:
    options:
        - "rfg_type",       (required)     | "iid_sphere", "quasi_sphere", or "flexible" 
        - "activation",     (required)     | "relu", "sigmoid", "exp", "tanh", "cos"/"cosine" (default)
        - "share_features", (required)     | True (default) or False
        - "n_features",
        - "n_features_t",      
        - "n_features_c",      
        - "random_state", 
        - "alpha",
        - "solver": "ridge" (default) or "pinv",
    """

    def __init__(self, name: str = "rf_ridge", options: Dict[str, Any] | None = None):
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
        self.solver_: str | None = None
        self.beta_t_: np.ndarray | None = None
        self.beta_c_: np.ndarray | None = None
        self.diagnostics_: dict[str, float | int] = {}

    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------
    def fit(self, parsed_data: Dict[str, np.ndarray]) -> EstimatorOutput:
        """
        Fit RF ridge separately on treated and control groups.

        Inputs
        ------
        parsed_data : dict
            Already-parsed data with keys:
                - "X_t": (n_t, d) or (n_t,)
                - "Y_t": (n_t,)
                - "X_c": (n_c, d) or (n_c,)
                - "Y_c": (n_c,)

        Outputs
        -------
        dict
            Common estimator output consumed by parameters and variance estimators.
        """
        # -----------------------------------------------------------
        # Unpack parsed data and standardize shapes
        # -----------------------------------------------------------
        X_t = np.asarray(parsed_data["X_t"], dtype=float)
        Y_t = np.asarray(parsed_data["Y_t"], dtype=float).ravel()
        X_c = np.asarray(parsed_data["X_c"], dtype=float)
        Y_c = np.asarray(parsed_data["Y_c"], dtype=float).ravel()

        if X_t.ndim == 1:
            X_t = X_t.reshape(-1, 1)
        if X_c.ndim == 1:
            X_c = X_c.reshape(-1, 1)

        share_features = bool(self.options.get("share_features", True))
        solver = str(self.options.get("solver", "ridge")).lower()
        if solver not in ("ridge", "pinv"):
            raise ValueError("RFRidgeEstimator.options['solver'] must be 'ridge' or 'pinv'.")

        alpha = 0.0 if solver == "pinv" else float(self.options.get("alpha", 1e-3))
        self.alpha_ = alpha
        self.solver_ = solver

        # -----------------------------------------------------------
        # Build feature maps
        # -----------------------------------------------------------
        if share_features:
            # One joint feature map using all X for scale/randomness
            X_all = np.vstack([X_t, X_c])
            feature_map = build_feature_map_from_options(self.options, X_all)
            self.diagnostics_ = {}
            feature_map_t = feature_map
            feature_map_c = feature_map

        else:
            # Independent random features for treated and control,
            # possibly with different numbers of features.
            base_seed = self.options.get("random_state", None)
            if base_seed is None:
                base_seed = np.random.randint(0, 2**31 - 1)
            seed_t = int(base_seed)
            seed_c = int(base_seed) + 1

            # group-specific K_t, K_c, falling back to n_features if needed
            if "n_features" not in self.options and (
                "n_features_t" not in self.options or "n_features_c" not in self.options
            ):
                raise ValueError(
                    "When share_features=False, specify either 'n_features' "
                    "or both 'n_features_t' and 'n_features_c' in options."
                )

            K_t = self.options.get("n_features_t", self.options.get("n_features"))
            K_c = self.options.get("n_features_c", self.options.get("n_features"))

            # Treat options separately for t and c
            opts_t = dict(self.options)
            opts_c = dict(self.options)

            opts_t["n_features"] = int(K_t)
            opts_c["n_features"] = int(K_c)

            opts_t["random_state"] = seed_t
            opts_c["random_state"] = seed_c

            feature_map_t = build_feature_map_from_options(opts_t, X_t)
            feature_map_c = build_feature_map_from_options(opts_c, X_c)

        Psi_t = feature_map_t(X_t)  # (n_t, K_t)
        Psi_c = feature_map_c(X_c)  # (n_c, K_c)

        # -----------------------------------------------------------
        # Fit separate linear models in the random-feature basis.
        # -----------------------------------------------------------
        pinv_rcond = float(self.options.get("pinv_rcond", np.sqrt(np.finfo(float).eps)))
        if solver == "pinv":
            beta_t = np.linalg.pinv(Psi_t.T @ Psi_t, rcond=pinv_rcond) @ Psi_t.T @ Y_t
            beta_c = np.linalg.pinv(Psi_c.T @ Psi_c, rcond=pinv_rcond) @ Psi_c.T @ Y_c
        else:
            beta_t = Ridge(alpha=alpha, fit_intercept=False).fit(Psi_t, Y_t).coef_.reshape(-1)
            beta_c = Ridge(alpha=alpha, fit_intercept=False).fit(Psi_c, Y_c).coef_.reshape(-1)

        # Store internals
        self.feature_map_t = feature_map_t
        self.feature_map_c = feature_map_c
        self.beta_t_ = beta_t
        self.beta_c_ = beta_c

        self.e_t_ = Y_t - Psi_t @ beta_t
        self.e_c_ = Y_c - Psi_c @ beta_c
        self.Psi_t_ = Psi_t
        self.Psi_c_ = Psi_c

        self.X_all_ = np.vstack([X_t, X_c])

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

    def __repr__(self) -> str:
        return f"RFRidgeEstimator(name={self.name}, options={self.options})"
