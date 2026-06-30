"""Analytic plug-in variance estimator for the welfare functional.

This is the closed-form asymptotic variance used by the Kalamazoo plug-in
welfare script (``TEST_Emp_Welfare_Spline_PlugIn.R``). It is specific to the
welfare functional ``E[max(h(X), 0)]`` and has no value-functional analogue, so
it raises if asked to run on a value parameter.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from opttreat.data import ensure_2d_features, ensure_vector
from opttreat.estimation.splines.spline_factory import build_spline_basis_from_options
from .base import VarianceEstimator


class WelfarePlugInVariance(VarianceEstimator):
    """Plug-in welfare asymptotic variance (welfare functional only).

    Implements the analytic variance of the plug-in welfare estimate
    ``W_hat = n^{-1} sum_i max(h_hat(X_i), 0)`` over the evaluation sample:

        Var(W_hat) = ( Var_i[ max(h_hat(X_i), 0) ]
                       + E_i[ 1{h_hat(X_i) >= 0} * u_i^2 / (p(X_i) (1 - p(X_i))) ]
                     ) / n,

    where ``u_i = Y_i - mu_hat_{D_i}(X_i)`` is the own-arm outcome residual and
    ``p(x) = P(D = 1 | x)`` is the propensity score. ``Var_i`` is the sample
    variance (``ddof=1``) and ``E_i`` the sample mean over the evaluation rows.

    The propensity is estimated internally by a sieve regression of ``D`` on
    ``X`` (generalized-inverse least squares), using ``options['propensity_options']``
    for the spline family. Alternatively a precomputed propensity may be supplied
    through ``options['propensity']`` as a callable ``p(X) -> (n,)`` or an array
    aligned with the evaluation rows.

    Required ``estimator_output`` keys
    ----------------------------------
    - ``X_eval``, ``D_eval``, ``Y_eval`` : evaluation sample (the support over
      which welfare is averaged), its treatment indicator and its outcome.
    - ``mu_hat_t``, ``mu_hat_c`` : fitted treated/control conditional means.
    - ``X_all``, ``D_all`` : full-sample covariates and treatment, row-aligned,
      used to fit the propensity (only needed when no ``propensity`` is supplied).

    Options
    -------
    - ``param_type`` : optional; must indicate a welfare parameter if given.
    - ``propensity`` : optional callable or array; skips the internal sieve fit.
    - ``propensity_options`` : spline options for the internal propensity sieve.
    - ``pinv_rcond`` : generalized-inverse cutoff for the propensity solve
      (default ``sqrt(eps)``, matching R's ``MASS::ginv``).
    """

    def __init__(self, options: Dict[str, Any] | None = None):
        super().__init__(name="welfare_plugin_var", options=options)

    def fit(self, estimator_output: Dict[str, Any]) -> float:
        self._require_welfare(self.options.get("param_type"))

        X = ensure_2d_features(estimator_output["X_eval"], name="X_eval")
        n = X.shape[0]
        D = ensure_vector(estimator_output["D_eval"], n=n, name="D_eval")
        Y = ensure_vector(estimator_output["Y_eval"], n=n, name="Y_eval")

        mu_t = estimator_output.get("mu_hat_t")
        mu_c = estimator_output.get("mu_hat_c")
        if not callable(mu_t) or not callable(mu_c):
            raise TypeError("WelfarePlugInVariance requires callable 'mu_hat_t' and 'mu_hat_c'.")
        mu_t_x = ensure_vector(mu_t(X), n=n, name="mu_hat_t(X_eval)")
        mu_c_x = ensure_vector(mu_c(X), n=n, name="mu_hat_c(X_eval)")
        gap = mu_t_x - mu_c_x

        # Own-arm outcome residual and the treatment-rule indicator.
        resid = np.where(D == 1, Y - mu_t_x, Y - mu_c_x)
        ind = (gap >= 0.0).astype(float)

        p = self._propensity(estimator_output, X)

        welfare = np.maximum(gap, 0.0)
        asy_var = float(
            np.var(welfare, ddof=1)
            + np.mean(ind * resid**2 / (p * (1.0 - p)))
        )
        var_hat = asy_var / n

        self.var_hat_ = float(var_hat)
        self.se_ = float(np.sqrt(max(var_hat, 0.0)))
        self.diagnostics_ = {"n_eval": int(n), "positive_share": float(ind.mean())}
        return float(var_hat)

    # ------------------------------------------------------------------
    def _propensity(self, estimator_output: Dict[str, Any], X_eval: np.ndarray) -> np.ndarray:
        """Return the propensity score at the evaluation rows."""
        prop = self.options.get("propensity", None)
        if callable(prop):
            return ensure_vector(prop(X_eval), n=X_eval.shape[0], name="propensity(X_eval)")
        if prop is not None:
            return ensure_vector(prop, n=X_eval.shape[0], name="propensity")

        # Fit p(x) = P(D = 1 | x) by a sieve regression of D on X over the full
        # sample, then evaluate it at the evaluation rows.
        X_all = ensure_2d_features(estimator_output["X_all"], name="X_all")
        D_all = ensure_vector(estimator_output["D_all"], n=X_all.shape[0], name="D_all")
        rcond = float(self.options.get("pinv_rcond", np.sqrt(np.finfo(float).eps)))
        feature_map = build_spline_basis_from_options(
            dict(self.options.get("propensity_options", {})),
            X_all,
        )
        Psi = feature_map(X_all)
        beta = np.linalg.pinv(Psi.T @ Psi, rcond=rcond) @ Psi.T @ D_all
        return np.asarray(feature_map(X_eval) @ beta, dtype=float)

    @staticmethod
    def _require_welfare(param_type: Any) -> None:
        if param_type is None:
            return
        pt = str(getattr(param_type, "value", param_type)).lower()
        if "welfare" not in pt:
            raise ValueError(
                "WelfarePlugInVariance is defined for the welfare functional only; "
                f"got param_type={param_type!r}."
            )

    def __repr__(self) -> str:
        return f"WelfarePlugInVariance(name={self.name}, options={self.options})"
