"""Sieve-style variance estimator for fitted treatment rules."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from opttreat.config import SOBOL_SEED_DEFAULT as _SOBOL_SEED_DEFAULT
from opttreat.data import ensure_2d_features, ensure_vector
from opttreat.sobol import boundary_band, sobol_grid, sobol_grid_with_boundary_points
from .base import VarianceEstimator


class SieveVariance(VarianceEstimator):
    """
    Sieve-based variance estimator for welfare/value parameters built on
    separately-estimated treated/control nuisance functions.

    Attributes:
    - name = "sieve_var"

    Options (``self.options``):

    Required
    - param_type: selects welfare vs value and known vs unknown distribution
      (a string containing "welfare"/"value" and, optionally, "unknown").
    - dim: covariate dimension; sets the Sobol integration grid width.
    - v_func, f_func: callables giving the value function v(X) -> (n,) and the
      covariate density f(X) -> (n,). The value weight is their product
      m(x) = v(x) * f(x); each factor defaults to 1 when absent. Read for value
      parameters only.

    Linear algebra (also read from estimator_output, which takes precedence)
    - alpha: ridge penalty for the Gram solve. No default; required via
      estimator_output['alpha'] or options['alpha']. alpha == 0 routes to pinv.
    - solver: "ridge" (default) or "pinv" (Moore-Penrose pseudo-inverse).
    - pinv_rcond: singular-value cutoff for the pinv path. Default sqrt(eps).

    Sobol integration grid (known-distribution parameters)
    - n_sobol: starting grid size. Default 1024.
    - transform: maps [0,1]^dim to the covariate support. Default identity.
    - sobol_seed: Sobol seed (used when scrambling). Default 456 (shared
      canonical seed opttreat.config.SOBOL_SEED_DEFAULT).
    - sobol_scramble: scramble the Sobol sequence. Default False.

    Value boundary band (value parameters; band is |h| < eps)
    - delta: relative band half-width, eps = delta * sd(h). Default 0.05.
    - loo_eps: absolute eps; overrides delta when present.

    Adaptive grid expansion (value + known distribution)
    - variance_expand_sobol: grow the grid until the band is populated.
      Default True. (Each option below falls back to its loo_* equivalent.)
    - variance_min_band: target # of in-band points. Default min(500, n_sobol).
    - variance_max_sobol: cap on grid size. Default 10 * n_sobol.
    - variance_sobol_expand_factor: per-iteration growth factor. Default 2.0.
    """

    def __init__(self, options: Dict[str, Any] | None = None):
        super().__init__(name="sieve_var", options=options)

    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------
    def fit(self, estimator_output: Dict[str, Any]) -> float:
        """
        Returns estimated asymptotic variance for sqrt(n)(theta_hat - theta).

        Required estimator_output:
            - Psi_t, Psi_c
            - e_t, e_c
            - feature_map_t, feature_map_c
            - h_hat
            - alpha (or options['alpha'])

        Options:
            - dim (required)
            - n_sobol (default 1024)
            - transform (default identity)
            - sobol_seed (default 456)
            - sobol_scramble (default False)
            - param_type (required: indicates welfare vs value + unknown distribution)
            - loo_eps / delta (value band; fixed bandwidth, else delta * sd(h), delta default 0.05)
            - variance_min_band (value only; default min(500, n_sobol))
            - variance_max_sobol (value only; default 10 * n_sobol)
            - variance_sobol_expand_factor (value only; default 2.0)
            - m (only for value)
        """
        # -------------------------
        # Unpack / validate inputs
        # -------------------------
        Psi_t = np.asarray(estimator_output["Psi_t"])
        Psi_c = np.asarray(estimator_output["Psi_c"])
        e_t = np.asarray(estimator_output["e_t"]).ravel()
        e_c = np.asarray(estimator_output["e_c"]).ravel()

        feature_map_t = estimator_output["feature_map_t"]
        feature_map_c = estimator_output["feature_map_c"]
        h_hat = estimator_output["h_hat"]

        if not callable(feature_map_t):
            raise TypeError("estimator_output['feature_map_t'] must be callable.")
        if not callable(feature_map_c):
            raise TypeError("estimator_output['feature_map_c'] must be callable.")
        if not callable(h_hat):
            raise TypeError("estimator_output['h_hat'] must be callable.")

        n_t, K_t = Psi_t.shape
        n_c, K_c = Psi_c.shape

        # Ridge penalty
        alpha = estimator_output.get("alpha", self.options.get("alpha", None))
        if alpha is None:
            raise ValueError("SieveVariance requires alpha in estimator_output['alpha'] or options['alpha'].")

        # -------------------------
        # Param type + integration/evaluation points
        # -------------------------
        param_type = self.options.get("param_type", None)
        pt = "" if param_type is None else str(param_type).lower()
        if "dim" not in self.options:
            raise ValueError("SieveVariance.options must include 'dim'.")

        if "unknown" in pt:
            X_unknown = estimator_output.get("X_eval", estimator_output.get("X_all"))
            if X_unknown is None:
                raise ValueError(
                    "SieveVariance: unknown-distribution parameters require "
                    "estimator_output['X_eval'] or ['X_all']."
                )
            X_int = ensure_2d_features(X_unknown, name="X_eval")
            h_int = ensure_vector(h_hat(X_int), n=X_int.shape[0], name="h_hat(X_eval)")
            grid_diagnostics: dict[str, Any] = {}
        else:
            # Known distribution: build a Sobol integration grid. Value
            # functionals expand it until the boundary band |h| < eps is
            # populated (the same boundary_band convention the kernel uses).
            dim = int(self.options["dim"])
            n_start = int(self.options.get("n_sobol", 1024))
            transform = self.options.get("transform", lambda u: u)
            sobol_seed = int(self.options.get("sobol_seed", _SOBOL_SEED_DEFAULT))
            scramble = bool(self.options.get("sobol_scramble", False))
            if "value" in pt and bool(self.options.get("variance_expand_sobol", True)):
                X_int, h_int = sobol_grid_with_boundary_points(
                    h_hat,
                    self.options,
                    delta=float(self.options.get("delta", 0.05)),
                    dim=dim,
                    n=n_start,
                    transform=transform,
                    sobol_seed=sobol_seed,
                    scramble=scramble,
                    min_band_option="variance_min_band",
                    max_sobol_option="variance_max_sobol",
                    expand_factor_option="variance_sobol_expand_factor",
                )
            else:
                X_int = sobol_grid(dim, n_start, transform, sobol_seed, scramble)
                h_int = ensure_vector(h_hat(X_int), n=X_int.shape[0], name="h_hat(X_int)")
            grid_diagnostics = {
                "n_sobol_requested": int(n_start),
                "n_sobol_final": int(X_int.shape[0]),
            }

        # Feature maps at integration points
        Psi_t_int = np.asarray(feature_map_t(X_int))
        Psi_c_int = np.asarray(feature_map_c(X_int))

        if Psi_t_int.shape[0] != Psi_c_int.shape[0]:
            raise ValueError("feature_map_t(X_int) and feature_map_c(X_int) must have same #rows.")

        # bases = [Psi_t, -Psi_c]
        bases = np.hstack([Psi_t_int, -Psi_c_int])
        n_int = bases.shape[0]

        # -------------------------
        # Bun computation
        # -------------------------
        if "welfare" in pt:
            # Welfare: 1{h >= 0}
            good = h_int >= 0.0
            diagnostics = {
                "n_int": int(n_int),
                "n_positive": int(good.sum()),
                "positive_share": float(good.mean()),
            }
            diagnostics.update(grid_diagnostics)
            if good.any():
                Bun = bases[good, :].sum(axis=0) / n_int
            else:
                Bun = np.zeros(K_t + K_c, dtype=float)

        elif "value" in pt:
            # Value: (1/(2eps)) * 1{|h| < eps} * m(X)
            good, eps = boundary_band(h_int, float(self.options.get("delta", 0.05)), self.options)
            n_band = int(good.sum())
            diagnostics = {
                "n_int": int(n_int),
                "eps": float(eps),
                "n_band": n_band,
                "band_share": float(n_band / n_int),
            }
            diagnostics.update(grid_diagnostics)

            v_func = self.options.get("v_func", None)
            f_func = self.options.get("f_func", None)

            def m_func(Z: np.ndarray) -> np.ndarray:
                out = np.ones(np.asarray(Z).shape[0], dtype=float)
                if callable(v_func):
                    out = out * ensure_vector(v_func(Z), n=out.shape[0], name="v_func(X)")
                if callable(f_func):
                    out = out * ensure_vector(f_func(Z), n=out.shape[0], name="f_func(X)")
                return out

            if good.any():
                # The weight enters only through the in-band points (it is
                # multiplied by 1{|h| < eps}), so evaluate m there only. This
                # avoids paying for an expensive m (e.g. a kernel density) at the
                # full integration grid while leaving the result unchanged.
                m_band = ensure_vector(m_func(X_int[good]), n=int(good.sum()), name="m(X_int[band])")
                Bun = (bases[good, :] * m_band[:, None]).sum(axis=0) / n_int
                Bun = Bun / (2.0 * eps)
            else:
                Bun = np.zeros(K_t + K_c, dtype=float)

        else:
            raise ValueError(
                "SieveVariance: options['param_type'] must indicate 'welfare' or 'value'. "
                f"Got param_type={param_type}."
            )

        Bun_t = Bun[:K_t]
        Bun_c = Bun[K_t:K_t + K_c]

        # -------------------------
        # Core sieve variance piece
        # -------------------------
        I_t = np.eye(K_t)
        I_c = np.eye(K_c)

        solver = str(estimator_output.get("solver", self.options.get("solver", "ridge"))).lower()
        if solver not in ("ridge", "pinv"):
            raise ValueError("SieveVariance: solver must be 'ridge' or 'pinv'.")
        pinv_rcond = float(self.options.get("pinv_rcond", np.sqrt(np.finfo(float).eps)))
        use_pinv = solver == "pinv" or float(alpha) == 0.0

        if use_pinv:
            gram_t_inv = np.linalg.pinv((Psi_t.T @ Psi_t) / n_t, rcond=pinv_rcond)
            gram_c_inv = np.linalg.pinv((Psi_c.T @ Psi_c) / n_c, rcond=pinv_rcond)
        else:
            gram_t = (Psi_t.T @ Psi_t + alpha * I_t) / n_t
            gram_c = (Psi_c.T @ Psi_c + alpha * I_c) / n_c
            gram_t_inv = np.linalg.solve(gram_t, I_t)
            gram_c_inv = np.linalg.solve(gram_c, I_c)

        weights_t = gram_t_inv @ Bun_t
        weights_c = gram_c_inv @ Bun_c

        score_t = e_t * (Psi_t @ weights_t)
        score_c = e_c * (Psi_c @ weights_c)
        influence = np.concatenate([score_t / n_t, score_c / n_c])
        var_sieve = float(np.sum(influence ** 2))

        self.diagnostics_ = dict(diagnostics)
        var_total = var_sieve

   
        # Optional: welfare unknown target-distribution extra term. Value parameters
        # under an unknown distribution take the plain sieve variance (no empirical-
        # distribution variance modification), as suggested by theory.
        if "unknown" in pt and "welfare" in pt:
            empirical_vals = np.maximum(h_int, 0.0)
            n_eval = X_int.shape[0]
            n_fit = int(n_t + n_c)
            var_empirical = float(empirical_vals.var(ddof=0)) if n_eval > 1 else 0.0
            var_total = var_sieve * (n_fit / n_eval) + var_empirical / n_eval

        # Store / return
        self.var_hat_ = float(var_total)
        self.se_ = float(np.sqrt(var_total))
        return float(var_total)

    # Temporarily disabled (multiplier bootstrap critical value).
    # NOTE: re-enabling needs the `influence` vector + `var_sieve` that fit() now
    # computes inline (these used to come from _linearization_components, removed),
    # and the _param_type_str() call below was likewise inlined into fit(). Re-add a
    # shared path (e.g. have fit() cache `influence`) before uncommenting.
    # def bootstrap_critical_value(
    #     self,
    #     estimator_output: Dict[str, Any],
    #     *,
    #     alpha: float = 0.05,
    #     n_boot: int = 500,
    #     random_state: int | None = None,
    #     distribution: str = "normal",
    #     batch_size: int = 100,
    # ) -> float:
    #     """
    #     Compute Chen-Gao style multiplier bootstrap critical value.

    #     The statistic is the conditional quantile of
    #     ``abs(sum_i xi_i * psi_hat_i / se_hat)``, where ``psi_hat_i`` are the
    #     same finite-sample influence contributions used by ``fit()`` to compute
    #     the sieve standard error.
    #     """
    #     if not (0.0 < float(alpha) < 1.0):
    #         raise ValueError("alpha must be between 0 and 1.")
    #     if int(n_boot) <= 0:
    #         raise ValueError("n_boot must be positive.")
    #     if int(batch_size) <= 0:
    #         raise ValueError("batch_size must be positive.")

    #     if "unknown" in self._param_type_str(self.options.get("param_type", None)):
    #         raise NotImplementedError(
    #             "Multiplier bootstrap critical values are implemented for known target distributions. "
    #             "Unknown-distribution targets include an extra empirical-distribution variance piece."
    #         )

    #     components = self._linearization_components(estimator_output)
    #     self.diagnostics_ = dict(components["diagnostics"])
    #     influence = np.asarray(components["influence"], dtype=float)
    #     var_hat = float(components["var_sieve"])
    #     if not np.isfinite(var_hat) or var_hat <= 0.0:
    #         raise ValueError("Cannot bootstrap a nonpositive or nonfinite sieve variance estimate.")

    #     se_hat = float(np.sqrt(var_hat))
    #     rng = np.random.default_rng(random_state)
    #     draws = np.empty(int(n_boot), dtype=float)
    #     distribution = str(distribution).lower()

    #     out = 0
    #     while out < int(n_boot):
    #         size = min(int(batch_size), int(n_boot) - out)
    #         if distribution in ("normal", "gaussian"):
    #             multipliers = rng.standard_normal(size=(size, influence.shape[0]))
    #         elif distribution in ("rademacher", "wild"):
    #             multipliers = rng.choice(np.array([-1.0, 1.0]), size=(size, influence.shape[0]))
    #         else:
    #             raise ValueError("distribution must be 'normal' or 'rademacher'.")
    #         draws[out: out + size] = np.abs(multipliers @ influence / se_hat)
    #         out += size

    #     critical_value = float(np.quantile(draws, 1.0 - float(alpha)))
    #     self.bootstrap_critical_value_ = critical_value
    #     self.bootstrap_diagnostics_ = {
    #         "bootstrap_alpha": float(alpha),
    #         "bootstrap_draws": int(n_boot),
    #         "bootstrap_distribution": distribution,
    #         "bootstrap_batch_size": int(batch_size),
    #         "bootstrap_critical_value": critical_value,
    #     }
    #     self.var_hat_ = var_hat
    #     self.se_ = se_hat
    #     return critical_value

    def __repr__(self) -> str:
        return f"SieveVariance(name={self.name}, options={self.options})"
