"""Public configuration objects used by OptTreat factories.

The ``*Config`` dataclasses each carry a free-form ``options`` dict that is
passed unchanged to the estimator/parameter/variance implementation. Defaults
for those keys live at the call sites that read them
(``options.get(key, default)``); the one exception is the Sobol seed, whose
canonical default is :data:`SOBOL_SEED_DEFAULT` here so every integration grid
shares it.

The commented "Configuration recipes" block at the bottom of this module is a
bookkeeping reference: which option keys each simulation/empirical combination
passes to the estimator, parameter, and variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class EstimatorConfig:
    """Factory input for first-stage estimators.

    ``method`` names the estimator implementation, such as ``"sieve"`` or
    ``"rf_ridge"``. ``options`` are passed to that estimator unchanged.
    """

    method: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class VarianceConfig:
    """Factory input for variance estimators.

    ``method`` names the variance estimator implementation. ``options`` are
    passed to that estimator unchanged.
    """

    method: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterConfig:
    """Factory input for target parameters.

    ``param_type`` is one of :attr:`VALID_TYPES` (``"welfare_known"``,
    ``"welfare_unknown"``, ``"value_known"``, ``"value_unknown"``) and selects
    welfare/value and known/unknown distribution logic. ``options`` are passed
    to the parameter implementation unchanged.
    """

    VALID_TYPES: ClassVar[tuple[str, ...]] = (
        "welfare_known",
        "welfare_unknown",
        "value_known",
        "value_unknown",
    )

    param_type: str
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pt = str(self.param_type).strip().lower()
        if pt not in self.VALID_TYPES:
            raise ValueError(
                f"param_type must be one of {self.VALID_TYPES}; got {self.param_type!r}."
            )
        self.param_type = pt


# Canonical Sobol integration-grid seed. The parameter and variance modules
# import this so plug-in, LOO, true-value, and variance grids share one default
# and cannot drift apart. (They historically disagreed: value used 456,
# welfare 1, variance 789.) The seed only matters when sobol_scramble=True; with
# the default scramble=False it is ignored entirely.
SOBOL_SEED_DEFAULT = 456


__all__ = [
    "EstimatorConfig",
    "ParameterConfig",
    "SOBOL_SEED_DEFAULT",
    "VarianceConfig",
]


# ===========================================================================
# Configuration recipes used in the simulations and empirical study
# ===========================================================================
# Bookkeeping reference (not executed): for each combination actually run, the
# option keys handed to the estimator / parameter / variance. Keys not listed
# fall back to each component's own default (its ``options.get(key, default)``
# call). "sieve | rf_ridge" = either learner is used with the keys shown under
# it. A "+ output:" line lists the extra keys the runner injects into
# estimator_output before calling variance.fit(...). A "calls:" line shows the
# runtime arguments to plug_in/loo: the LOO band half-width delta0 and the
# chunk/max_per_arm/rng knobs travel as function ARGUMENTS, not options. The
# runners override only delta0=LOO_DELTA0 and leave chunk/max_per_arm/rng at the
# loo() defaults (256 / None / None). The band-grid knobs loo_min_band /
# loo_max_sobol / loo_sobol_expand_factor are read by the known-dist band LOO
# but no runner overrides them, so they never appear below.
#
# ----------------------------------------------------------------------------
# SIMULATIONS
# ----------------------------------------------------------------------------
#
# CCG2025 / welfare_known            (ccg2025/ccg_welfare_known)
#   estimator sieve    : solver, share_features, J_x_degree, J_x_segments_t,
#                        J_x_segments_c, knots, basis, X_min, X_max, pinv_rcond
#   estimator rf_ridge : rfg_type, activation, share_features, n_features,
#                        random_state, alpha
#   parameter welfare_known : dim, n_sobol, transform, sobol_scramble,
#                             loo_method, pinv_rcond
#   variance  sieve_var     : param_type, dim, n_sobol, transform,
#                             sobol_scramble, pinv_rcond
#   calls     : plug_in(h);  loo(output, delta0=LOO_DELTA0);  variance.fit(output)
#
# CCG2025 / welfare_unknown          (ccg2025/ccg_welfare_unknown)
#   estimator sieve | rf_ridge : same key sets as welfare_known above
#   parameter welfare_unknown : dim, n_sobol, transform, sobol_scramble,
#                               loo_method, pinv_rcond
#   variance  sieve_var       : param_type, dim, pinv_rcond
#                               (unknown dist averages over data X_eval, no grid)
#                       + output: X_eval
#   calls     : plug_in(h, X_eval);  loo(output, X_eval, delta0=LOO_DELTA0);
#               variance.fit(output + X_eval)
#
# CCG2025 / value_known              (ccg2025/ccg_value_known)
#   estimator sieve | rf_ridge : same key sets as welfare_known above
#   parameter value_known : dim, n_sobol, transform, sobol_scramble, v_func,
#                           loo_method, true_value, pinv_rcond
#   variance  sieve_var   : param_type, dim, n_sobol, transform, sobol_scramble,
#                           v_func, loo_eps, pinv_rcond
#   calls     : plug_in(h);  loo(output, delta0=LOO_DELTA0);  variance.fit(output)
#
# DenseIndexDGP / welfare_known      (DenseIndexDGP/welfare_known)
#   estimator rf_ridge : rfg_type, activation, share_features, n_features,
#                        random_state, alpha, solver, pinv_rcond
#   parameter welfare_known : dim, n_sobol, sobol_scramble, pinv_rcond
#   variance  sieve_var     : param_type, dim, n_sobol, sobol_scramble, alpha,
#                             solver, pinv_rcond
#   calls     : plug_in(h);  loo(output, delta0=LOO_DELTA0);  variance.fit(output)
#
# DenseIndexDGP / value_known        (DenseIndexDGP/value_known)
#   estimator rf_ridge : rfg_type, activation, share_features, n_features,
#                        random_state, alpha, solver, pinv_rcond
#   parameter value_known : dim, n_sobol, sobol_scramble, v_func, pinv_rcond
#   variance  sieve_var   : param_type, dim, n_sobol, sobol_scramble, v_func,
#                           delta, alpha, solver, pinv_rcond
#   calls     : plug_in(h);  loo(output, delta0=LOO_DELTA0);  variance.fit(output)
#
# TaylorModel / welfare_known        (TaylorModel/run_taylor_rf)
#   estimator rf_ridge : rfg_type, activation, share_features, n_features,
#                        random_state, alpha, solver, pinv_rcond
#   parameter welfare_known : dim, n_sobol, sobol_scramble, pinv_rcond
#   variance  sieve_var     : param_type, dim, n_sobol, sobol_scramble, alpha,
#                             solver, pinv_rcond
#   calls     : plug_in(h);  loo(output, delta0=LOO_DELTA0);  variance.fit(output)
#
# ----------------------------------------------------------------------------
# EMPIRICAL  (empirical/run_empirical.py, KT data; estimator is always sieve)
# ----------------------------------------------------------------------------
# Shared sieve estimator: share_features, J_x_degree, J_x_segments_t,
#   J_x_segments_c, knots, basis, solver, pinv_rcond, extrapolate
#
# Welfare plug-in SE                 (welfare_plugin)
#   parameter welfare_unknown : dim
#   variance  welfare_plugin  : param_type, propensity_options, pinv_rcond
#                       + output: X_eval, X_all, D_all, D_eval, Y_eval
#   calls     : plug_in(h, X_trim);  variance.fit(output + ...)   [plug-in only, no loo]
#
# Welfare sieve SE                   (welfare_sievevar)
#   parameter welfare_unknown : dim
#   variance  sieve_var       : param_type, dim
#                       + output: X_eval
#   calls     : plug_in(h, support);  variance.fit(output + X_eval)   [plug-in only, no loo]
#
# Value sieve SE                     (value_sievevar)
#   parameter value_unknown : dim, v_func
#   variance  sieve_var     : param_type (= "value_known"), dim, n_sobol,
#                             sobol_scramble, transform, v_func, f_func, loo_eps,
#                             variance_expand_sobol, solver, pinv_rcond, alpha
#   calls     : plug_in(h, support);  variance.fit(output)   [plug-in only, no loo]
#   NB: the parameter is value_unknown but the variance is run as value_known
#       (Sobol integral against the KDE density f_func), to match the R script.
# ===========================================================================
