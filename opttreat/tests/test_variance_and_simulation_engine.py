from __future__ import annotations

import numpy as np
import pytest

from opttreat.config import EstimatorConfig, VarianceConfig
from opttreat.data import split_treated_control
from opttreat.estimation import get_estimator
from opttreat.models import Model4
from opttreat.variance import get_variance_estimator


def _rf_output():
    np.random.seed(123)
    parsed = split_treated_control(Model4().generate_data(90))
    estimator = get_estimator(
        EstimatorConfig(
            method="rf_ridge",
            options={
                "rfg_type": "iid_sphere",
                "activation": "sigmoid",
                "share_features": True,
                "n_features": 14,
                "random_state": 123,
                "alpha": 1e-3,
            },
        )
    )
    return estimator.fit(parsed)


def test_sieve_variance_welfare_known_and_unknown() -> None:
    output = _rf_output()

    for param_type in ["welfare_known", "welfare_unknown"]:
        variance = get_variance_estimator(
            VarianceConfig(
                method="sieve",
                options={
                    "alpha": 1e-3,
                    "dim": 2,
                    "n_sobol": 16,
                    "sobol_seed": 321,
                    "param_type": param_type,
                },
            )
        )
        var_hat = variance.fit(output)
        assert np.isfinite(var_hat)
        assert var_hat >= 0.0


@pytest.mark.skip(reason="bootstrap_critical_value temporarily disabled")
def test_sieve_variance_multiplier_bootstrap_critical_value() -> None:
    output = _rf_output()
    variance = get_variance_estimator(
        VarianceConfig(
            method="sieve",
            options={
                "alpha": 1e-3,
                "dim": 2,
                "n_sobol": 16,
                "sobol_seed": 321,
                "param_type": "welfare_known",
            },
        )
    )

    var_hat = variance.fit(output)
    critical_value = variance.bootstrap_critical_value(
        output,
        alpha=0.1,
        n_boot=25,
        random_state=99,
        batch_size=7,
    )

    assert np.isfinite(var_hat)
    assert np.isfinite(critical_value)
    assert critical_value > 0.0
    assert variance.bootstrap_diagnostics_["bootstrap_draws"] == 25
    assert variance.bootstrap_diagnostics_["bootstrap_alpha"] == 0.1


def test_sieve_variance_value_known() -> None:
    output = _rf_output()
    variance = get_variance_estimator(
        VarianceConfig(
            method="sieve",
            options={
                "alpha": 1e-3,
                "dim": 2,
                "n_sobol": 16,
                "sobol_seed": 321,
                "param_type": "value_known",
                "loo_eps": 0.1,
                "v_func": lambda X: np.ones(X.shape[0]),
            },
        )
    )

    var_hat = variance.fit(output)
    assert np.isfinite(var_hat)
    assert var_hat >= 0.0
    assert variance.diagnostics_["n_sobol_requested"] == 16
    assert 16 <= variance.diagnostics_["n_int"] <= 160
    assert variance.diagnostics_["n_sobol_final"] == variance.diagnostics_["n_int"]
    assert 0 <= variance.diagnostics_["n_band"] <= variance.diagnostics_["n_int"]
    assert 0.0 <= variance.diagnostics_["band_share"] <= 1.0
    assert variance.diagnostics_["eps"] == 0.1
