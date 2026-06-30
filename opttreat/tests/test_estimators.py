from __future__ import annotations

import numpy as np

from opttreat.config import EstimatorConfig
from opttreat.data import split_treated_control
from opttreat.estimation import get_estimator
from opttreat.models import Model4


def _parsed_model4_data():
    np.random.seed(123)
    return split_treated_control(Model4().generate_data(80))


def test_rf_ridge_estimator_output_contract() -> None:
    estimator = get_estimator(
        EstimatorConfig(
            method="rf_ridge",
            options={
                "rfg_type": "iid_sphere",
                "activation": "sigmoid",
                "share_features": True,
                "n_features": 12,
                "random_state": 123,
                "alpha": 1e-3,
            },
        )
    )

    output = estimator.fit(_parsed_model4_data())
    X_all = output["X_all"]
    h_vals = np.asarray(output["h_hat"](X_all)).reshape(-1)

    assert h_vals.shape == (X_all.shape[0],)
    assert output["Psi_t"].shape[1] == 12
    assert output["alpha"] == 1e-3


def test_rf_estimator_pinv_matches_normal_equation_solution() -> None:
    parsed = _parsed_model4_data()
    estimator = get_estimator(
        EstimatorConfig(
            method="rf_ridge",
            options={
                "solver": "pinv",
                "rfg_type": "iid_sphere",
                "activation": "sigmoid",
                "share_features": True,
                "n_features": 12,
                "random_state": 123,
            },
        )
    )

    output = estimator.fit(parsed)
    rcond = np.sqrt(np.finfo(float).eps)
    beta_t_expected = np.linalg.pinv(output["Psi_t"].T @ output["Psi_t"], rcond=rcond) @ output["Psi_t"].T @ parsed["Y_t"]
    beta_c_expected = np.linalg.pinv(output["Psi_c"].T @ output["Psi_c"], rcond=rcond) @ output["Psi_c"].T @ parsed["Y_c"]

    np.testing.assert_allclose(output["beta_t"], beta_t_expected)
    np.testing.assert_allclose(output["beta_c"], beta_c_expected)
    np.testing.assert_allclose(output["e_t"], parsed["Y_t"] - output["Psi_t"] @ output["beta_t"])
    assert output["solver"] == "pinv"
    assert output["alpha"] == 0.0


def test_sieve_estimator_output_has_finite_design() -> None:
    estimator = get_estimator(
        EstimatorConfig(
            method="sieve",
            options={
                "knots": "uniform",
                "J_x_degree": 2,
                "J_x_segments": 2,
                "basis": "tensor",
                "X_min": None,
                "X_max": None,
                "alpha": 1e-3,
                "share_features": True,
            },
        )
    )

    output = estimator.fit(_parsed_model4_data())

    assert np.isfinite(output["Psi_t"]).all()
    assert np.isfinite(output["Psi_c"]).all()
    assert output["h_hat"](output["X_all"][:3]).shape == (3,)


def test_sieve_estimator_pinv_matches_normal_equation_solution() -> None:
    parsed = _parsed_model4_data()
    estimator = get_estimator(
        EstimatorConfig(
            method="sieve",
            options={
                "solver": "pinv",
                "knots": "uniform",
                "J_x_degree": 2,
                "J_x_segments": 1,
                "basis": "tensor",
                "share_features": False,
            },
        )
    )

    output = estimator.fit(parsed)
    rcond = np.sqrt(np.finfo(float).eps)
    beta_t_expected = np.linalg.pinv(output["Psi_t"].T @ output["Psi_t"], rcond=rcond) @ output["Psi_t"].T @ parsed["Y_t"]
    beta_c_expected = np.linalg.pinv(output["Psi_c"].T @ output["Psi_c"], rcond=rcond) @ output["Psi_c"].T @ parsed["Y_c"]

    np.testing.assert_allclose(output["beta_t"], beta_t_expected)
    np.testing.assert_allclose(output["beta_c"], beta_c_expected)
    np.testing.assert_allclose(output["e_t"], parsed["Y_t"] - output["Psi_t"] @ output["beta_t"])
    assert output["solver"] == "pinv"
    assert output["alpha"] == 0.0
