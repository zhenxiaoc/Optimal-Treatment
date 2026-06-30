from __future__ import annotations

import numpy as np

from opttreat.config import EstimatorConfig, ParameterConfig
from opttreat.data import split_treated_control, trimmed_std
from opttreat.estimation import get_estimator
from opttreat.models import Model4, TaylorExpansionModel
from opttreat.parameters import get_parameter
from opttreat.parameters.value import _loo_value_central_difference


def test_welfare_known_and_unknown_plug_in() -> None:
    h = lambda X: X[:, 0] - 0.5
    known = get_parameter(
        ParameterConfig(
            "welfare_known",
            {"dim": 1, "n_sobol": 8, "sobol_seed": 1},
        )
    )
    unknown = get_parameter(ParameterConfig("welfare_unknown", {"dim": 1}))
    X = np.array([[0.25], [0.75]])

    assert known.plug_in(h) >= 0.0
    assert unknown.plug_in(h, X) == 0.125


def test_value_unknown_uses_indicator_not_relu_weight() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    h = lambda Z: np.array([2.0, -2.0, 0.5])
    v = lambda Z: np.ones(Z.shape[0]) * 3.0
    parameter = get_parameter(
        ParameterConfig(
            "value_unknown",
            {"dim": 1, "X": X, "v_func": v},
        )
    )

    assert parameter.plug_in(h) == 2.0


def test_value_true_value_broadcasts_scalar_h_and_m() -> None:
    parameter = get_parameter(
        ParameterConfig(
            "value_known",
            {
                "dim": 1,
                "n_sobol": 8,
                "sobol_seed": 2,
                "v_func": lambda X: 2.0,
            },
        )
    )

    assert parameter.get_true_value(TaylorExpansionModel(K=1, expansion="tan2")) == 2.0


def test_loo_parameters_accept_estimator_output() -> None:
    np.random.seed(123)
    parsed = split_treated_control(Model4().generate_data(70))
    estimator = get_estimator(
        EstimatorConfig(
            method="rf_ridge",
            options={
                "solver": "pinv",
                "rfg_type": "iid_sphere",
                "activation": "sigmoid",
                "share_features": True,
                "n_features": 8,
                "random_state": 123,
            },
        )
    )
    output = estimator.fit(parsed)
    X_eval = output["X_all"][:20]

    welfare = get_parameter(ParameterConfig("welfare_unknown", {"dim": 2}))
    value = get_parameter(
        ParameterConfig(
            "value_unknown",
            {
                "dim": 2,
                "v_func": lambda X: np.ones(X.shape[0]),
            },
        )
    )
    value_fd = get_parameter(
        ParameterConfig(
            "value_unknown",
            {
                "dim": 2,
                "v_func": lambda X: np.ones(X.shape[0]),
                "loo_method": "central_difference",
            },
        )
    )
    known_band = get_parameter(
        ParameterConfig("welfare_known", {"dim": 2, "loo_method": "band"})
    )
    known_fd = get_parameter(
        ParameterConfig("welfare_known", {"dim": 2, "loo_method": "central_difference"})
    )
    unknown_band = get_parameter(
        ParameterConfig("welfare_unknown", {"dim": 2, "loo_method": "band"})
    )

    assert np.isfinite(welfare.loo(output, X_eval, chunk=5, max_per_arm=10, rng=np.random.default_rng(1)))
    assert np.isfinite(value.loo(output, X_eval, chunk=5, max_per_arm=10, rng=np.random.default_rng(1)))
    assert np.isfinite(value_fd.loo(output, X_eval, chunk=5, max_per_arm=10, rng=np.random.default_rng(1)))
    assert np.isfinite(known_band.loo(output, X_eval, chunk=5, max_per_arm=10, rng=np.random.default_rng(1)))
    assert np.isfinite(known_fd.loo(output, X_eval, chunk=5, max_per_arm=10, rng=np.random.default_rng(1)))
    assert np.isfinite(unknown_band.loo(output, X_eval, chunk=5, max_per_arm=10, rng=np.random.default_rng(1)))


def test_value_known_central_loo_expands_sobol_grid() -> None:
    call_sizes: list[int] = []

    def h_hat(X: np.ndarray) -> np.ndarray:
        call_sizes.append(int(X.shape[0]))
        values = np.ones(X.shape[0], dtype=float)
        values[: X.shape[0] // 16] = 0.0
        return values

    def feature_map(X: np.ndarray) -> np.ndarray:
        return np.ones((X.shape[0], 1), dtype=float)

    estimator_output = {
        "h_hat": h_hat,
        "X_all": np.array([[0.0], [1.0]]),
        "solver": "pinv",
        "Psi_t": np.ones((1, 1), dtype=float),
        "Psi_c": np.ones((1, 1), dtype=float),
        "e_t": np.array([0.1]),
        "e_c": np.array([-0.1]),
        "feature_map_t": feature_map,
        "feature_map_c": feature_map,
    }
    parameter = get_parameter(
        ParameterConfig(
            "value_known",
            {
                "dim": 1,
                "v_func": lambda X: np.ones(X.shape[0]),
                "n_sobol": 8,
                "sobol_seed": 1,
                "loo_method": "central_difference",
                "loo_min_band": 2,
                "loo_max_sobol": 64,
                "loo_sobol_expand_factor": 2.0,
            },
        )
    )

    assert np.isfinite(parameter.loo(estimator_output, chunk=1))
    assert max(call_sizes) >= 32


def test_value_central_loo_uses_raw_paper_direction() -> None:
    X_eval = np.array([[1.0], [0.0], [0.0], [0.0]])
    # A spread-out h so the band-trimmed scale is non-degenerate (no point mass
    # in the inlier core), matching the step the implementation actually uses.
    h_vals = np.array([-0.01, 1.0, 2.0, 3.0])
    plugin = float((h_vals > 0.0).mean())

    def feature_map(X: np.ndarray) -> np.ndarray:
        return X[:, :1]

    estimator_output = {
        "X_all": np.zeros((4, 1), dtype=float),
        "solver": "pinv",
        "Psi_t": np.ones((2, 1), dtype=float),
        "Psi_c": np.ones((2, 1), dtype=float),
        "e_t": np.array([1.0, 0.0]),
        "e_c": np.array([0.0, 0.0]),
        "feature_map_t": feature_map,
        "feature_map_c": feature_map,
    }

    eta = 0.1 * trimmed_std(h_vals)
    expected = plugin - 1.0 / (32.0 * eta**2)
    actual = _loo_value_central_difference(
        estimator_output,
        X_eval,
        np.ones(X_eval.shape[0], dtype=float),
        h_vals,
        plugin,
        {},
        delta0=0.1,
        chunk=2,
    )

    assert np.isclose(actual, expected)
