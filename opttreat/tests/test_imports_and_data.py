from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opttreat import EstimatorConfig, ParameterConfig, VarianceConfig
from opttreat.data import common_support_mask, split_treated_control
from opttreat.estimation import get_estimator
from opttreat.parameters import get_parameter
from opttreat.variance import get_variance_estimator


def test_package_imports_and_factories() -> None:
    estimator = get_estimator(EstimatorConfig(method="rf_ridge", options={}))
    parameter = get_parameter(ParameterConfig("welfare_known", {"dim": 1}))
    variance = get_variance_estimator(VarianceConfig(method="sieve", options={"dim": 1}))

    assert estimator.name == "rf_ridge"
    assert parameter.name == "welfare_known"
    assert variance.name == "sieve_var"


def test_rf_short_name_is_not_an_estimator_alias() -> None:
    with pytest.raises(ValueError, match="Unknown Estimator method"):
        get_estimator(EstimatorConfig(method="rf", options={}))


def test_split_treated_control_dataframe() -> None:
    data = pd.DataFrame(
        {
            "X1": [0.1, 0.2, 0.3, 0.4],
            "X2": [0.4, 0.3, 0.2, 0.1],
            "d": [1, 0, 1, 0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    parsed = split_treated_control(data)
    assert parsed["X_t"].shape == (2, 2)
    assert parsed["X_c"].shape == (2, 2)
    assert parsed["Y_t"].tolist() == [1.0, 3.0]


def test_split_treated_control_pooled_and_presplit_dicts() -> None:
    pooled = split_treated_control(
        {
            "X": np.array([[1.0], [2.0], [3.0]]),
            "Y": np.array([1.0, 2.0, 3.0]),
            "d": np.array([1, 0, 1]),
        }
    )
    presplit = split_treated_control(
        {
            "X_t": pooled["X_t"],
            "Y_t": pooled["Y_t"],
            "X_c": pooled["X_c"],
            "Y_c": pooled["Y_c"],
        }
    )

    assert pooled["X_t"].shape == (2, 1)
    assert presplit["X_c"].shape == (1, 1)


def test_common_support_mask_uses_strict_treated_control_overlap() -> None:
    X = np.array([[0.0], [0.1], [0.5], [0.9], [1.0]])
    d = np.array([1, 0, 1, 0, 1])

    mask = common_support_mask(X, d, strict=True)
    assert mask.tolist() == [False, False, True, False, False]
