from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from opttreat.models import (
    CCGModel,
    Model1,
    Model2,
    Model3,
    Model4,
    Model5,
    Model6,
    Model7,
    Model8,
    Model9,
    Model10,
    Model11,
    Model12,
    Model13,
    Model14,
    Model15,
    TaylorExpansionModel,
    get_model,
)


def test_registered_models_generate_data() -> None:
    for model in [Model1(), Model4(), Model8(), Model15(), TaylorExpansionModel(K=3)]:
        data = model.generate_data(20)
        assert {"d", "y"}.issubset(data.columns)
        assert data.filter(like="X").shape[0] == 20


def test_model_registry_uses_two_model_families() -> None:
    assert isinstance(get_model("Model4"), CCGModel)
    assert isinstance(TaylorExpansionModel(K=3, expansion="tan2"), TaylorExpansionModel)

    for name in ["Model99", "Model100", "Model101", "Model102"]:
        with pytest.raises(ValueError, match=f"Unknown model '{name}'"):
            get_model(name)


def test_ccg_models_are_registered_as_ccg_models() -> None:
    factories = [
        Model1,
        Model2,
        Model3,
        Model4,
        Model5,
        Model6,
        Model7,
        Model8,
        Model9,
        Model10,
        Model11,
        Model12,
        Model13,
        Model14,
        Model15,
    ]

    assert set(CCGModel.DEFINITIONS) == {f"Model{index}" for index in range(1, 16)}

    for index, factory in enumerate(factories, start=1):
        model = factory()
        assert isinstance(model, CCGModel)
        assert model.name == f"Model{index}"
        assert isinstance(get_model(f"Model{index}"), CCGModel)


def test_ccg_model_exposes_definition_information() -> None:
    model = CCGModel("Model1")

    assert model.definition is CCGModel.DEFINITIONS["Model1"]
    assert model.dim == 1
    assert model.observed_support == ((-0.2,), (1.2,))
    assert model.target_support == ((0.0,), (1.0,))


def test_explicit_taylor_tan2_special_case_returns_constant_effect_vectors() -> None:
    model = TaylorExpansionModel(K=3, expansion="tan2")
    X = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
    d = np.array([0, 1, 0])

    assert model.p0(X).shape == (3,)
    assert model.h0(X).shape == (3,)
    assert model.mu0(X, d).shape == (3,)
    assert np.allclose(model.h0(X), 1.0)
    assert model.K == 3
    assert model.expansion == "tan2"


def test_taylor_model_true_effect_shape() -> None:
    model = TaylorExpansionModel(K=4, expansion="hyperbolic")
    X = np.random.default_rng(123).uniform(0, 1, size=(5, 4))
    assert model.h0(X).shape == (5,)


def test_models_4_to_14_match_ccg_2025_specs() -> None:
    X = np.array([[0.25, 0.75], [0.8, 0.1]])
    d = np.array([0.0, 1.0])
    x1 = X[:, 0]
    x2 = X[:, 1]

    expectations = [
        (
            Model4(),
            expit(x1 - x2),
            (1 - x1**2 - x2**2) * (4 + np.sin(x1) * x2 + np.cos(x2)) + d * (0.5 * x1 - 0.4 * x2),
        ),
        (
            Model5(),
            expit(x1 - x2),
            (1 - x1 * x2) * (3 + np.sin(np.pi * x1) * np.cos(np.pi * x2)) + d * (0.3 * x1 - 0.3 * x2),
        ),
        (
            Model6(),
            expit(1.5 * x1 - 0.5 * x2),
            np.log(1 + x1 + x2) + d * (x1 - 0.7 * x2),
        ),
        (
            Model7(),
            expit(-0.5 + x1 + 2 * x2),
            (x1**2 + x2**2) * np.exp(-x1 - x2) + d * (0.5 - x2),
        ),
        (
            Model11(),
            expit(x1 - x2),
            (1 - x1**2 - x2**2) * (4 + np.sin(x1) * x2 + np.cos(x2)) + d * (0.5 * x1 - 0.4 * x2),
        ),
        (
            Model12(),
            expit(x1 - x2),
            (1 - x1 * x2) * (3 + np.sin(np.pi * x1) * np.cos(np.pi * x2)) + d * (0.3 * x1 - 0.3 * x2),
        ),
        (
            Model13(),
            expit(1.5 * x1 - 0.5 * x2),
            np.log(1 + x1 + x2) + d * (x1 - 0.7 * x2),
        ),
        (
            Model14(),
            expit(-0.5 + x1 + 2 * x2),
            (x1**2 + x2**2) * np.exp(-x1 - x2) + d * (0.5 - x2),
        ),
    ]

    for model, expected_p0, expected_mu0 in expectations:
        np.testing.assert_allclose(model.p0(X), expected_p0)
        np.testing.assert_allclose(model.mu0(X, d), expected_mu0)


def test_models_1_to_3_and_8_to_10_match_ccg_2025_specs() -> None:
    X = np.array([[0.25], [0.8]])
    d = np.array([0.0, 1.0])
    x = X[:, 0]

    expectations = [
        (
            Model1(),
            expit(1 - 2 * x),
            5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * x) + d * (-0.4 + 2 * x**2),
        ),
        (
            Model2(),
            expit(-0.5 + x),
            0.5 * np.abs(x) + d * (0.5 - x**2),
        ),
        (
            Model3(),
            expit(0.5 - x),
            x**2 + d * (1 - x),
        ),
        (
            Model8(),
            expit(1 - 2 * x),
            5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * x) + d * (-0.4 + 2 * x**2),
        ),
        (
            Model9(),
            expit(-0.5 + x),
            0.5 * np.abs(x) + d * (0.5 - x**2),
        ),
        (
            Model10(),
            expit(0.5 - x),
            x**2 + d * (1 - x),
        ),
    ]

    for model, expected_p0, expected_mu0 in expectations:
        np.testing.assert_allclose(model.p0(X), expected_p0)
        np.testing.assert_allclose(model.mu0(X, d), expected_mu0)


def test_model15_matches_ccg_2025_value_spec() -> None:
    X = np.array([[0.0, 0.0], [1.2, 0.0]])
    model = Model15()
    expected_h = (1 - X[:, 0] ** 2 - X[:, 1] ** 2) * (4 + np.sin(X[:, 0]) * X[:, 1] + np.cos(X[:, 1]))

    np.testing.assert_allclose(model.p0(X), expit(X[:, 0] - X[:, 1]))
    np.testing.assert_allclose(model.h0(X), expected_h)
    np.testing.assert_allclose(model.mu0(X, np.array([1.0, 1.0])), expected_h)
