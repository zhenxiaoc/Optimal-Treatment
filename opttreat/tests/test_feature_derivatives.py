from __future__ import annotations

import numpy as np

from opttreat.estimation.features.feature_factory import build_feature_map_from_options


def _feature_map(activation: str):
    X_sample = np.array(
        [
            [-0.4, 0.1],
            [0.2, -0.3],
            [0.7, 0.5],
            [-0.1, -0.6],
        ],
        dtype=float,
    )
    return build_feature_map_from_options(
        {
            "rfg_type": "iid_sphere",
            "activation": activation,
            "n_features": 5,
            "random_state": 2026,
        },
        X_sample,
    )


def test_smooth_random_feature_maps_expose_analytic_derivative_hooks() -> None:
    for activation in ("sigmoid", "cosine", "exp"):
        fmap = _feature_map(activation)

        assert callable(getattr(fmap, "derivative", None))
        assert callable(getattr(fmap, "gradient", None))
        assert callable(getattr(fmap, "laplacian_values", None))
        assert callable(getattr(fmap, "hessian_quadratic_values", None))


def test_smooth_random_feature_coordinate_derivatives_match_finite_differences() -> None:
    X = np.array([[-0.25, 0.4], [0.3, -0.2], [0.6, 0.7]], dtype=float)
    step = 1e-5

    for activation in ("sigmoid", "cosine", "exp"):
        fmap = _feature_map(activation)
        F0 = np.asarray(fmap(X), dtype=float)
        for axis in range(X.shape[1]):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[:, axis] += step
            X_minus[:, axis] -= step
            F_plus = np.asarray(fmap(X_plus), dtype=float)
            F_minus = np.asarray(fmap(X_minus), dtype=float)

            first_fd = (F_plus - F_minus) / (2.0 * step)
            second_fd = (F_plus - 2.0 * F0 + F_minus) / step**2

            np.testing.assert_allclose(fmap.derivative(X, axis=axis, order=1), first_fd, atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(fmap.derivative(X, axis=axis, order=2), second_fd, atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(fmap.gradient(X)[:, :, axis], fmap.derivative(X, axis=axis, order=1))


def test_smooth_random_feature_laplacian_and_hessian_quadratic_hooks() -> None:
    X = np.array([[-0.25, 0.4], [0.3, -0.2], [0.6, 0.7]], dtype=float)
    vectors = np.array([[0.2, -0.1], [-0.3, 0.5], [0.4, 0.1]], dtype=float)
    step = 1e-5

    for activation in ("sigmoid", "cosine", "exp"):
        fmap = _feature_map(activation)
        beta = np.linspace(-0.3, 0.4, np.asarray(fmap(X)).shape[1])

        lap_expected = sum(fmap.derivative(X, axis=axis, order=2) @ beta for axis in range(X.shape[1]))
        np.testing.assert_allclose(fmap.laplacian_values(X, beta), lap_expected)

        h0 = np.asarray(fmap(X), dtype=float) @ beta
        h_plus = np.asarray(fmap(X + step * vectors), dtype=float) @ beta
        h_minus = np.asarray(fmap(X - step * vectors), dtype=float) @ beta
        hessian_quad_fd = (h_plus - 2.0 * h0 + h_minus) / step**2

        np.testing.assert_allclose(
            fmap.hessian_quadratic_values(X, beta, vectors),
            hessian_quad_fd,
            atol=1e-5,
            rtol=1e-4,
        )
