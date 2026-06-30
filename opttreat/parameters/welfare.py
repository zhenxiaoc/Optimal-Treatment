"""Welfare target parameters."""

from typing import Callable
import warnings
import numpy as np
from scipy.stats.qmc import Sobol

from opttreat.config import SOBOL_SEED_DEFAULT as _SOBOL_SEED_DEFAULT
from opttreat.data import ensure_2d_features, ensure_vector, trimmed_std
from opttreat.sobol import boundary_band, sobol_grid, sobol_grid_with_boundary_points
from .base import Parameter
from opttreat.models.model_base import ModelBase


def _welfare_second_derivative_matrix(
    feature_map: Callable[[np.ndarray], np.ndarray],
    X_band: np.ndarray,
    *,
    n_eval: int,
    eps: float,
) -> np.ndarray:
    """
    Build the coefficient-space second-derivative matrix for welfare.

    For directions s(x) = feature_map(x) @ a, this returns A such that
    a.T @ A @ a approximates D^2 W(h_hat)[s, s] on the boundary band.
    """
    B = np.asarray(feature_map(X_band), dtype=float)
    weight = 1.0 / (2.0 * float(eps) * float(n_eval))
    A = weight * (B.T @ B)
    return 0.5 * (A + A.T)


def _normalized_arm_gram_inverse(
    Psi_arm: np.ndarray,
    *,
    solver: str,
    alpha: float,
    rcond: float,
) -> tuple[np.ndarray, int]:
    """
    Return the inverse of the normalized empirical arm Gram matrix.

    Uses G_a = (Psi_a.T @ Psi_a + alpha * I) / n_a for ridge and
    G_a = Psi_a.T @ Psi_a / n_a for pinv/OLS. With this convention,
    H_ii = b(X_i).T @ G_a^{-1} @ b(X_i) / n_a.
    """
    n_arm = int(Psi_arm.shape[0])
    gram = (Psi_arm.T @ Psi_arm) / float(n_arm)
    if solver == "ridge" and alpha > 0.0:
        gram = gram + (alpha / float(n_arm)) * np.eye(gram.shape[0])
    return np.linalg.pinv(gram, rcond=rcond), n_arm


def _arm_quadratic_correction(
    A: np.ndarray,
    Psi_arm: np.ndarray,
    resid: np.ndarray,
    *,
    solver: str,
    alpha: float,
    rcond: float,
    chunk: int,
) -> float:
    """
    Compute one arm's leave-one-out quadratic correction.

    The input A is produced by _welfare_second_derivative_matrix. For each
    LOO direction a_i = G^{-1} b(X_i), this evaluates a_i.T @ A @ a_i,
    i.e. D^2 W(h_hat)[s_i, s_i], and sums the terms times e_i,loo^2.

    The loop over chunks is only for memory control. The correction is an
    additive sum over observations,
    sum_i e_i,loo^2 * D^2 W(h_hat)[s_i, s_i], so splitting the observations
    into blocks and adding the block sums gives the same result as forming all
    directions at once.
    """
    inv_gram, n_arm = _normalized_arm_gram_inverse(
        Psi_arm,
        solver=solver,
        alpha=alpha,
        rcond=rcond,
    )

    total = 0.0
    chunk_int = max(1, int(chunk))
    for start in range(0, Psi_arm.shape[0], chunk_int):
        block = Psi_arm[start : start + chunk_int]
        H_diag = np.einsum("ij,jk,ik->i", block, inv_gram, block, optimize=True) / float(n_arm)
        H_diag = np.clip(H_diag, 0.0, 0.999)
        e_loo = resid[start : start + chunk_int] / (1.0 - H_diag)
        directions = (inv_gram @ block.T) / float(n_arm)
        d2_vals = np.einsum("kj,kj->j", directions, A @ directions, optimize=True)
        total += float(np.sum(d2_vals * e_loo**2))
    return 0.5 * total


def _welfare_loo_method(options: dict, *, default: str) -> str:
    """
    Read and validate the welfare LOO second-derivative approximation method.

    Accepts aliases for the boundary-band matrix method and the central-
    difference method, returning either "band" or "central_difference".
    """
    method = str(options.get("loo_method", default)).lower().replace("-", "_")
    if method in ("band", "boundary_band", "tube"):
        return "band"
    if method in ("central", "central_difference", "finite_difference", "fd"):
        return "central_difference"
    raise ValueError(
        "options['loo_method'] must be one of "
        "'band' or 'central_difference'."
    )


def _welfare_band_loo(
    estimator_output: dict,
    X: np.ndarray,
    h_vals: np.ndarray,
    plugin: float,
    options: dict,
    *,
    delta0: float | None,
    chunk: int,
) -> float:
    """
    Apply the boundary-band LOO correction to the plug-in welfare estimate.

    Builds the fixed-band second-derivative matrix near h_hat(X) = 0, computes
    the quadratic LOO correction for the treated and control arms, and subtracts
    that correction from the plug-in welfare.

    If h_hat(x) = mu_hat_t(x) - mu_hat_c(x), then a treated observation changes
    h_hat through a treated-arm direction +s_i,t, while a control observation
    changes h_hat through a control-arm direction -s_i,c. The LOO correction is
    the sum of one-observation quadratic terms
    0.5 * sum_a sum_i e_i,a,loo^2 * D^2 W(h_hat)[s_i,a, s_i,a],
    for a in {t, c}. The minus sign for the control arm drops out because the
    second derivative is quadratic: D^2 W[-s, -s] = D^2 W[s, s].
    """
    delta = float(options.get("loo_delta0", 0.05) if delta0 is None else delta0)
    mask, eps = boundary_band(h_vals, delta, options)
    if not np.any(mask):
        return plugin

    X_band = X[mask]
    solver = str(estimator_output.get("solver", "ridge")).lower()
    alpha = float(estimator_output.get("alpha", 0.0))
    rcond = float(options.get("pinv_rcond", np.sqrt(np.finfo(float).eps)))

    correction = 0.0
    for prefix in ("t", "c"):
        Psi_arm = np.asarray(estimator_output[f"Psi_{prefix}"], dtype=float)
        resid = np.asarray(estimator_output[f"e_{prefix}"], dtype=float).reshape(-1)
        feature_map = estimator_output[f"feature_map_{prefix}"]
        A = _welfare_second_derivative_matrix(
            feature_map,
            X_band,
            n_eval=X.shape[0],
            eps=eps,
        )
        correction += _arm_quadratic_correction(
            A,
            Psi_arm,
            resid,
            solver=solver,
            alpha=alpha,
            rcond=rcond,
            chunk=chunk,
        )

    return float(plugin - correction)


def _welfare_central_difference_loo(
    estimator_output: dict,
    X: np.ndarray,
    h_vals: np.ndarray,
    plugin: float,
    options: dict,
    *,
    delta0: float | None,
    chunk: int,
) -> float:
    """
    Apply the central-difference LOO correction to plug-in welfare.

    For each arm and each observation, forms the LOO sensitivity direction of
    h_hat, approximates the welfare second derivative by perturbing h_vals up
    and down along that direction, then subtracts the summed quadratic
    correction from the plug-in welfare estimate.

    Let G_a = (Psi_a.T @ Psi_a + alpha * I) / n_a. The LOO direction evaluated
    on X is ell_i = Psi_eval @ G_a^{-1} b(X_i) / n_a. For numerical
    stability, the code finite-differences the scaled direction
    s_i = n * ell_i, rescales it as d_i = s_i / scale_i, where
    scale_i = std_eval(s_i), computes D^2 W(h_hat)[d_i, d_i], and converts back
    using D^2 W(h_hat)[s_i, s_i] = scale_i^2 * D^2 W(h_hat)[d_i, d_i].
    """
    step = float(options.get("loo_delta0", 0.05) if delta0 is None else delta0) * max(
        trimmed_std(h_vals),
        1e-12,
    )
    n_total = ensure_2d_features(estimator_output["X_all"], name="X_all").shape[0]
    solver = str(estimator_output.get("solver", "ridge")).lower()
    alpha = float(estimator_output.get("alpha", 0.0))
    rcond = float(options.get("pinv_rcond", np.sqrt(np.finfo(float).eps)))

    total = 0.0
    chunk_int = max(1, int(chunk))
    for prefix in ("t", "c"):
        Psi_arm = np.asarray(estimator_output[f"Psi_{prefix}"], dtype=float)
        resid = np.asarray(estimator_output[f"e_{prefix}"], dtype=float).reshape(-1)
        feature_map = estimator_output[f"feature_map_{prefix}"]
        Psi_eval = np.asarray(feature_map(X), dtype=float)
        inv_gram, n_arm = _normalized_arm_gram_inverse(
            Psi_arm,
            solver=solver,
            alpha=alpha,
            rcond=rcond,
        )

        arm_total = 0.0
        for start in range(0, Psi_arm.shape[0], chunk_int):
            block = Psi_arm[start : start + chunk_int]
            H_diag = np.einsum("ij,jk,ik->i", block, inv_gram, block, optimize=True) / float(n_arm)
            H_diag = np.clip(H_diag, 0.0, 0.999)
            e_loo = resid[start : start + chunk_int] / (1.0 - H_diag)
            loo_direction = (Psi_eval @ (inv_gram @ block.T)) / float(n_arm)
            sensitivity = n_total * loo_direction
            scale = sensitivity.std(axis=0)
            scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
            direction = sensitivity / scale[None, :]
            second_derivative = (
                np.maximum(h_vals[:, None] + step * direction, 0.0).mean(axis=0)
                - 2.0 * plugin
                + np.maximum(h_vals[:, None] - step * direction, 0.0).mean(axis=0)
            ) / step**2
            arm_total += float(np.sum(second_derivative * scale**2 * e_loo**2))

        total += arm_total

    return float(plugin - total / (2.0 * n_total**2))


class WelfareKnownDist(Parameter):
    """
    Welfare under a known target distribution: E[max(h(X), 0)].
    """

    def plug_in(self, h: Callable[[np.ndarray], np.ndarray]) -> float:
        if not callable(h):
            raise TypeError("h must be a callable of the form h(X).")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        transform = self.options.get("transform", lambda u: u)
        sobol_seed = int(self.options.get("sobol_seed", _SOBOL_SEED_DEFAULT))
        scramble = bool(self.options.get("sobol_scramble", False))

        X = sobol_grid(dim, n, transform, sobol_seed, scramble)
        h_vals = ensure_vector(h(X), n=X.shape[0], name="h(X)")
        return float(np.maximum(h_vals, 0.0).mean())

    def loo(
        self,
        estimator_output: dict,
        X: np.ndarray | None = None,
        *,
        delta0: float | None = None,
        chunk: int = 256,
        max_per_arm: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        h_hat = estimator_output["h_hat"]
        method = _welfare_loo_method(self.options, default="band")
        if X is None:
            dim = int(self.options["dim"])
            n = int(self.options.get("n_sobol", 1024))
            transform = self.options.get("transform", lambda u: u)
            sobol_seed = int(self.options.get("sobol_seed", _SOBOL_SEED_DEFAULT))
            scramble = bool(self.options.get("sobol_scramble", False))
            if method == "band":
                delta = float(self.options.get("loo_delta0", 0.05) if delta0 is None else delta0)
                X, h_vals = sobol_grid_with_boundary_points(
                    h_hat,
                    self.options,
                    delta=delta,
                    dim=dim,
                    n=n,
                    transform=transform,
                    sobol_seed=sobol_seed,
                    scramble=scramble,
                )
            else:
                X = sobol_grid(dim, n, transform, sobol_seed, scramble)
                h_vals = ensure_vector(h_hat(X), n=X.shape[0], name="h_hat(X)")
        else:
            X = ensure_2d_features(X, name="X")
            h_vals = ensure_vector(h_hat(X), n=X.shape[0], name="h_hat(X)")

        plugin = float(np.maximum(h_vals, 0.0).mean())
        if method == "central_difference":
            return _welfare_central_difference_loo(
                estimator_output,
                X,
                h_vals,
                plugin,
                self.options,
                delta0=delta0,
                chunk=chunk,
            )
        return _welfare_band_loo(
            estimator_output,
            X,
            h_vals,
            plugin,
            self.options,
            delta0=delta0,
            chunk=chunk,
        )

    def get_true_value(self, model: ModelBase) -> float:
        h0 = model.h0
        if not callable(h0):
            raise TypeError("model.h0 must be callable.")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        sobol_seed = int(self.options.get("true_sobol_seed", self.options.get("sobol_seed", _SOBOL_SEED_DEFAULT)))
        scramble = bool(self.options.get("sobol_scramble", False))

        engine = Sobol(d=dim, scramble=scramble, seed=sobol_seed if scramble else None)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The balance properties of Sobol")
            U = engine.random(n)
        transform = self.options.get("transform", model.inverse_CDF)
        X = ensure_2d_features(transform(U), name="X_int")
        h_vals = ensure_vector(h0(X), n=X.shape[0], name="h0(X)")
        return float(np.maximum(h_vals, 0.0).mean())



class WelfareUnknownDist(Parameter):
    """
    Welfare under the empirical distribution: n^{-1} sum_i max(h(X_i), 0).
    """

    def plug_in(self, h: Callable[[np.ndarray], np.ndarray], X: np.ndarray | None = None) -> float:
        if not callable(h):
            raise TypeError("h must be a callable of the form h0(X).")
        if X is None:
            X = self.options.get("X")
        if X is None:
            raise ValueError("WelfareUnknownDist.plug_in requires observed X.")

        X_arr = ensure_2d_features(X, name="X")
        h_vals = ensure_vector(h(X_arr), n=X_arr.shape[0], name="h(X)")
        return float(np.maximum(h_vals, 0.0).mean())

    def loo(
        self,
        estimator_output: dict,
        X: np.ndarray | None = None,
        *,
        delta0: float | None = None,
        chunk: int = 256,
        max_per_arm: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        if X is None:
            X = self.options.get("X", estimator_output.get("X_all"))
        if X is None:
            raise ValueError("WelfareUnknownDist.loo requires observed X.")
        X_arr = ensure_2d_features(X, name="X")

        h_hat = estimator_output["h_hat"]
        h_vals = ensure_vector(h_hat(X_arr), n=X_arr.shape[0], name="h_hat(X)")
        plugin = float(np.maximum(h_vals, 0.0).mean())
        if _welfare_loo_method(self.options, default="central_difference") == "band":
            return _welfare_band_loo(
                estimator_output,
                X_arr,
                h_vals,
                plugin,
                self.options,
                delta0=delta0,
                chunk=chunk,
            )
        return _welfare_central_difference_loo(
            estimator_output,
            X_arr,
            h_vals,
            plugin,
            self.options,
            delta0=delta0,
            chunk=chunk,
        )

    def get_true_value(self, model: ModelBase) -> float:
        h0 = model.h0
        if not callable(h0):
            raise TypeError("model.h0 must be callable.")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        transform = self.options.get("transform", model.inverse_CDF)
        sobol_seed = int(self.options.get("true_sobol_seed", self.options.get("sobol_seed", _SOBOL_SEED_DEFAULT)))
        scramble = bool(self.options.get("sobol_scramble", False))
        engine = Sobol(d=dim, scramble=scramble, seed=sobol_seed if scramble else None)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The balance properties of Sobol")
            U = engine.random(n)
        X = ensure_2d_features(transform(U), name="X_int")
        h_vals = ensure_vector(h0(X), n=X.shape[0], name="h0(X)")
        return float(np.maximum(h_vals, 0.0).mean())
