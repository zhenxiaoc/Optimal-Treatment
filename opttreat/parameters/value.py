"""Value target parameters."""

from typing import Callable
import numpy as np

from opttreat.config import SOBOL_SEED_DEFAULT as _SOBOL_SEED_DEFAULT
from opttreat.data import ensure_2d_features, ensure_vector, trimmed_std
from opttreat.sobol import boundary_band, sobol_grid, sobol_grid_with_boundary_points
from .base import Parameter


def _m_gradient_values(
    m_func: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    options: dict,
) -> np.ndarray:
    """
    Evaluate grad_x m(X_i) for the boundary surface formula.

    If X is n by d, this returns an n by d array. Row i is grad_x m(X[i, :]).

    Constant weights have zero gradient. Otherwise this uses an optional
    m_grad_func or central differences of m(x).
    """
    m_values = ensure_vector(m_func(X), n=X.shape[0], name="m(X)")
    spread = float(np.nanmax(m_values) - np.nanmin(m_values)) if m_values.size else 0.0
    # Case 1: m is constant on these rows, so grad m is zero.
    if spread <= 1e-12 * max(1.0, float(np.nanmax(np.abs(m_values))) if m_values.size else 1.0):
        return np.zeros_like(X, dtype=float)

    # Case 2: the user supplied analytic grad m; require an n by d array.
    grad_func = options.get("m_grad_func", None)
    if callable(grad_func):
        grad_v = np.asarray(grad_func(X), dtype=float)
        if grad_v.shape != X.shape:
            raise ValueError(
                "m_grad_func(X) must return an n by d array with the same shape as X. "
                f"Here X has shape {X.shape}, but m_grad_func(X) returned {grad_v.shape}."
            )
        return grad_v

    # Case 3: no analytic grad m was supplied, so approximate it by finite differences.
    step = float(options.get("loo_m_grad_step", 1e-5))
    cols = []
    for j in range(X.shape[1]):
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, j] += step
        X_minus[:, j] -= step
        v_plus = ensure_vector(m_func(X_plus), n=X.shape[0], name="m(X_plus)")
        v_minus = ensure_vector(m_func(X_minus), n=X.shape[0], name="m(X_minus)")
        cols.append((v_plus - v_minus) / (2.0 * step))
    return np.column_stack(cols)


def _h_gradient_values(estimator_output: dict, X: np.ndarray) -> np.ndarray:
    """
    Evaluate grad_x h_hat(X_i), where h_hat = mu_hat_t - mu_hat_c.

    If X is n by d, this returns an n by d array. Row i is
    grad_x h_hat(X[i, :]).
    """
    h_gradient = np.zeros_like(X, dtype=float)
    for sign, prefix in ((1.0, "t"), (-1.0, "c")):
        feature_map = estimator_output[f"feature_map_{prefix}"]
        beta = np.asarray(estimator_output[f"beta_{prefix}"], dtype=float).reshape(-1)
        gradient = getattr(feature_map, "gradient", None)
        if not callable(gradient):
            raise ValueError("feature_map must provide gradient(X), the n by K by d Jacobian of the feature map.")
        feature_gradient = np.asarray(gradient(X), dtype=float)
        if (
            feature_gradient.ndim != 3
            or feature_gradient.shape[0] != X.shape[0]
            or feature_gradient.shape[1] != beta.shape[0]
            or feature_gradient.shape[2] != X.shape[1]
        ):
            raise ValueError(
                "feature_map.gradient(X) must return the Jacobian as an n by K by d array. "
                f"Expected {(X.shape[0], beta.shape[0], X.shape[1])}, got {feature_gradient.shape}."
            )
        h_gradient += sign * np.einsum("nkd,k->nd", feature_gradient, beta, optimize=True)
    return h_gradient


def _h_hessian_values(estimator_output: dict, X: np.ndarray) -> np.ndarray:
    """
    Evaluate Hessian_x h_hat(X_i), where h_hat = mu_hat_t - mu_hat_c.

    If X is n by d, this returns an n by d by d array. Slice i is the
    d by d Hessian matrix of h_hat at X[i, :].
    """
    h_hessian = np.empty((X.shape[0], X.shape[1], X.shape[1]), dtype=float)
    step = 1e-4
    for j in range(X.shape[1]):
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, j] += step
        X_minus[:, j] -= step
        h_hessian[:, :, j] = (
            _h_gradient_values(estimator_output, X_plus)
            - _h_gradient_values(estimator_output, X_minus)
        ) / (2.0 * step)
    return 0.5 * (h_hessian + np.swapaxes(h_hessian, 1, 2))


def _h_laplacian_values(h_hessian: np.ndarray) -> np.ndarray:
    """
    Evaluate Delta_x h_hat(X_i) from Hessian_x h_hat(X_i).
    """
    return np.trace(h_hessian, axis1=1, axis2=2)


def _h_hessian_quadratic_values(h_hessian: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Evaluate v_i.T @ Hessian_x h_hat(X_i) @ v_i.
    """
    return np.einsum("ni,nij,nj->n", vectors, h_hessian, vectors, optimize=True)


def _h_surface_divergence_values(h_gradient: np.ndarray, h_hessian: np.ndarray) -> np.ndarray:
    """
    Evaluate H_hat = div(grad h_hat / ||grad h_hat||).
    """
    h_gradient_norm = np.linalg.norm(h_gradient, axis=1)
    h_laplacian = _h_laplacian_values(h_hessian)
    return (
        h_laplacian / h_gradient_norm
        - _h_hessian_quadratic_values(h_hessian, vectors=h_gradient) / h_gradient_norm**3
    )


def _D2_V_matrix(
    feature_map: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    eps: float,
    n_eval: int,
    m_values: np.ndarray,
    m_gradient: np.ndarray,
    h_gradient: np.ndarray,
    h_gradient_norm: np.ndarray,
    h_hessian: np.ndarray,
    h_surface_divergence: np.ndarray,
) -> np.ndarray:
    """
    Build the coefficient-space matrix for D^2 V(h_hat).

    For directions s(x) = feature_map(x) @ a, this returns A such that
    a.T @ A @ a approximates D^2 V(h_hat)[s, s] on the boundary band, using
    the surface formula

        -int_M s^2 [
            ||grad h||^{-1} n.T grad{m / ||grad h||}
            + m ||grad h||^{-2} H
        ] dH
        -2 int_M m ||grad h||^{-2} s n.T grad s dH,

    where n = grad h / ||grad h|| and H = div(n). The boundary band supplies
    the quadrature weights approximating dH.
    """
    r = h_gradient_norm
    n = h_gradient / r[:, None]
    H = h_surface_divergence
    grad_m_over_h = (
        m_gradient / r[:, None]
        - (m_values / r**3)[:, None] * np.einsum("nij,nj->ni", h_hessian, h_gradient, optimize=True)
    )

    first_integral_coefficient = np.sum(n * grad_m_over_h, axis=1) / r + m_values * H / r**2
    boundary_weight = r / (2.0 * eps * float(n_eval))

    direction_values = np.asarray(feature_map(X), dtype=float)
    gradient = getattr(feature_map, "gradient", None)
    if not callable(gradient):
        raise ValueError("feature_map must provide gradient(X), the n by K by d Jacobian of the feature map.")
    direction_gradient = np.asarray(gradient(X), dtype=float)
    if (
        direction_gradient.ndim != 3
        or direction_gradient.shape[0] != X.shape[0]
        or direction_gradient.shape[1] != direction_values.shape[1]
        or direction_gradient.shape[2] != X.shape[1]
    ):
        raise ValueError(
            "feature_map.gradient(X) must return the Jacobian as an n by K by d array. "
            f"Expected {(X.shape[0], direction_values.shape[1], X.shape[1])}, got {direction_gradient.shape}."
        )
    first_surface_matrix = direction_values.T @ (
        (boundary_weight * first_integral_coefficient)[:, None] * direction_values
    )
    second_surface_matrix = direction_values.T @ (
        ((boundary_weight * m_values / r**2)[:, None])
        * np.einsum("nd,nkd->nk", n, direction_gradient, optimize=True)
    )
    A = -(first_surface_matrix + second_surface_matrix + second_surface_matrix.T)
    return 0.5 * (A + A.T)


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

    The matrix A represents D^2 V(h_hat) in coefficient space. For each LOO
    direction a_i = G_a^{-1} b(X_i) / n_a, this evaluates
    a_i.T @ A @ a_i = D^2 V(h_hat)[s_i, s_i] and sums the terms times
    e_i,loo^2. Chunking only controls memory because the correction is an
    additive sum over observations.
    """
    n_arm = int(Psi_arm.shape[0])
    gram = (Psi_arm.T @ Psi_arm) / float(n_arm)
    if solver == "ridge" and alpha > 0.0:
        gram = gram + (alpha / float(n_arm)) * np.eye(gram.shape[0])
    inv_gram = np.linalg.pinv(gram, rcond=rcond)

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


def _value_loo_method(options: dict, *, default: str) -> str:
    """
    Read and validate the value LOO second-derivative approximation method.

    Returns either "band" for the boundary-band formula or
    "central_difference" for the finite-difference approximation.
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


def _loo_value_central_difference(
    estimator_output: dict,
    X: np.ndarray,
    value_weight: np.ndarray,
    h_vals: np.ndarray,
    plugin: float,
    options: dict,
    *,
    delta0: float | None,
    chunk: int,
) -> float:
    """
    Apply the central-difference LOO correction to plug-in value.

    The value functional is V(h) = mean(1{h(X) > 0} m(X)). For each arm
    and observation, the code forms the LOO direction ell_i of h_hat on the
    evaluation rows X, approximates D^2 V(h_hat)[ell_i, ell_i] by central
    differences, and subtracts the summed quadratic correction.

    Let G_a = (Psi_a.T @ Psi_a + alpha * I) / n_a. The LOO direction evaluated
    on X is ell_i = Psi_eval @ G_a^{-1} b(X_i) / n_a. Following the paper's
    LOO formula, the code finite-differences the scaled direction
    s_i = n * ell_i directly and then subtracts the summed quadratic
    correction divided by 2 n^2.
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
        n_arm = int(Psi_arm.shape[0])
        gram = (Psi_arm.T @ Psi_arm) / float(n_arm)
        if solver == "ridge" and alpha > 0.0:
            gram = gram + (alpha / float(n_arm)) * np.eye(gram.shape[0])
        inv_gram = np.linalg.pinv(gram, rcond=rcond)

        arm_total = 0.0
        for start in range(0, Psi_arm.shape[0], chunk_int):
            block = Psi_arm[start : start + chunk_int]
            H_diag = np.einsum("ij,jk,ik->i", block, inv_gram, block, optimize=True) / float(n_arm)
            H_diag = np.clip(H_diag, 0.0, 0.999)
            e_loo = resid[start : start + chunk_int] / (1.0 - H_diag)
            loo_direction = (Psi_eval @ (inv_gram @ block.T)) / float(n_arm)
            sensitivity = n_total * loo_direction

            plus = ((h_vals[:, None] + step * sensitivity > 0.0).astype(float) * value_weight[:, None]).mean(axis=0)
            minus = ((h_vals[:, None] - step * sensitivity > 0.0).astype(float) * value_weight[:, None]).mean(axis=0)
            second_derivative = (plus - 2.0 * plugin + minus) / step**2
            arm_total += float(np.sum(second_derivative * e_loo**2))

        total += arm_total

    return float(plugin - total / (2.0 * n_total**2))


def _loo_value_boundary_correction(
    estimator_output: dict,
    X: np.ndarray,
    m_values: np.ndarray,
    h_vals: np.ndarray,
    plugin: float,
    m_func: Callable[[np.ndarray], np.ndarray],
    options: dict,
    *,
    delta0: float | None,
    chunk: int,
) -> float:
    """
    Apply the boundary-band LOO correction to plug-in value.

    For V(h)=E[1{h(X)>0}m(X)], the second derivative is concentrated near
    h(X)=0. This helper keeps evaluation rows with |h_hat(X)|<eps, builds the
    coefficient-space matrix A for the tube/divergence approximation, sums the
    treated and control arm quadratic LOO terms, and subtracts the correction.

    If h_hat=mu_hat_t-mu_hat_c, treated observations perturb h_hat through
    +s_i,t and control observations through -s_i,c. The sign drops out because
    D^2 V[-s,-s]=D^2 V[s,s], so the arm corrections add.
    """
    delta = float(options.get("loo_delta0", 0.05) if delta0 is None else delta0)
    mask, eps = boundary_band(h_vals, delta, options)
    if not np.any(mask):
        return plugin

    X_band = X[mask]
    m_values_band = m_values[mask]
    h_gradient = _h_gradient_values(estimator_output, X_band)
    h_gradient_norm = np.linalg.norm(h_gradient, axis=1)
    keep = np.isfinite(h_gradient_norm) & (h_gradient_norm > float(options.get("loo_grad_floor", 1e-10)))
    if not np.any(keep):
        return plugin

    X_band = X_band[keep]
    m_values_band = m_values_band[keep]
    h_gradient = h_gradient[keep]
    h_gradient_norm = h_gradient_norm[keep]

    m_gradient = _m_gradient_values(m_func, X_band, options)
    h_hessian = _h_hessian_values(estimator_output, X_band)
    h_surface_divergence = _h_surface_divergence_values(h_gradient, h_hessian)
    solver = str(estimator_output.get("solver", "ridge")).lower()
    alpha = float(estimator_output.get("alpha", 0.0))
    rcond = float(options.get("pinv_rcond", np.sqrt(np.finfo(float).eps)))

    correction = 0.0
    for prefix in ("t", "c"):
        A = _D2_V_matrix(
            estimator_output[f"feature_map_{prefix}"],
            X_band,
            eps,
            X.shape[0],
            m_values_band,
            m_gradient,
            h_gradient,
            h_gradient_norm,
            h_hessian,
            h_surface_divergence,
        )
        Psi_arm = np.asarray(estimator_output[f"Psi_{prefix}"], dtype=float)
        resid = np.asarray(estimator_output[f"e_{prefix}"], dtype=float).reshape(-1)
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


class ValueKnownDist(Parameter):
    """
    Value under a known target distribution: integral 1{h(x)>0} m(x) dx.

    The weight m(x) = v(x) * f(x) is the product of the user value function v
    (options['v_func']) and the covariate density f (options['f_func']). Each
    factor defaults to 1 when its option is absent.
    """

    def plug_in(self, h: Callable[[np.ndarray], np.ndarray]) -> float:
        if not callable(h):
            raise TypeError("h must be a callable of the form h(x).")

        v_func = self.options.get("v_func", None)
        f_func = self.options.get("f_func", None)

        def m_func(Z: np.ndarray) -> np.ndarray:
            out = np.ones(np.asarray(Z).shape[0], dtype=float)
            if callable(v_func):
                out = out * ensure_vector(v_func(Z), n=out.shape[0], name="v_func(X)")
            if callable(f_func):
                out = out * ensure_vector(f_func(Z), n=out.shape[0], name="f_func(X)")
            return out

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        transform = self.options.get("transform", lambda u: u)
        sobol_seed = int(self.options.get("sobol_seed", _SOBOL_SEED_DEFAULT))
        scramble = bool(self.options.get("sobol_scramble", False))

        X = sobol_grid(dim, n, transform, sobol_seed, scramble)
        h_vals = ensure_vector(h(X), n=X.shape[0], name="h(X)")
        m_values = ensure_vector(m_func(X), n=X.shape[0], name="m(X)")
        return float(((h_vals > 0.0).astype(float) * m_values).mean())

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
        v_func = self.options.get("v_func", None)
        f_func = self.options.get("f_func", None)

        def m_func(Z: np.ndarray) -> np.ndarray:
            out = np.ones(np.asarray(Z).shape[0], dtype=float)
            if callable(v_func):
                out = out * ensure_vector(v_func(Z), n=out.shape[0], name="v_func(X)")
            if callable(f_func):
                out = out * ensure_vector(f_func(Z), n=out.shape[0], name="f_func(X)")
            return out
        method = _value_loo_method(self.options, default="band")
        h_hat = estimator_output["h_hat"]
        if X is None:
            dim = int(self.options["dim"])
            n = int(self.options.get("n_sobol", 1024))
            transform = self.options.get("transform", lambda u: u)
            sobol_seed = int(self.options.get("sobol_seed", _SOBOL_SEED_DEFAULT))
            scramble = bool(self.options.get("sobol_scramble", False))
            if method == "band" or (
                method == "central_difference" and bool(self.options.get("loo_expand_sobol", True))
            ):
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

        m_values = ensure_vector(m_func(X), n=X.shape[0], name="m(X)")
        plugin = float(((h_vals > 0.0).astype(float) * m_values).mean())
        if method == "central_difference":
            return _loo_value_central_difference(
                estimator_output,
                X,
                m_values,
                h_vals,
                plugin,
                self.options,
                delta0=delta0,
                chunk=chunk,
            )
        return _loo_value_boundary_correction(
            estimator_output,
            X,
            m_values,
            h_vals,
            plugin,
            m_func,
            self.options,
            delta0=delta0,
            chunk=chunk,
        )

    def get_true_value(self, model):
        if "true_value" in self.options:
            return float(self.options["true_value"])

        h0 = model.h0
        v_func = self.options.get("v_func", None)
        f_func = self.options.get("f_func", None)

        def m_func(Z: np.ndarray) -> np.ndarray:
            out = np.ones(np.asarray(Z).shape[0], dtype=float)
            if callable(v_func):
                out = out * ensure_vector(v_func(Z), n=out.shape[0], name="v_func(X)")
            if callable(f_func):
                out = out * ensure_vector(f_func(Z), n=out.shape[0], name="f_func(X)")
            return out

        if not callable(h0):
            raise TypeError("model.h0 must be callable.")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        sobol_seed = int(self.options.get("true_sobol_seed", self.options.get("sobol_seed", _SOBOL_SEED_DEFAULT)))
        scramble = bool(self.options.get("sobol_scramble", False))
        transform = self.options.get("transform", model.inverse_CDF)
        X = sobol_grid(dim, n, transform, sobol_seed, scramble)
        h_vals = ensure_vector(h0(X), n=X.shape[0], name="h0(X)")
        m_values = ensure_vector(m_func(X), n=X.shape[0], name="m(X)")
        return float(((h_vals > 0.0).astype(float) * m_values).mean())



class ValueUnknownDist(Parameter):
    """
    Value using supplied rows: mean_i 1{h(X_i)>0} m(X_i).

    The weight m(x) = v(x) * f(x) is the product of the user value function v
    (options['v_func']) and the covariate density f (options['f_func']); each
    factor defaults to 1 when its option is absent. Because the supplied rows
    are already drawn from the data distribution, the empirical mean integrates
    against f implicitly, so here m is typically just the value function v_func.
    """

    def plug_in(self, h: Callable[[np.ndarray], np.ndarray], X: np.ndarray | None = None) -> float:
        if not callable(h):
            raise TypeError("h must be a callable of the form h(x).")

        v_func = self.options.get("v_func", None)
        f_func = self.options.get("f_func", None)

        def m_func(Z: np.ndarray) -> np.ndarray:
            out = np.ones(np.asarray(Z).shape[0], dtype=float)
            if callable(v_func):
                out = out * ensure_vector(v_func(Z), n=out.shape[0], name="v_func(X)")
            if callable(f_func):
                out = out * ensure_vector(f_func(Z), n=out.shape[0], name="f_func(X)")
            return out

        if X is None:
            X = self.options.get("X")
        if X is None:
            raise ValueError("ValueUnknownDist.plug_in requires observed X.")

        X_arr = ensure_2d_features(X, name="X")
        h_vals = ensure_vector(h(X_arr), n=X_arr.shape[0], name="h(X)")
        m_values = ensure_vector(m_func(X_arr), n=X_arr.shape[0], name="m(X)")
        return float(((h_vals > 0.0).astype(float) * m_values).mean())

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
        v_func = self.options.get("v_func", None)
        f_func = self.options.get("f_func", None)

        def m_func(Z: np.ndarray) -> np.ndarray:
            out = np.ones(np.asarray(Z).shape[0], dtype=float)
            if callable(v_func):
                out = out * ensure_vector(v_func(Z), n=out.shape[0], name="v_func(X)")
            if callable(f_func):
                out = out * ensure_vector(f_func(Z), n=out.shape[0], name="f_func(X)")
            return out
        method = _value_loo_method(self.options, default="band")
        if X is None:
            X = self.options.get("X", estimator_output.get("X_all"))
        if X is None:
            raise ValueError("ValueUnknownDist.loo requires observed X.")
        X_arr = ensure_2d_features(X, name="X")
        m_values = ensure_vector(m_func(X_arr), n=X_arr.shape[0], name="m(X)")

        h_hat = estimator_output["h_hat"]
        h_vals = ensure_vector(h_hat(X_arr), n=X_arr.shape[0], name="h_hat(X)")
        plugin = float(((h_vals > 0.0).astype(float) * m_values).mean())
        if method == "central_difference":
            return _loo_value_central_difference(
                estimator_output,
                X_arr,
                m_values,
                h_vals,
                plugin,
                self.options,
                delta0=delta0,
                chunk=chunk,
            )
        return _loo_value_boundary_correction(
            estimator_output,
            X_arr,
            m_values,
            h_vals,
            plugin,
            m_func,
            self.options,
            delta0=delta0,
            chunk=chunk,
        )

    def get_true_value(self, model):
        h0 = model.h0
        v_func = self.options.get("v_func", None)
        f_func = self.options.get("f_func", None)

        def m_func(Z: np.ndarray) -> np.ndarray:
            out = np.ones(np.asarray(Z).shape[0], dtype=float)
            if callable(v_func):
                out = out * ensure_vector(v_func(Z), n=out.shape[0], name="v_func(X)")
            if callable(f_func):
                out = out * ensure_vector(f_func(Z), n=out.shape[0], name="f_func(X)")
            return out

        if not callable(h0):
            raise TypeError("model.h0 must be callable.")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        transform = self.options.get("transform", model.inverse_CDF)
        sobol_seed = int(self.options.get("true_sobol_seed", self.options.get("sobol_seed", _SOBOL_SEED_DEFAULT)))
        scramble = bool(self.options.get("sobol_scramble", False))
        X = sobol_grid(dim, n, transform, sobol_seed, scramble)
        h_vals = ensure_vector(h0(X), n=X.shape[0], name="h0(X)")
        m_values = ensure_vector(m_func(X), n=X.shape[0], name="m(X)")
        return float(((h_vals > 0.0).astype(float) * m_values).mean())
