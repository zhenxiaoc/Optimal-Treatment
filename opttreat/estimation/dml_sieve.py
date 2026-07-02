"""Cross-fitted sieve-debiased (DML) estimator via the sieve Riesz representer.

Implements the cross-fitted sieve-debiased estimator of
``main_260630ZX.tex`` Section ``sec:SieveDML`` (eq. ``eq:debias_est``),

.. math::

    \\tilde\\theta = \\frac{1}{K} \\sum_{\\kappa=1}^{K} \\Big(
        \\Theta(\\hat\\mu_{-\\kappa})
        + \\frac{K}{n} \\sum_{i \\in I_\\kappa}
          \\hat v^{*}_{K_n,-\\kappa}(X_i, D_i)\\,(Y_i - \\hat\\mu_{-\\kappa}(X_i, D_i))
    \\Big),

which unifies the (regular) welfare functional and the (irregular) value
functional under a known target distribution ``F``: the same sieve Riesz
representer

.. math::

    \\hat v^{*}_{K_n}(x, d)
      = \\bar\\psi(x, d)^{\\top}\\, \\hat R_{K_n}^{-1}\\,
        \\hat D_\\mu \\Theta(\\hat\\mu)[\\bar\\psi]

is used in both cases, and its norm ``sigma_n = ||v*||_sd`` (bounded for
welfare, divergent for value) governs the rate. The pathwise derivative
``Bun = D_mu Theta(mu_hat)[psi-bar]`` and the arm Riesz weights
``w_a = G_a^{-1} Bun_a`` are computed by
:func:`opttreat.variance.sieve_riesz_core` -- the exact same engine used for the
sieve variance -- so the point estimate and the SieveVar standard error share
one construction. Evaluated off-fold, the representer is
``v*(x, 1) = (n / n_t) psi_t(x)^T w_t`` and
``v*(x, 0) = (n / n_c) psi_c(x)^T w_c``.

The first stage ``hat mu_{-kappa}`` is refit on each training fold with the
supplied estimator; because the correction annihilates its first-order error on
the sieve space, the estimator tolerates generic machine-learning first stages
(here the B-spline sieve or random-feature ridge).

The standard error is the sieve sandwich variance (Theorem ``thm:SieveDML``:
"if sigma_hat_n^2 is any consistent estimator of sigma_n^2 -- such as the sieve
sandwich variance"), computed from a full-sample fit by
:class:`opttreat.variance.SieveVariance`, exactly as for the plug-in/LOO rows, so
the three estimators are directly comparable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from opttreat.data import ensure_2d_features, ensure_vector, normalize_treatment, split_treated_control
from opttreat.estimation.splines.spline_factory import build_spline_basis_from_options
from opttreat.variance.sieve_var import sieve_riesz_core


def _make_folds(n: int, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Balanced random fold assignment in ``{0, ..., n_folds-1}`` of length ``n``."""
    base = np.arange(n) % int(n_folds)
    return rng.permutation(base)


def _fit_propensity(
    X_train: np.ndarray,
    D_train: np.ndarray,
    options: Dict[str, Any],
    rcond: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Linear-probability B-spline propensity fit (matches the R DML ``ginv`` regression of D on X)."""
    fmap = build_spline_basis_from_options(options, X_train)
    Psi = np.asarray(fmap(X_train), dtype=float)
    beta = np.linalg.pinv(Psi.T @ Psi, rcond=rcond) @ Psi.T @ D_train

    def p_hat(X: np.ndarray) -> np.ndarray:
        return np.asarray(fmap(X), dtype=float) @ beta

    return p_hat


def dml_welfare_aipw(
    data: Dict[str, np.ndarray],
    estimator: Any,
    param: Any,
    *,
    target_low: tuple,
    target_high: tuple,
    observed_low: tuple,
    observed_high: tuple,
    n_folds: int = 5,
    rng: np.random.Generator | int | None = None,
    propensity_options: Dict[str, Any] | None = None,
    pinv_rcond: float | None = None,
    p_trim: float = 1e-3,
    return_diagnostics: bool = False,
) -> tuple[float, float] | tuple[float, float, dict]:
    """Cross-fitted closed-form (AIPW) debiased welfare estimator, ``E_F[max(h,0)]``.

    This is the equivalent closed-form / Neyman-orthogonal-moment construction of
    Remark ``rem:Welfare_DML`` (eq. ``eq:W_DML_est``) in ``main_260630ZX.tex`` and
    the Python port of the R ``TEST_DML_1D_Spline.R`` welfare DML. Because welfare
    is regular, its influence function is available in closed form,

    .. math::

        \\nu^{*}(x, d) = \\mathbf 1\\{h_0(x)\\ge 0\\}\\, \\lambda(x)
            \\Big(\\tfrac{d}{p_0(x)} - \\tfrac{1-d}{1-p_0(x)}\\Big),
        \\qquad \\lambda = f / f_0,

    so the estimator avoids the sieve-Riesz representer entirely (and its
    finite-sample fragility on folds). For the CCG known-distribution designs the
    target ``F`` is uniform on the target box and the sampling law ``F_0`` is
    uniform on the (larger) observed box, so the density ratio is the constant
    ``lambda = vol(F_0)/vol(F)`` on the target box and ``0`` outside it. The
    cross-fitted estimator is

    .. math::

        \\tilde\\theta = \\frac{1}{K}\\sum_\\kappa W(\\hat\\mu_{-\\kappa})
          + \\frac{1}{n}\\sum_i \\hat\\nu^{*}_{-\\kappa}(X_i, D_i)\\,
            (Y_i - \\hat\\mu_{-\\kappa}(X_i, D_i)),

    with ``W(hat mu_{-kappa})`` the deterministic Sobol integral over ``F``
    (``param.plug_in``), matching the plug-in/LOO rows. The standard error is the
    known-density influence-function variance
    ``sigma_W^2 = E[1{h>=0} lambda^2 sigma_eps^2 / (p (1-p))]`` (no ``Var([h]_+)``
    term, since the welfare integral is deterministic under known ``F``),
    estimated by ``(1/n^2) sum_i 1{h>=0} lambda^2 e_i^2 / (p_hat (1-p_hat))``.

    Parameters
    ----------
    data
        Pooled ``{"X", "Y", "d"}`` sample.
    estimator
        First-stage outcome estimator (refit per fold for ``mu_t``/``mu_c``).
    param
        Welfare parameter whose ``plug_in`` gives ``W(mu)`` over ``F``.
    target_low, target_high, observed_low, observed_high
        Target/observed hyper-rectangles; set ``lambda`` and the target-box
        indicator ``1{x in F}``.
    n_folds, rng
        Cross-fitting folds and RNG for the partition.
    propensity_options
        Spline-basis options for the linear-probability propensity fit. Defaults
        to degree-3, 1-segment tensor B-splines (the R DML choice).
    pinv_rcond
        Singular-value cutoff for the propensity ``ginv`` solve.
    p_trim
        Propensity clip; ``p_hat`` is restricted to ``[p_trim, 1 - p_trim]``.

    Returns
    -------
    (theta_tilde, se)
        Point estimate and standard error. With ``return_diagnostics`` also a
        dict.
    """
    X = ensure_2d_features(data["X"], name="X")
    Y = np.asarray(data["Y"], dtype=float).ravel()
    n = X.shape[0]
    D = normalize_treatment(data["d"], n)

    rcond = float(pinv_rcond if pinv_rcond is not None else np.sqrt(np.finfo(float).eps))
    if propensity_options is None:
        propensity_options = {"J_x_degree": 3, "J_x_segments": 1, "knots": "uniform", "basis": "tensor"}

    lo = np.asarray(target_low, dtype=float)
    hi = np.asarray(target_high, dtype=float)
    vol_target = float(np.prod(hi - lo))
    vol_obs = float(np.prod(np.asarray(observed_high, dtype=float) - np.asarray(observed_low, dtype=float)))
    lam = vol_obs / vol_target  # density ratio f/f_0 on the target box

    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)
    fold_id = _make_folds(n, n_folds, gen)

    welfare_folds: list[float] = []
    correction_sum = 0.0
    var_sum = 0.0
    for k in range(int(n_folds)):
        test = fold_id == k
        train = ~test
        if not test.any() or not train.any():
            continue

        parsed = split_treated_control({"X": X[train], "Y": Y[train], "d": D[train]})
        out_k = estimator.fit(parsed)
        welfare_folds.append(float(param.plug_in(out_k["h_hat"])))

        p_hat = _fit_propensity(X[train], D[train], propensity_options, rcond)

        Xk, Yk, Dk = X[test], Y[test], D[test]
        in_target = np.all((Xk >= lo) & (Xk <= hi), axis=1)  # lambda = 0 outside F
        if not in_target.any():
            continue
        Xk, Yk, Dk = Xk[in_target], Yk[in_target], Dk[in_target]

        h_k = ensure_vector(out_k["h_hat"](Xk), n=Xk.shape[0], name="h_hat")
        mu_t = ensure_vector(out_k["mu_hat_t"](Xk), n=Xk.shape[0], name="mu_t")
        mu_c = ensure_vector(out_k["mu_hat_c"](Xk), n=Xk.shape[0], name="mu_c")
        mu_obs = np.where(Dk == 1, mu_t, mu_c)
        resid = Yk - mu_obs

        p = np.clip(p_hat(Xk), p_trim, 1.0 - p_trim)
        ind = (h_k >= 0.0).astype(float)
        # nu*(X_i, D_i) = 1{h>=0} lambda (D/p - (1-D)/(1-p))
        nu_star = ind * lam * (Dk / p - (1.0 - Dk) / (1.0 - p))
        correction_sum += float(np.sum(nu_star * resid))
        # known-f variance integrand: 1{h>=0} lambda^2 resid^2 / (p (1-p))
        var_sum += float(np.sum(ind * lam**2 * resid**2 / (p * (1.0 - p))))

    welfare = float(np.mean(welfare_folds)) if welfare_folds else float("nan")
    theta_tilde = welfare + correction_sum / n
    se = float(np.sqrt(max(var_sum / n**2, 0.0)))

    if return_diagnostics:
        return theta_tilde, se, {
            "welfare_folds": welfare_folds,
            "correction": correction_sum / n,
            "lambda": lam,
        }
    return theta_tilde, se


def dml_sieve_estimate(
    data: Dict[str, np.ndarray],
    estimator: Any,
    param: Any,
    variance_options: Dict[str, Any],
    *,
    n_folds: int = 5,
    rng: np.random.Generator | int | None = None,
    trim_support: bool = True,
    riesz_rcond: float | None = 5e-3,
    riesz_ridge: float = 0.0,
    riesz_cap_mult: float | None = 10.0,
    return_diagnostics: bool = False,
) -> tuple[float, float] | tuple[float, float, dict]:
    """Cross-fitted sieve-debiased (DML) estimate of ``Theta(mu_0)`` and its SE.

    Parameters
    ----------
    data
        Pooled sample as ``{"X": (n, d), "Y": (n,), "d": (n,)}``.
    estimator
        A first-stage :class:`~opttreat.estimation.base.Estimator` (e.g. the
        B-spline sieve). It is refit on each training fold; only its ``options``
        matter across folds, so a single instance is reused.
    param
        The target :class:`~opttreat.parameters.base.Parameter` whose
        ``plug_in(h_hat)`` returns ``Theta(mu_hat)`` integrated over the known
        ``F`` (welfare ``E_F[max(h,0)]`` or value ``E_F[1{h>0} m]``).
    variance_options
        Options driving the sieve Riesz representer (the same dict passed to
        :class:`~opttreat.variance.SieveVariance`): ``param_type``, ``dim``,
        ``n_sobol``, ``transform``, ``v_func``/``f_func``, ``loo_eps``/``delta``,
        ``solver``/``pinv_rcond``, etc. The Riesz derivative ``Bun`` is built on
        each training fold's fitted ``h_hat`` and integrated over ``F``.
    n_folds
        Number of cross-fitting folds ``K`` (default 5, matching the R DML code).
    rng
        Seed or generator for the fold partition.
    trim_support
        If True (default), restrict each held-out fold's correction to the
        treated/control **common support of the training fold** before evaluating
        the representer. This mirrors the ``df_test_trim`` step of the reference R
        DML code and is essential: the per-arm B-spline basis is anchored on the
        training fold's covariate range, so a held-out point beyond that range
        makes the (cubic) sieve extrapolate wildly and the representer explode.
        Out-of-support points contribute ~0 in expectation (mean-zero residual),
        so they are dropped from the correction while the sum is still divided by
        the full ``n``.
    riesz_rcond
        Optional singular-value cutoff override for the representer ``pinv`` solve
        (see ``sieve_riesz_core``); pinv truncation of the ill-conditioned fold
        Gram.
    riesz_ridge
        Tikhonov floor added to the (normalized) representer Gram, ``G_a +
        riesz_ridge I``. A *global* penalty: it stabilizes the rare near-singular
        fold but also shrinks every well-conditioned fold, which deflates the SE.
        Default 0.0 (off) -- prefer ``riesz_cap_mult`` instead.
    riesz_cap_mult
        Per-fold representer-norm cap (default 10.0). The blow-up from a
        near-singular fold Gram is rare (perhaps one fold in hundreds), so rather
        than penalize every fold, this rescales arm ``a`` in fold ``kappa`` by
        ``min(1, cap_mult * median_k ||w_{a,k}|| / ||w_{a,kappa}||)`` -- it only
        touches folds whose representer norm is a gross outlier and leaves the
        (unshrunk) representer intact everywhere else, so the score-variance SE
        stays honest. Set to ``None`` to disable.
    return_diagnostics
        If True, also return a dict with per-fold plug-ins and the correction.

    Returns
    -------
    (theta_tilde, se)
        The cross-fitted sieve-debiased estimate and its standard error. The SE
        is the cross-fitted score standard deviation
        ``sqrt( sum_i (v*_{-k(i)}(X_i,D_i) e_i)^2 ) / n`` -- the sample analog of
        ``sigma_n / sqrt(n)`` (Theorem thm:SieveDML) using *out-of-fold*
        residuals, so it captures the debiasing variance the in-sample SieveVar
        misses. With ``return_diagnostics`` also returns a dict.
    """
    X = ensure_2d_features(data["X"], name="X")
    Y = np.asarray(data["Y"], dtype=float).ravel()
    n = X.shape[0]
    D = normalize_treatment(data["d"], n)

    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)
    fold_id = _make_folds(n, n_folds, gen)

    # The Riesz representer inverts the (fold) arm Gram; on a training fold this
    # can be far more ill-conditioned than on the full sample, so regularize the
    # representer solve only. riesz_rcond forces a relative-truncation pinv (basis-
    # agnostic: works for the B-spline sieve and random-feature ridge alike);
    # riesz_ridge is an optional absolute floor.
    riesz_options = dict(variance_options)
    if riesz_rcond is not None:
        riesz_options["riesz_rcond"] = float(riesz_rcond)
    if riesz_ridge:
        riesz_options["riesz_ridge"] = float(riesz_ridge)

    # Pass 1: fit each fold, form the held-out per-observation scores
    # psi_i = v*_{-kappa}(X_i, D_i) e_i, and record the arm representer norms so
    # a rare blow-up fold can be capped without shrinking the others.
    theta_plug_folds: list[float] = []
    fold_scores: list[dict] = []
    norms_t: list[float] = []
    norms_c: list[float] = []
    for k in range(int(n_folds)):
        test = fold_id == k
        train = ~test
        if not test.any() or not train.any():
            continue

        parsed = split_treated_control({"X": X[train], "Y": Y[train], "d": D[train]})
        out_k = estimator.fit(parsed)

        # Theta(mu_hat_{-kappa}) integrated over the known target F.
        theta_plug_folds.append(float(param.plug_in(out_k["h_hat"])))

        # Sieve Riesz representer w_a = G_a^{-1} Bun_a on the training fold.
        core = sieve_riesz_core(out_k, riesz_options)
        n_t_tr, n_c_tr = core["n_t"], core["n_c"]
        n_tr = n_t_tr + n_c_tr
        beta_t = np.asarray(out_k["beta_t"], dtype=float).ravel()
        beta_c = np.asarray(out_k["beta_c"], dtype=float).ravel()

        Xk, Yk, Dk = X[test], Y[test], D[test]
        if trim_support:
            lo = np.maximum(parsed["X_t"].min(axis=0), parsed["X_c"].min(axis=0))
            hi = np.minimum(parsed["X_t"].max(axis=0), parsed["X_c"].max(axis=0))
            in_support = np.all((Xk > lo) & (Xk < hi), axis=1)
            Xk, Yk, Dk = Xk[in_support], Yk[in_support], Dk[in_support]

        entry: dict = {}
        mask_t = Dk == 1
        if mask_t.any():
            Psi_t_test = np.asarray(core["feature_map_t"](Xk[mask_t]), dtype=float)
            e_t_test = Yk[mask_t] - Psi_t_test @ beta_t
            # v*(X_i, 1) = (n_tr / n_t) psi_t(X_i)^T w_t
            v_star_t = (n_tr / n_t_tr) * (Psi_t_test @ core["weights_t"])
            entry["psi_t"] = v_star_t * e_t_test  # per-observation influence score
            norm_t = float(np.linalg.norm(core["weights_t"]))
            entry["norm_t"] = norm_t
            norms_t.append(norm_t)
        mask_c = Dk == 0
        if mask_c.any():
            Psi_c_test = np.asarray(core["feature_map_c"](Xk[mask_c]), dtype=float)
            e_c_test = Yk[mask_c] - Psi_c_test @ beta_c
            # v*(X_i, 0) = (n_tr / n_c) psi_c(X_i)^T w_c (w_c carries the control sign)
            v_star_c = (n_tr / n_c_tr) * (Psi_c_test @ core["weights_c"])
            entry["psi_c"] = v_star_c * e_c_test
            norm_c = float(np.linalg.norm(core["weights_c"]))
            entry["norm_c"] = norm_c
            norms_c.append(norm_c)
        fold_scores.append(entry)

    # Pass 2: cap the rare outlier fold's representer, then accumulate.
    cap_t = float(riesz_cap_mult) * float(np.median(norms_t)) if (riesz_cap_mult and norms_t) else np.inf
    cap_c = float(riesz_cap_mult) * float(np.median(norms_c)) if (riesz_cap_mult and norms_c) else np.inf
    correction_sum = 0.0
    score_sq_sum = 0.0
    n_capped = 0
    for entry in fold_scores:
        for arm, cap in (("t", cap_t), ("c", cap_c)):
            psi = entry.get(f"psi_{arm}")
            if psi is None:
                continue
            norm = entry[f"norm_{arm}"]
            if np.isfinite(cap) and norm > cap:
                psi = psi * (cap / norm)  # rescale only the blow-up fold's arm
                n_capped += 1
            correction_sum += float(np.sum(psi))
            score_sq_sum += float(np.sum(psi ** 2))

    theta_plug = float(np.mean(theta_plug_folds)) if theta_plug_folds else float("nan")
    correction = correction_sum / n
    theta_tilde = theta_plug + correction
    se = float(np.sqrt(max(score_sq_sum, 0.0)) / n)

    if return_diagnostics:
        return theta_tilde, se, {
            "theta_plug_folds": theta_plug_folds,
            "theta_plug_mean": theta_plug,
            "correction": correction,
            "n_capped": int(n_capped),
            "n_folds": int(n_folds),
        }
    return theta_tilde, se
