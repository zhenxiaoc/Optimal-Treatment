"""KT empirical welfare/value estimators, in the simulation-runner style.

Python ports of the five ``TEST_Emp_*`` R scripts for the KT data set
(``KT_Data1.csv``), written like the ``DenseIndexDGP`` runners: the data prep,
spline tuning, and settings are module-level, and each estimator is assembled
from the shared OptTreat components rather than bespoke helpers --

* the treated/control conditional means come from the ``sieve`` estimator
  (:class:`opttreat.estimation.SieveEstimator`) with the generalized-inverse
  ("pinv") solve and ``extrapolate=True`` boundary behaviour that matches R's
  ``ginv`` OLS on ``crs::gsl.bs`` tensor splines;
* the welfare/value point estimates come from
  :class:`~opttreat.parameters.WelfareUnknownDist` /
  :class:`~opttreat.parameters.ValueUnknownDist`;
* the sieve standard errors come from
  :class:`~opttreat.variance.SieveVariance`.

Reproduction against the committed R results: the three welfare scripts match to
floating-point precision; the value scripts match ``V_hat``, ``eps`` and ``num``
exactly. The value standard error uses a pure-Python exact Gaussian KDE for the
covariate density weight rather than R's binned ``ks::Hscv`` density. The default
bandwidth is the data-driven Silverman normal reference (``DENSITY_BW_METHOD``,
``SMOOTHING=1``): a bandwidth study (``results/density_method_comparison.csv``,
``density_comparison.png``, ``density_se_sensitivity.png``) showed that likelihood
cross-validation undersmooths on these semi-discrete covariates while R's
``Hscv``*3 / the old ``SMOOTHING=3`` oversmooths, and that the value SE is flat
across the sensible bandwidth range, so this choice best-calibrates the density
with a negligible SE change.

Run as a module::

    python -m opttreat.empirical.run_empirical
    python -m opttreat.empirical.run_empirical --out some/dir --M 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from opttreat.config import EstimatorConfig, ParameterConfig, VarianceConfig
from opttreat.data import common_support_mask, split_treated_control, trimmed_std
from opttreat.estimation import get_estimator
from opttreat.parameters import get_parameter
from opttreat.variance import get_variance_estimator

Z = 1.96
DATA_FILE = Path(__file__).with_name("KT_Data1.csv")
OUTPUT_DIR = Path(__file__).with_name("results")

# ---------------- Configuration shared by every TEST_Emp_* script ----------------
COST_PER_TREATED = 774.0           # average per-assignment service cost
DIM = 2                            # covariates: (prevearn, edu)
RCOND = float(np.sqrt(np.finfo(float).eps))   # MASS::ginv singular-value cutoff
ETA = 0.01                         # value band half-width: eps = ETA * sd(h)
# Covariate-density bandwidth for the value-functional SE weight. `bw_method`
# picks the (full-covariance, anisotropic) gaussian_kde bandwidth; `SMOOTHING`
# multiplies the *variance*, i.e. the bandwidth is scaled by sqrt(SMOOTHING).
# A bandwidth exploration (see results/density_method_comparison.csv and
# density_comparison.png) found that likelihood cross-validation *undersmooths*
# here -- the covariates are semi-discrete (integer `edu`, a large `prevearn`=0
# mass), so CV chases those spikes -- while the previous SMOOTHING=3 (the R
# `Hscv`*3 convention) *oversmooths*. The data-driven Silverman normal reference
# with no extra oversmoothing (SMOOTHING=1) sits on the value-SE plateau (SE
# changes by <0.005) yet gives a far better-calibrated density, so it is the
# recommended default.
DENSITY_BW_METHOD = "silverman"
SMOOTHING = 1.0
N_SOBOL = 1_000_000                # Sobol points for the value-functional SE
PS_SEGMENTS = 8                    # propensity sieve dimension (plug-in welfare SE)

# Degree-3 uniform-knot tensor sieve, group-specific dimensions, ginv ("pinv") OLS.
SIEVE_OPTIONS = {
    "share_features": False,
    "J_x_degree": 3,
    "J_x_segments_t": 4,
    "J_x_segments_c": 1,
    "knots": "uniform",
    "basis": "tensor",
    "solver": "pinv",
    "pinv_rcond": RCOND,
    "extrapolate": True,
}


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------
def _row(cost: bool, estimate_name: str, estimate: float, se: float, extra: dict | None = None) -> dict:
    """Cost-labelled report row with the symmetric normal confidence interval."""
    row = {
        "cost": cost,
        "cost_label": "cost_774" if cost else "no_cost",
        estimate_name: estimate,
        "SE": se,
        "CI_low": estimate - Z * se,
        "CI_high": estimate + Z * se,
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# TEST_Emp_Welfare_Spline_PlugIn.R
# ---------------------------------------------------------------------------
def welfare_plugin(cost: bool, M: int = N_SOBOL) -> dict:
    """Plug-in welfare with the propensity-weighted influence-function SE."""
    df = pd.read_csv(DATA_FILE)
    D = df["D"].to_numpy(float)
    Y = df["earnings"].to_numpy(float)
    if cost:
        Y = Y - D * COST_PER_TREATED
    X = df[["prevearn", "edu"]].to_numpy(float)
    mask = common_support_mask(X, D, strict=True)

    estimator = get_estimator(EstimatorConfig("sieve", dict(SIEVE_OPTIONS)))
    estimator.fit(split_treated_control({"X": X, "Y": Y, "d": D}))
    output = estimator.get_output()
    X_trim = X[mask]

    parameter = get_parameter(ParameterConfig("welfare_unknown", {"dim": DIM}))
    W_hat = float(parameter.plug_in(output["h_hat"], X_trim))

    variance = get_variance_estimator(VarianceConfig("welfare_plugin", {
        "param_type": "welfare_unknown",
        "propensity_options": {"J_x_degree": 3, "J_x_segments": PS_SEGMENTS,
                               "knots": "uniform", "basis": "tensor", "extrapolate": True},
        "pinv_rcond": RCOND,
    }))
    output_for_variance = {**output, "X_eval": X_trim, "X_all": X, "D_all": D,
                           "D_eval": D[mask], "Y_eval": Y[mask]}
    se = float(np.sqrt(max(variance.fit(output_for_variance), 0.0)))
    return _row(cost, "W_hat", W_hat, se)


# ---------------------------------------------------------------------------
# TEST_Emp_Welfare_Spline_SieveVar_{Notrim,Trim}.R
# ---------------------------------------------------------------------------
def welfare_sievevar(cost: bool, trim: bool, M: int = N_SOBOL) -> dict:
    """Welfare with the sieve influence-function SE, over the full or trimmed support."""
    df = pd.read_csv(DATA_FILE)
    D = df["D"].to_numpy(float)
    Y = df["earnings"].to_numpy(float)
    if cost:
        Y = Y - D * COST_PER_TREATED
    X = df[["prevearn", "edu"]].to_numpy(float)
    mask = common_support_mask(X, D, strict=True)

    estimator = get_estimator(EstimatorConfig("sieve", dict(SIEVE_OPTIONS)))
    estimator.fit(split_treated_control({"X": X, "Y": Y, "d": D}))
    output = estimator.get_output()
    support = X[mask] if trim else X

    parameter = get_parameter(ParameterConfig("welfare_unknown", {"dim": DIM}))
    W_hat = float(parameter.plug_in(output["h_hat"], support))

    variance = get_variance_estimator(VarianceConfig("sieve_var", {
        "param_type": "welfare_unknown",
        "dim": DIM,
    }))
    output_for_variance = {**output, "X_eval": support}
    se = float(np.sqrt(max(variance.fit(output_for_variance), 0.0)))
    return _row(cost, "W_hat", W_hat, se)


# ---------------------------------------------------------------------------
# TEST_Emp_Value_Spline_SieveVar_{NoTrim,Trim}.R
# ---------------------------------------------------------------------------
def value_sievevar(cost: bool, trim: bool, M: int = N_SOBOL) -> dict:
    """Value functional (v0 = 1) with the sieve boundary-band SE.

    ``V_hat``, ``eps`` and ``num`` reproduce R; the SE uses an exact Gaussian KDE
    (Silverman bandwidth times ``SMOOTHING``) as the covariate-density weight.
    """
    df = pd.read_csv(DATA_FILE)
    D = df["D"].to_numpy(float)
    Y = df["earnings"].to_numpy(float)
    if cost:
        Y = Y - D * COST_PER_TREATED
    X = df[["prevearn", "edu"]].to_numpy(float)
    mask = common_support_mask(X, D, strict=True)

    estimator = get_estimator(EstimatorConfig("sieve", dict(SIEVE_OPTIONS)))
    estimator.fit(split_treated_control({"X": X, "Y": Y, "d": D}))
    output = estimator.get_output()
    support = X[mask] if trim else X
    h_hat = output["h_hat"]

    # Value function v0(x) = 1 over the support. The plug-in averages over the
    # observed rows, which already follow the data distribution, so the density
    # enters implicitly and m = v0 here.
    v0 = lambda points: np.ones(np.asarray(points, dtype=float).shape[0], dtype=float)
    parameter = get_parameter(ParameterConfig("value_unknown", {"dim": DIM, "v_func": v0}))
    V_hat = float(parameter.plug_in(h_hat, support))

    eps = ETA * trimmed_std(np.asarray(h_hat(support), dtype=float))

    # Covariate density f(x), an exact Gaussian KDE Radon-Nikodym-rescaled by the
    # box area, for the Monte Carlo integral of the boundary-band variance. The
    # variance weight is m(x) = v0(x) * f(x), supplied as the two factors below.
    lo, hi = support.min(axis=0), support.max(axis=0)
    area = float(np.prod(hi - lo))
    kde = gaussian_kde(support.T, bw_method=DENSITY_BW_METHOD)
    kde.set_bandwidth(kde.factor * np.sqrt(SMOOTHING))
    f_density = lambda points: kde(np.asarray(points, dtype=float).T) * area

    variance = get_variance_estimator(VarianceConfig("sieve_var", {
        "param_type": "value_known",
        "dim": DIM,
        "n_sobol": M,
        "sobol_scramble": False,
        "transform": lambda U: lo + U * (hi - lo),
        "v_func": v0,
        "f_func": f_density,
        "loo_eps": eps,
        "variance_expand_sobol": False,
        "solver": "pinv",
        "pinv_rcond": RCOND,
        "alpha": 0.0,
    }))
    se = float(np.sqrt(max(variance.fit(output), 0.0)))
    num = int(variance.diagnostics_["n_band"])
    return _row(cost, "V_hat", V_hat, se, extra={"eps": float(eps), "num": num})


# ---------------------------------------------------------------------------
# Runner: every script x {no-cost, cost-774} -> result CSVs (matching R names)
# ---------------------------------------------------------------------------
SPECS = [
    ("TEST_Emp_Welfare_Spline_PlugIn_results.csv", lambda cost, M: welfare_plugin(cost, M)),
    ("TEST_Emp_Welfare_Spline_SieveVar_Notrim_results.csv", lambda cost, M: welfare_sievevar(cost, trim=False, M=M)),
    ("TEST_Emp_Welfare_Spline_SieveVar_Trim_results.csv", lambda cost, M: welfare_sievevar(cost, trim=True, M=M)),
    ("TEST_Emp_Value_Spline_SieveVar_NoTrim_results.csv", lambda cost, M: value_sievevar(cost, trim=False, M=M)),
    ("TEST_Emp_Value_Spline_SieveVar_Trim_results.csv", lambda cost, M: value_sievevar(cost, trim=True, M=M)),
]


def run_all(output_dir: Path | str | None = OUTPUT_DIR, *, M: int = N_SOBOL, verbose: bool = True) -> dict:
    """Run every estimator for both cost settings, writing one CSV per R script."""
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for filename, estimator in SPECS:
        frame = pd.DataFrame([estimator(False, M), estimator(True, M)])
        results[filename] = frame
        if out is not None:
            frame.to_csv(out / filename, index=False)
        if verbose:
            print(f"=== {filename} ===")
            print(frame.to_string(index=False))
            print()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KT empirical estimators.")
    parser.add_argument("--out", default=str(OUTPUT_DIR), help="output directory for the result CSVs")
    parser.add_argument("--M", type=int, default=N_SOBOL, help="Sobol points for the value-functional SE")
    args = parser.parse_args()
    run_all(args.out, M=args.M)


if __name__ == "__main__":
    main()
