"""CCG 2025 Theorem 1 known-distribution welfare simulation (Models 1-7).

Self-contained runner in the style of the R ``TEST_Thm1_*_Spline_SieveVar.R``
scripts: the model list and per-model sieve dimensions are inline, the
simulation settings are top-level constants, and the Monte Carlo loop is spelled
out below. The first-stage estimator, the target parameter, and the SieveVar
standard error come straight from ``opttreat.estimation``, ``opttreat.parameters``
and ``opttreat.variance``.

Set ``ESTIMATOR`` to ``"sieve"`` (B-spline sieve, pseudo-inverse OLS = R ginv) or
``"rf_ridge"`` (random-feature ridge). Both estimators return the same
estimator-output structure, so the plug-in / LOO / SieveVar code is identical.
The LOO debiasing defaults to the analytical boundary-band method
(``LOO_METHOD = "band"``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opttreat.config import EstimatorConfig, ParameterConfig, VarianceConfig
from opttreat.data import split_treated_control
from opttreat.estimation import get_estimator
from opttreat.estimation.dml_sieve import dml_sieve_estimate
from opttreat.models import CCGModel
from opttreat.parameters import get_parameter
from opttreat.variance import get_variance_estimator


PINV_RCOND = float(np.sqrt(np.finfo(float).eps))
Z = 1.96

# ---------------- Models and their sieve dimensions (Jt, Jc) ----------------
SPECS = [
    {"model": "Model1", "Jt": 16, "Jc": 16},
    {"model": "Model2", "Jt": 1, "Jc": 1},
    {"model": "Model3", "Jt": 1, "Jc": 1},
    {"model": "Model4", "Jt": 1, "Jc": 1},
    {"model": "Model5", "Jt": 1, "Jc": 1},
    {"model": "Model6", "Jt": 1, "Jc": 4},
    {"model": "Model7", "Jt": 1, "Jc": 4},
]

# ---------------- Simulation settings (unchanged from the paper run) ----------------
ITE = 2000
NS = (1500, 3000, 6000)
THETA_SOBOL = 5000
VARIANCE_SOBOL = 40000
SEED = 2025
JOBS = 3
LOO_DELTA0 = 0.05
LOO_METHOD = "band"  # analytical debiasing (alternative: "central_difference")
# Cross-fitted sieve-Riesz DML (main_260630ZX.tex Section sec:SieveDML, eq:debias_est).
DML_FOLDS = 5
DML_RIESZ_RCOND = 5e-3  # relative-truncation floor for the per-fold representer solve

# ---------------- First-stage estimator: "sieve" or "rf_ridge" ----------------
ESTIMATOR = "sieve"
# B-spline sieve tuning (per-model Jt/Jc above)
SPLINE_DEGREE, KNOTS, BASIS = 3, "uniform", "tensor"
# Random-feature tuning (used when ESTIMATOR == "rf_ridge")
RF_N_FEATURES = 100
RF_RFG_TYPE = "iid_sphere"
RF_ACTIVATION = "exp"
RF_ALPHA = 1e-5

STEM = "ccg_welfare_known"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def run(specs=SPECS, ns=NS, ite=ITE, *, jobs=JOBS, theta_sobol=THETA_SOBOL,
        variance_sobol=VARIANCE_SOBOL, output_dir=OUTPUT_DIR):
    draw_rows = []
    for s_idx, spec in enumerate(specs):
        model = CCGModel(spec["model"])

        # First-stage estimator: B-spline sieve (pseudo-inverse OLS = R ginv) or
        # random-feature ridge. Both return the same estimator-output structure.
        if ESTIMATOR == "sieve":
            estimator = get_estimator(EstimatorConfig(method="sieve", options={
                "solver": "pinv",
                "share_features": False,
                "J_x_degree": SPLINE_DEGREE,
                "J_x_segments_t": spec["Jt"],
                "J_x_segments_c": spec["Jc"],
                "knots": KNOTS,
                "basis": BASIS,
                "X_min": None,
                "X_max": None,
                "pinv_rcond": PINV_RCOND,
            }))
        elif ESTIMATOR == "rf_ridge":
            estimator = get_estimator(EstimatorConfig(method="rf_ridge", options={
                "rfg_type": RF_RFG_TYPE,
                "activation": RF_ACTIVATION,
                "share_features": False,
                "n_features": RF_N_FEATURES,
                "random_state": SEED,
                "alpha": RF_ALPHA,
            }))
        else:
            raise ValueError(f"Unknown ESTIMATOR {ESTIMATOR!r}; use 'sieve' or 'rf_ridge'.")

        # Target parameter: welfare functional E[max(h(X), 0)] (known dist).
        param = get_parameter(ParameterConfig("welfare_known", options={
            "dim": model.dim,
            "n_sobol": int(theta_sobol),
            "transform": model.inverse_CDF,
            "sobol_scramble": False,
            "loo_method": LOO_METHOD,
            "pinv_rcond": PINV_RCOND,
        }))

        # SieveVar standard error for the known-distribution welfare functional.
        variance_options = {
            "param_type": "welfare_known",
            "dim": model.dim,
            "n_sobol": int(variance_sobol),
            "transform": model.inverse_CDF,
            "sobol_scramble": False,
            "pinv_rcond": PINV_RCOND,
        }
        variance = get_variance_estimator(VarianceConfig(method="sieve_var", options=variance_options))

        W_true = float(param.get_true_value(model))

        for n_idx, n in enumerate(ns):
            print(f"Running {spec['model']} | {ESTIMATOR} | n = {n} | ite = {ite}", flush=True)

            def one_ite(seed_i, n=n):
                np.random.seed(seed_i)
                df = model.generate_data(n)
                output = estimator.fit(split_treated_control(df))
                W_plug = float(param.plug_in(output["h_hat"]))
                W_loo = float(param.loo(output, delta0=LOO_DELTA0))
                se = float(np.sqrt(max(variance.fit(output), 0.0)))
                # Cross-fitted sieve-Riesz DML (its own cross-fitted score-variance SE).
                data = {"X": df[model.feature_columns].to_numpy(),
                        "Y": df["y"].to_numpy(), "d": df["d"].to_numpy()}
                W_dml, se_dml = dml_sieve_estimate(
                    data, estimator, param, variance_options,
                    n_folds=DML_FOLDS, rng=seed_i, riesz_rcond=DML_RIESZ_RCOND)
                return W_plug, W_loo, se, W_dml, se_dml

            base_seed = SEED + 100_000 * s_idx + 10_000 * n_idx
            if jobs <= 1:
                results = [one_ite(base_seed + r) for r in range(ite)]
            else:
                from joblib import Parallel, delayed

                temp = Path.cwd() / ".joblib_tmp"
                temp.mkdir(exist_ok=True)
                results = Parallel(n_jobs=jobs, temp_folder=str(temp), max_nbytes=None)(
                    delayed(one_ite)(base_seed + r) for r in range(ite)
                )

            for rep, (W_plug, W_loo, se, W_dml, se_dml) in enumerate(results):
                draw_rows.append({"spec": spec["model"], "n": n, "estimator": "plug_in", "rep": rep, "W_hat": W_plug, "W_true": W_true, "se": se})
                draw_rows.append({"spec": spec["model"], "n": n, "estimator": "loo", "rep": rep, "W_hat": W_loo, "W_true": W_true, "se": se})
                draw_rows.append({"spec": spec["model"], "n": n, "estimator": "dml", "rep": rep, "W_hat": W_dml, "W_true": W_true, "se": se_dml})

    draws = pd.DataFrame(draw_rows)
    summary_rows = []
    for (spec_name, n, estimator_name), g in draws.groupby(["spec", "n", "estimator"], sort=False):
        W = g["W_hat"].to_numpy(dtype=float)
        se = g["se"].to_numpy(dtype=float)
        W_true = float(g["W_true"].iloc[0])
        summary_rows.append({
            "spec": spec_name,
            "n": int(n),
            "estimator": estimator_name,
            "W_true": W_true,
            "bias": float((W - W_true).mean()),
            "sd": float(W.std(ddof=1)) if W.size > 1 else 0.0,
            "se": float(se.mean()),
            "sd_se": float(se.std(ddof=1)) if se.size > 1 else 0.0,
            "coverage": float(((W - Z * se <= W_true) & (W + Z * se >= W_true)).mean()),
            "replications": int(W.size),
        })
    summary = pd.DataFrame(summary_rows)

    suffix = f"{ESTIMATOR}_n{'_'.join(str(n) for n in ns)}_rep{ite}_loo{str(LOO_DELTA0).replace('.', 'p')}_{LOO_METHOD}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{STEM}_summary_{suffix}.csv", index=False)
    draws.to_csv(output_dir / f"{STEM}_draws_{suffix}.csv", index=False)
    print(summary.to_string(index=False))
    return summary, draws


def main():
    return run()


def smoke_main(output_dir=OUTPUT_DIR):
    """Tiny local check over a single model in this group."""
    return run(specs=[SPECS[0]], ns=(90,), ite=1, jobs=1, theta_sobol=512, variance_sobol=32, output_dir=output_dir)


if __name__ == "__main__":
    main()
