"""DenseIndexDGP known-distribution welfare simulation, RF first stage.

Self-contained runner in the style of the CCG 2025
``ccg_welfare_known/run_ccg_welfare_known.py`` script: the DGP, the RF-sieve
tuning, and the simulation settings are top-level constants, and the Monte Carlo
loop is spelled out in ``run``. The random-feature sieve map is built by the
``rf_ridge`` estimator directly from the quasi-sphere options, so there is no
custom feature-map helper; the first-stage estimator, the target parameter, and
the SieveVar standard error come straight from ``opttreat.estimation``,
``opttreat.parameters`` and ``opttreat.variance``.

Configuration is the best-coverage DenseIndexDGP welfare design from the existing
simulation results: 200 quasi-sphere sigmoid random features, with the
leave-one-out debiased estimator reaching ~0.95 coverage at the largest sample
size. Each run reports, for every sample size, both the plug-in estimate and the
LOO-debiased estimate with SieveVar inference (standard error and 95% coverage).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opttreat.config import EstimatorConfig, ParameterConfig, VarianceConfig
from opttreat.data import split_treated_control
from opttreat.estimation import get_estimator
from opttreat.models.dense_index_model import DenseIndexDGP
from opttreat.parameters import get_parameter
from opttreat.variance import get_variance_estimator


Z = 1.959963984540054
PARAMETER_TYPE = "welfare_known"

# ---------------- Simulation settings ----------------
DGP_DIM = 50
ITE = 2000
NS = (1500, 3000, 6000, 12000)
SEED = 2026
JOBS = 4
LOO_DELTA0 = 0.05

# ---------------- RF-sieve first stage (best welfare design) ----------------
N_FEATURES = 200
RFG_TYPE = "quasi_sphere"
ACTIVATION = "sigmoid"
SOLVER = "ridge"          # "pinv" (Moore-Penrose OLS) or "ridge"
ALPHA = 0.0005             # ridge penalty; used only when SOLVER == "ridge" (ignored under "pinv")
PINV_RCOND = 1e-10
THETA_SOBOL = 8192
VARIANCE_SOBOL = 32768

STEM = "dense_index_welfare_known"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def run(ns=NS, ite=ITE, *, jobs=JOBS, output_dir=OUTPUT_DIR):
    model = DenseIndexDGP(dim=DGP_DIM)
    estimator = get_estimator(EstimatorConfig(method="rf_ridge", options={
        "rfg_type": RFG_TYPE,
        "activation": ACTIVATION,
        "share_features": True,
        "n_features": N_FEATURES,
        "random_state": SEED,
        "alpha": ALPHA,
        "solver": SOLVER,
        "pinv_rcond": PINV_RCOND,
    }))
    param = get_parameter(ParameterConfig(PARAMETER_TYPE, options={
        "dim": DGP_DIM,
        "n_sobol": THETA_SOBOL,
        "sobol_scramble": False,
        "pinv_rcond": PINV_RCOND,
    }))
    variance = get_variance_estimator(VarianceConfig(method="sieve_var", options={
        "param_type": PARAMETER_TYPE,
        "dim": DGP_DIM,
        "n_sobol": VARIANCE_SOBOL,
        "sobol_scramble": False,
        "alpha": ALPHA,
        "solver": SOLVER,
        "pinv_rcond": PINV_RCOND,
    }))
    W_true = float(param.get_true_value(model))

    draw_rows = []
    for n_idx, n in enumerate(ns):
        print(f"Running {STEM} | rf_ridge | n = {n} | ite = {ite}", flush=True)

        def one_ite(seed_i, n=n):
            np.random.seed(seed_i)
            df = model.generate_data(n)
            output = estimator.fit(split_treated_control(df))
            W_plug = float(param.plug_in(output["h_hat"]))
            W_loo = float(param.loo(output, delta0=LOO_DELTA0))
            se = float(np.sqrt(max(variance.fit(output), 0.0)))
            return W_plug, W_loo, se

        base_seed = SEED + 10_000 * n_idx
        if jobs <= 1:
            results = [one_ite(base_seed + r) for r in range(ite)]
        else:
            from joblib import Parallel, delayed

            temp = Path.cwd() / ".joblib_tmp"
            temp.mkdir(exist_ok=True)
            results = Parallel(n_jobs=jobs, temp_folder=str(temp), max_nbytes=None)(
                delayed(one_ite)(base_seed + r) for r in range(ite)
            )

        for rep, (W_plug, W_loo, se) in enumerate(results):
            draw_rows.append({"spec": "dense_index", "n": n, "estimator": "plug_in", "rep": rep, "W_hat": W_plug, "W_true": W_true, "se": se})
            draw_rows.append({"spec": "dense_index", "n": n, "estimator": "loo", "rep": rep, "W_hat": W_loo, "W_true": W_true, "se": se})

    draws = pd.DataFrame(draw_rows)
    summary = _summarize(draws)

    suffix = _suffix(ns, ite)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{STEM}_summary_{suffix}.csv", index=False)
    draws.to_csv(output_dir / f"{STEM}_draws_{suffix}.csv", index=False)
    print(summary.to_string(index=False))
    return summary, draws


def _summarize(draws):
    rows = []
    for (spec_name, n, estimator_name), g in draws.groupby(["spec", "n", "estimator"], sort=False):
        W = g["W_hat"].to_numpy(dtype=float)
        se = g["se"].to_numpy(dtype=float)
        W_true = float(g["W_true"].iloc[0])
        rows.append({
            "spec": spec_name,
            "n": int(n),
            "estimator": estimator_name,
            "W_true": W_true,
            "W_mean": float(W.mean()),
            "bias": float((W - W_true).mean()),
            "sd": float(W.std(ddof=1)) if W.size > 1 else 0.0,
            "se": float(se.mean()),
            "sd_se": float(se.std(ddof=1)) if se.size > 1 else 0.0,
            "coverage": float(((W - Z * se <= W_true) & (W + Z * se >= W_true)).mean()),
            "replications": int(W.size),
        })
    return pd.DataFrame(rows)


def _suffix(ns, ite):
    return (
        f"n{'_'.join(str(n) for n in ns)}_rep{ite}_d{DGP_DIM}_nf{N_FEATURES}"
        f"_vs{VARIANCE_SOBOL}_{RFG_TYPE}_{ACTIVATION}"
        f"_loo{str(LOO_DELTA0).replace('.', 'p')}"
    )


def main():
    return run()


def smoke_main(output_dir=OUTPUT_DIR):
    """Tiny local check (small dim, few reps, small Sobol)."""
    global DGP_DIM, N_FEATURES, THETA_SOBOL, VARIANCE_SOBOL
    DGP_DIM, N_FEATURES, THETA_SOBOL, VARIANCE_SOBOL = 4, 12, 64, 64
    return run(ns=(120,), ite=2, jobs=1, output_dir=output_dir)


if __name__ == "__main__":
    main()
