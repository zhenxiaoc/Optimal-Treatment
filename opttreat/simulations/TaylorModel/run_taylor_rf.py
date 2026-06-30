"""TaylorExpansionModel known-distribution welfare simulation, RF first stage.

Self-contained runner in the style of the CCG 2025 and DenseIndexDGP scripts: the
DGP specs, the RF-sieve tuning, and the simulation settings are top-level
constants, and the Monte Carlo loop is spelled out in ``run``. The first-stage
estimator, the target parameter, and the SieveVar standard error come straight
from ``opttreat.estimation``, ``opttreat.parameters`` and ``opttreat.variance``;
the RF feature map is built by the ``rf_ridge`` estimator from the iid-sphere
options. There is no shared simulation engine.

Configuration is the best-results TaylorExpansionModel design from the existing
estimation-only simulation results: the ``hyperbolic`` expansion, whose RF welfare
estimator has the smallest, most stable bias across the Taylor order K (the
``rational`` expansion blows up as K grows, and ``tan2``/``sinh2``/``exp_pm`` pick
up bias at K=10). Each run reports, for every spec and sample size, both the
plug-in estimate and the LOO-debiased estimate with SieveVar inference (standard
error and 95% coverage).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opttreat.config import EstimatorConfig, ParameterConfig, VarianceConfig
from opttreat.data import split_treated_control
from opttreat.estimation import get_estimator
from opttreat.models import TaylorExpansionModel
from opttreat.parameters import get_parameter
from opttreat.variance import get_variance_estimator


Z = 1.959963984540054
PARAMETER_TYPE = "welfare_known"

# ---------------- DGP specs (best-results expansion across Taylor order K) ----------------
SPECS = [
    {"expansion": "hyperbolic", "K": 4},
    {"expansion": "hyperbolic", "K": 7},
    {"expansion": "hyperbolic", "K": 10},
]

# ---------------- Simulation settings ----------------
ITE = 500
NS = (1500, 3000, 6000)
SEED = 2025
JOBS = 4
LOO_DELTA0 = 0.05

# ---------------- RF-sieve first stage ----------------
N_FEATURES = 100
RFG_TYPE = "iid_sphere"
ACTIVATION = "exp"
SOLVER = "pinv"
PINV_RCOND = 1e-10
THETA_SOBOL = 2048
VARIANCE_SOBOL = 2048

STEM = "taylor_welfare_known"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def run(specs=SPECS, ns=NS, ite=ITE, *, jobs=JOBS, output_dir=OUTPUT_DIR):
    draw_rows = []
    for s_idx, spec in enumerate(specs):
        expansion, K = spec["expansion"], int(spec["K"])
        label = f"{expansion}_K{K}"
        estimator = get_estimator(EstimatorConfig(method="rf_ridge", options={
            "rfg_type": RFG_TYPE,
            "activation": ACTIVATION,
            "share_features": True,
            "n_features": N_FEATURES,
            "random_state": SEED,
            "alpha": 0.0,
            "solver": SOLVER,
            "pinv_rcond": PINV_RCOND,
        }))
        param = get_parameter(ParameterConfig(PARAMETER_TYPE, options={
            "dim": K,
            "n_sobol": THETA_SOBOL,
            "sobol_scramble": False,
            "pinv_rcond": PINV_RCOND,
        }))
        variance = get_variance_estimator(VarianceConfig(method="sieve_var", options={
            "param_type": PARAMETER_TYPE,
            "dim": K,
            "n_sobol": VARIANCE_SOBOL,
            "sobol_scramble": False,
            "alpha": 0.0,
            "solver": SOLVER,
            "pinv_rcond": PINV_RCOND,
        }))
        W_true = float(param.get_true_value(TaylorExpansionModel(K=K, expansion=expansion)))

        for n_idx, n in enumerate(ns):
            print(f"Running {label} | rf_ridge | n = {n} | ite = {ite}", flush=True)

            def one_ite(seed_i, expansion=expansion, K=K, n=n):
                np.random.seed(seed_i)
                model = TaylorExpansionModel(K=K, expansion=expansion)
                output = estimator.fit(split_treated_control(model.generate_data(n)))
                W_plug = float(param.plug_in(output["h_hat"]))
                W_loo = float(param.loo(output, delta0=LOO_DELTA0))
                se = float(np.sqrt(max(variance.fit(output), 0.0)))
                return W_plug, W_loo, se

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

            for rep, (W_plug, W_loo, se) in enumerate(results):
                draw_rows.append({"spec": label, "expansion": expansion, "K": K, "n": n, "estimator": "plug_in", "rep": rep, "W_hat": W_plug, "W_true": W_true, "se": se})
                draw_rows.append({"spec": label, "expansion": expansion, "K": K, "n": n, "estimator": "loo", "rep": rep, "W_hat": W_loo, "W_true": W_true, "se": se})

    draws = pd.DataFrame(draw_rows)
    summary = _summarize(draws)

    suffix = _suffix(specs, ns, ite)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{STEM}_summary_{suffix}.csv", index=False)
    draws.to_csv(output_dir / f"{STEM}_draws_{suffix}.csv", index=False)
    print(summary.to_string(index=False))
    return summary, draws


def _summarize(draws):
    rows = []
    for (spec_name, expansion, K, n, estimator_name), g in draws.groupby(
        ["spec", "expansion", "K", "n", "estimator"], sort=False
    ):
        W = g["W_hat"].to_numpy(dtype=float)
        se = g["se"].to_numpy(dtype=float)
        W_true = float(g["W_true"].iloc[0])
        rows.append({
            "spec": spec_name,
            "expansion": expansion,
            "K": int(K),
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


def _suffix(specs, ns, ite):
    expansions = "_".join(dict.fromkeys(s["expansion"] for s in specs))
    ks = "_".join(str(s["K"]) for s in specs)
    return (
        f"n{'_'.join(str(n) for n in ns)}_rep{ite}_nf{N_FEATURES}"
        f"_{RFG_TYPE}_{ACTIVATION}_{expansions}_K{ks}"
        f"_loo{str(LOO_DELTA0).replace('.', 'p')}"
    )


def main():
    return run()


def smoke_main(output_dir=OUTPUT_DIR):
    """Tiny local check (one spec, few reps, small Sobol)."""
    global N_FEATURES, THETA_SOBOL, VARIANCE_SOBOL
    N_FEATURES, THETA_SOBOL, VARIANCE_SOBOL = 20, 128, 128
    return run(specs=[{"expansion": "hyperbolic", "K": 4}], ns=(120,), ite=2, jobs=1, output_dir=output_dir)


if __name__ == "__main__":
    main()
