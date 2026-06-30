# TaylorModel Random-Feature Simulations

This folder runs known-distribution welfare Monte Carlo simulations for
`TaylorExpansionModel` using a random-feature ridge first stage. The workflow is:

```text
TaylorExpansionModel -> generated data -> RF first stage
-> known-distribution welfare (plug-in + LOO) -> SieveVar inference -> CSVs
```

The runner is a single self-contained, linear script in the style of the
`ccg2025` and `DenseIndexDGP` runners: the DGP specs, the RF-sieve tuning, and the
simulation settings are top-level constants, the Monte Carlo loop is spelled out
in `run`, and the estimator, parameter, and SieveVar standard error are called
straight from `opttreat.estimation`, `opttreat.parameters` and
`opttreat.variance`. There is no shared simulation engine.

Run from the project root:

```bash
python -m opttreat.simulations.TaylorModel.run_taylor_rf
```

For a fast local check call `smoke_main()` from Python. Each run reports, for
every spec and sample size, both the plug-in estimate and the LOO-debiased
estimate with SieveVar inference (standard error and 95% coverage).

## Configuration

The default `SPECS` use the best-results design from the existing estimation-only
simulation results: the **`hyperbolic`** expansion across Taylor orders
`K = 4, 7, 10`. In those bias results the `hyperbolic` RF welfare estimator had
the smallest, most stable bias across `K`, while `rational` blew up as `K` grew
and `tan2`/`sinh2`/`exp_pm` picked up bias at `K = 10`. Other expansions can be
added back to `SPECS` (the model still supports `tan2`, `sinh2`, `rational`,
`exp_pm`).

| setting | value |
| --- | --- |
| `SPECS` | `hyperbolic` at `K = 4, 7, 10` |
| `NS` | `(1500, 3000, 6000)` |
| `ITE` | `500` |
| `N_FEATURES` | `100` |
| `RFG_TYPE` | `iid_sphere` |
| `ACTIVATION` | `exp` |
| `SOLVER` | `pinv` |
| `THETA_SOBOL` / `VARIANCE_SOBOL` | `2048` / `2048` |
| `LOO_DELTA0` | `0.05` |

Each design cell is labelled `<expansion>_K<K>`, e.g. `hyperbolic_K7`.

## Data-Generating Process

For all TaylorModel designs:

- observed covariates are drawn from `[-0.2, 1.2]^K`;
- target covariates for welfare are `Uniform([0, 1]^K)`;
- treatment is randomized with `p0(x) = 0.5`;
- outcomes satisfy `Y = baseline(X) + D * h0(X) + noise`, noise sd `1`.

The known-distribution welfare target is `W0 = E[max(h0(X), 0)]` over
`Uniform([0, 1]^K)`, evaluated with unscrambled Sobol points.

For the `hyperbolic` expansion only `x_1` enters the signal:

```text
baseline(x) = Taylor_K[sinh(x_1)]
mu0(x, 1)   = Taylor_K[cosh(x_1)]
h0(x)       = Taylor_K[cosh(x_1)] - Taylor_K[sinh(x_1)]
```

so `h0(x) = Taylor_K[exp(-x_1)] > 0`, and the welfare is the smooth functional
`E[h0(X)]`. The remaining coordinates are nuisance dimensions when `K > 1`. The
model formulas live in `opttreat/models/taylor_expansion_model.py`.

## RF Ridge First Stage

The estimator is `method="rf_ridge"` with `solver="pinv"`. It fits treated and
control regressions separately in a shared random-feature basis
(`rfg_type="iid_sphere"`, `activation="exp"`, `n_features=100`) and returns the
callable `h_hat = mu_t_hat - mu_c_hat`. The welfare plug-in averages
`max(h_hat(X), 0)` over the Sobol target grid; the LOO estimate removes the
diagonal quadratic parameter bias; and the SieveVar estimator supplies the
standard error and 95% coverage.

## Output Files

Outputs are written under `results/`:

```text
taylor_welfare_known_summary_<settings>.csv
taylor_welfare_known_draws_<settings>.csv
```

The summary CSV has one row per spec, sample size, and estimator (`plug_in` and
`loo`) with `bias`, `sd`, `se`, `sd_se`, and `coverage`.
