# DenseIndexDGP Simulations

This folder holds the dense-index (CodeWG 260610) RF-sieve simulation exercises
for the two target functionals we report:

```text
DenseIndexDGP/
  welfare_known/   # W0 = E[max(tau(X), 0)]
  value_known/     # V0 = E[1{tau(X) > 0} m(X)], m(x) = 1
```

Each leaf has one self-contained `run_dense_index_<functional>.py` runner and a
`results/` folder. The DGP definition lives in `opttreat/models/dense_index_model.py`.

## Runners

The runners are written in the style of the `ccg2025` scripts: a single linear
file with the DGP, the RF-sieve tuning, and the simulation settings as top-level
constants, the random-feature sieve map built inline, and the Monte Carlo loop
spelled out. The first-stage estimator, target parameter, and SieveVar standard
error come straight from `opttreat.estimation`, `opttreat.parameters` and
`opttreat.variance`; the RF feature builders come from `opttreat.estimation`.
There is no shared simulation helper.

Run from the repository root:

```bash
python -m opttreat.simulations.DenseIndexDGP.welfare_known.run_dense_index_welfare_known
python -m opttreat.simulations.DenseIndexDGP.value_known.run_dense_index_value_known
```

Each runner reports, for every sample size, both the plug-in estimate and the
LOO-debiased estimate with SieveVar inference (standard error and 95% coverage).
For a fast local check call `smoke_main()` from Python.

## Configuration

Both runners use the best-coverage designs from prior tuning runs: 200
quasi-sphere sigmoid random features at `gamma = 1.0`, `solver = "pinv"`, and the
leave-one-out debiased estimator (`loo_delta0 = 0.05`) approaching ~0.95 coverage
at the largest sample size. The defaults are `rep = 500` over
`n = 1500, 3000, 6000, 12000` at dimension `50`.

| setting | welfare_known | value_known |
| --- | ---: | ---: |
| random features | 200 | 200 |
| RF generator | quasi_sphere | quasi_sphere |
| activation | sigmoid | sigmoid |
| gamma | 1.0 | 1.0 |
| theta Sobol | 512 | 8192 |
| variance Sobol | 512 | 32768 |
| value iota | -- | 0.01 |

The value functional integrates a discontinuous indicator with an eps-band
level-set derivative (`eps = iota * sd(h_hat)`), so it needs a much finer Sobol
grid than the smooth welfare functional.

## Common Setup

The DGP uses `X ~ Uniform([0, 1]^d)` and `Y = b0(X) + D * tau0(X) + epsilon`
with homoskedastic default errors. The RF first stage is a fixed random-feature
linear sieve conditional on the feature draw, using shared treated/control
features and `solver = "pinv"`. The target distribution is the known
`Uniform([0, 1]^d)` covariate distribution.

## Result Naming

```text
DenseIndexDGP/<functional>/results/
  dense_index_<functional>_summary_<settings>.csv
  dense_index_<functional>_draws_<settings>.csv
```

The summary CSV has one row per sample size and estimator (`plug_in` and `loo`)
with `bias`, `sd`, `se`, `sd_se`, and `coverage`; the draws CSV has one row per
replication and estimator.
