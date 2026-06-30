# OptTreat Core Package

OptTreat is a Python package for optimal treatment parameter estimation and
inference. The reusable workflow is:

```text
model.generate_data(n) -> split_treated_control(...) -> estimator.fit(...)
-> parameter.plug_in(...) -> optional variance.fit(...) -> summary/draw tables
```

For direct Chen, Chen, and Gao (2025)-style inference, use the public inference
workflow:

```python
from opttreat import EstimatorConfig, ParameterConfig, ParameterType, fit_inference
from opttreat.models import Model4

data = Model4().generate_data(500)
result = fit_inference(
    data,
    estimator_config=EstimatorConfig(
        "sieve",
        {
            "solver": "pinv",
            "share_features": False,
            "J_x_degree": 3,
            "J_x_segments_t": 1,
            "J_x_segments_c": 1,
            "knots": "uniform",
            "basis": "tensor",
        },
    ),
    parameter_config=ParameterConfig(
        ParameterType.WELFARE_KNOWN_DIST,
        {"dim": 2, "n_sobol": 1024},
    ),
)

print(result.W_hat, result.se, result.ci)
```

`fit_inference` supports plug-in and leave-one-out point estimators through
`point_method="plug_in"` or `"loo"`.
When no `VarianceConfig` is supplied, it builds the default sieve variance
estimator from the parameter options and returns standard errors and confidence
intervals.

## Module Map

| Path | Purpose |
| --- | --- |
| `config.py` | Dataclasses for estimator, parameter, and variance configuration. |
| `data.py` | Parses DataFrames/dicts into treated and control samples; normalizes array shapes. |
| `models/` | DGP families: `CCGModel` for CCG M1-M15 and `TaylorExpansionModel` for Taylor-style designs. |
| `estimation/` | First-stage estimators and their random-feature/spline basis builders. |
| `inference.py` | High-level welfare/value inference API with point estimates, SieveVar standard errors, CIs, and multiplier bootstrap critical values. |
| `parameters/` | Welfare and value functionals for known and unknown target distributions. |
| `variance/` | Generic sieve-style variance estimators. |
| `simulations/simulation_engine.py` | Shared Monte Carlo engine used by runnable simulation files. |
| `simulations/ccg2025/` | Chen, Chen, and Gao (2025) SieveVar workflow. |
| `simulations/DenseIndexDGP/` | CodeWG 260610 dense-index simulations, split into welfare and value workflows. |
| `simulations/TaylorModel/` | Random-feature simulations for Taylor expansions. |
| `tests/` | Pytest coverage for models, estimators, parameters, variance, and simulations. |

## Model Families

`CCGModel` contains `Model1` through `Model15` through one definition table
in `models/ccg_model.py`. `get_model("Model1")` through
`get_model("Model15")` return `CCGModel` instances.

`TaylorExpansionModel` contains Taylor-style expansion designs. High-D tan2
simulations use explicit Taylor specs with `K=3`, `K=7`, and `K=10`.
`Model99`, `Model100`, `Model101`, and `Model102` are not active model names.

## Running Simulations

Simulation files are plain Python modules with explicit configuration variables
near the top. From the repository root:

```bash
python -m opttreat.simulations.ccg2025.ccg_welfare_known.run_ccg_welfare_known
python -m opttreat.simulations.ccg2025.ccg_welfare_unknown.run_ccg_welfare_unknown
python -m opttreat.simulations.ccg2025.ccg_value_known.run_ccg_value_known
python -m opttreat.simulations.TaylorModel.run_taylor_rf
```

Most simulation files keep smoke-size settings near the top. The CCG 2025
runner exposes both `R_CODE_SETTINGS` for the full R-script replication and
`SMOKE_SETTINGS` for quick checks.

Simulation outputs use the shared naming convention:

```text
<simulation>_summary_n<sample-sizes>_rep<replications>_<settings>.csv
<simulation>_draws_n<sample-sizes>_rep<replications>_<settings>.csv
<simulation>_results_n<sample-sizes>_rep<replications>_<settings>.md
```

For example, `n1500_3000_6000_rep1500_nf100_K4_7_10` means sample sizes
`1500, 3000, 6000`, 1500 Monte Carlo replications per cell, 100 random
features, and `K` values `4, 7, 10`.

## Supported Components

Estimators:
- `rf_ridge`: ridge or pinv least squares on random features.
- `sieve`: B-spline basis functions with `solver="ridge"` or `solver="pinv"`.

Parameters:
- `welfare_known`: `E[max(h(X), 0)]` under a known target distribution.
- `welfare_unknown`: empirical/common-support average of `max(h(X_i), 0)`.
- `value_known`: `E[1{h(X)>0}m(X)]` under a known target distribution.
- `value_unknown`: empirical average of `1{h(X_i)>0}m(X_i)`.

Variance:
- `sieve`: generic sieve-style variance.

## Running Tests

```bash
python -m pytest opttreat/tests
```

## Estimator Output Contract

Every estimator exposes `fit(parsed_data)` and `get_output()`. The output dict
contains `h_hat`, fitted nuisance functions, design matrices, residuals,
feature maps, `X_all`, `alpha`, and `solver` where applicable.
