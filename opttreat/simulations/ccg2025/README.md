# CCG 2025 SieveVar Simulations

This folder contains the OptTreat simulation workflow for the main Chen, Chen,
and Gao (2025) SieveVar Monte Carlo designs. The goal is to keep the paper
setup readable while reusing the package components:

```text
CCGModel -> simulated data -> sieve estimator -> target parameter -> SieveVariance -> output tables
```

The simulation is organized into one runnable folder per CCG model group. Each
folder has its own `run_simulation.py` and `results/` directory:

```text
ccg2025/
  ccg_welfare_known/      # Theorem 1 known-distribution welfare, Model1-Model7
  ccg_welfare_unknown/    # Theorem 2 unknown/common-support welfare, Model8-Model14
  ccg_value_known/        # Theorem 3 known-distribution value functional, Model15
```

Run each group from the repository root:

```bash
python -m opttreat.simulations.ccg2025.ccg_welfare_known.run_ccg_welfare_known
python -m opttreat.simulations.ccg2025.ccg_welfare_unknown.run_ccg_welfare_unknown
python -m opttreat.simulations.ccg2025.ccg_value_known.run_ccg_value_known
```

Each runner is a fully self-contained, linear script in the style of the R
`TEST_Thm*_Spline_SieveVar.R` scripts: it lists its models and per-model sieve
dimensions inline, sets the simulation settings (replications, sample sizes,
Sobol counts, seed, jobs) as top-level constants, and writes out the Monte Carlo
loop explicitly. The first-stage estimator, the target parameter, and the
SieveVar standard error are called directly from `opttreat.estimation`,
`opttreat.parameters` and `opttreat.variance` -- there is no shared
ccg2025-local helper. Each run reports, for every model and sample size, both the
plug-in estimate and the LOO-debiased estimate with SieveVar inference (standard
error and 95% coverage). The LOO second-derivative correction defaults to the
analytical boundary-band method (`LOO_METHOD = "band"`) rather than
finite-difference `"central_difference"`.

Each runner's `ESTIMATOR` constant selects the first stage: `"sieve"` (B-spline
sieve, pseudo-inverse OLS = R `ginv`) or `"rf_ridge"` (random-feature ridge).
Both estimators return the same output structure, so the plug-in / LOO / SieveVar
code is identical; the estimator name is recorded in the output file names. The
runners are intentionally not command-line applications; `main()` runs the full
paper configuration. For a fast check, call `smoke_main()` from Python.

## What Lives Where

| Location | Role |
| --- | --- |
| `opttreat/models/ccg_model.py` | Defines `CCGModel` and its Model1-Model15 definition table. |
| `opttreat/estimation`, `opttreat/parameters`, `opttreat/variance` | The first-stage estimator (`get_estimator`), target parameters (`get_parameter`), and SieveVar standard error (`get_variance_estimator`) called directly by each runner. |
| `ccg_welfare_known/run_ccg_welfare_known.py` | Runs Model1-Model7 (Theorem 1 known-distribution welfare); models + sieve dimensions + settings inline. |
| `ccg_welfare_unknown/run_ccg_welfare_unknown.py` | Runs Model8-Model14 (Theorem 2 unknown/common-support welfare). |
| `ccg_value_known/run_ccg_value_known.py` | Runs Model15 (Theorem 3 known-distribution value functional). |
| `opttreat/estimation/` | Provides the sieve estimator used as the first-stage estimator. |
| `opttreat/parameters/` | Provides welfare and value target parameter classes. |
| `opttreat/variance/sieve_var.py` | Provides the generic sieve variance estimator used for standard errors and coverage. |
| `<group>/results/` | Each group runner stores its summary CSVs, draw-level CSVs, and Markdown reports here. |

Models only define data-generating processes. The paper-specific tuning lives
in this simulation folder, not inside the model classes.

## Model Groups

| Models | Target | Dimension | Target support | Observed support | Paper role |
| --- | --- | ---: | --- | --- | --- |
| Model1-Model3 | Known-distribution welfare | 1 | `[0, 1]` | `[-0.2, 1.2]` | Theorem 1 |
| Model4-Model7 | Known-distribution welfare | 2 | `[0, 1]^2` | `[-0.2, 1.2]^2` | Theorem 1 |
| Model8-Model10 | Unknown/common-support welfare | 1 | observed common support | `[0, 1]` | Theorem 2 |
| Model11-Model14 | Unknown/common-support welfare | 2 | observed common support | `[0, 1]^2` | Theorem 2 |
| Model15 | Known-distribution value functional | 2 | `[-1.5, 1.5]^2` | `[-2, 2]^2` | Theorem 3 |

The main CCG paper set has 15 numbered models. The value design is represented
as `Model15` in OptTreat even though one local R script labels that theorem's
example internally as "Model 1".

## Explicit Model Formulas

Let `expit(z) = 1 / (1 + exp(-z))`. Each simulation draw uses:

```text
D | X ~ Bernoulli(p0(X))
Y = mu0(X, D) + epsilon
epsilon ~ N(0, 1)
h0(X) = mu0(X, 1) - mu0(X, 0)
```

For Model1-Model14:

```text
mu0(x, d) = mu0(x, 0) + d*h0(x)
```

For Model15, `mu0(x,0)=0`, so `mu0(x,d)=d*h0(x)`.

### One-Dimensional Models

| Model | Observed support | Target support | `p0(x)` | `mu0(x,0)` | `h0(x)` |
| --- | --- | --- | --- | --- | --- |
| Model1 | `[-0.2, 1.2]` | `[0, 1]` | `expit(1 - 2*x)` | `5*sin(2*pi*x)*cos(2*pi*x)` | `-0.4 + 2*x^2` |
| Model2 | `[-0.2, 1.2]` | `[0, 1]` | `expit(-0.5 + x)` | `0.5*abs(x)` | `0.5 - x^2` |
| Model3 | `[-0.2, 1.2]` | `[0, 1]` | `expit(0.5 - x)` | `x^2` | `1 - x` |
| Model8 | `[0, 1]` | `[0, 1]` | `expit(1 - 2*x)` | `5*sin(2*pi*x)*cos(2*pi*x)` | `-0.4 + 2*x^2` |
| Model9 | `[0, 1]` | `[0, 1]` | `expit(-0.5 + x)` | `0.5*abs(x)` | `0.5 - x^2` |
| Model10 | `[0, 1]` | `[0, 1]` | `expit(0.5 - x)` | `x^2` | `1 - x` |

Model1-Model3 and Model8-Model10 share the same formulas by pairs. The
difference is the observed covariate support: Model1-Model3 draw `X` from the
larger interval `[-0.2, 1.2]`, while Model8-Model10 draw `X` from `[0, 1]`.

### Two-Dimensional Welfare Models

For these models, write `x = (x1, x2)`.

| Model | Observed support | Target support | `p0(x)` | `mu0(x,0)` | `h0(x)` |
| --- | --- | --- | --- | --- | --- |
| Model4 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `(1 - x1^2 - x2^2) * (4 + sin(x1)*x2 + cos(x2))` | `0.5*x1 - 0.4*x2` |
| Model5 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `(1 - x1*x2) * (3 + sin(pi*x1)*cos(pi*x2))` | `0.3*x1 - 0.3*x2` |
| Model6 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(1.5*x1 - 0.5*x2)` | `log(1 + x1 + x2)` | `x1 - 0.7*x2` |
| Model7 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(-0.5 + x1 + 2*x2)` | `(x1^2 + x2^2) * exp(-x1 - x2)` | `0.5 - x2` |
| Model11 | `[0, 1]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `(1 - x1^2 - x2^2) * (4 + sin(x1)*x2 + cos(x2))` | `0.5*x1 - 0.4*x2` |
| Model12 | `[0, 1]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `(1 - x1*x2) * (3 + sin(pi*x1)*cos(pi*x2))` | `0.3*x1 - 0.3*x2` |
| Model13 | `[0, 1]^2` | `[0, 1]^2` | `expit(1.5*x1 - 0.5*x2)` | `log(1 + x1 + x2)` | `x1 - 0.7*x2` |
| Model14 | `[0, 1]^2` | `[0, 1]^2` | `expit(-0.5 + x1 + 2*x2)` | `(x1^2 + x2^2) * exp(-x1 - x2)` | `0.5 - x2` |

Model4-Model7 and Model11-Model14 share the same formulas by pairs. The
difference is again the observed covariate support: Model4-Model7 use
`[-0.2, 1.2]^2`, while Model11-Model14 use `[0, 1]^2`.

### Value Model

Model15 is the CCG value-functional design:

| Model | Observed support | Target support | `p0(x)` | `mu0(x,0)` | `h0(x)` |
| --- | --- | --- | --- | --- | --- |
| Model15 | `[-2, 2]^2` | `[-1.5, 1.5]^2` | `expit(x1 - x2)` | `0` | `(1 - x1^2 - x2^2) * (4 + sin(x1)*x2 + cos(x2))` |

The CCG value target used in this folder is:

```text
E[1{h0(X) > 0} * 9] = pi
```

## Target Parameters

The simulation file assigns one parameter type to each model:

| Parameter type | Models | Formula computed by OptTreat |
| --- | --- | --- |
| Known welfare | Model1-Model7 | `E[max(h(X), 0)]` over the target uniform distribution. |
| Unknown/common-support welfare | Model8-Model14 | empirical `mean(max(h(X_i), 0))` over the treated/control common-support sample. |
| Known value | Model15 | `E[1{h(X)>0} v(X)]` over `[-1.5, 1.5]^2`, with `v(X)=9`. |

For Model15 the target is the area of the positive treatment-effect region
scaled by `v(X)=9`; the true value is set to `pi`.

Known-distribution integrals use Sobol points with `sobol_scramble=False`, so
the integration rule mirrors the deterministic Sobol style used by the CCG
R workflow.

## Sieve First Stage

All CCG simulations in this folder use the sieve estimator:

```text
mu_hat_1(x) = fitted conditional mean for treated observations
mu_hat_0(x) = fitted conditional mean for control observations
h_hat(x)   = mu_hat_1(x) - mu_hat_0(x)
```

The treated and control regressions use separate spline bases
(`share_features=False`). The current CCG settings are:

| Setting | Value |
| --- | --- |
| spline degree | `3` |
| basis | tensor-product B-spline basis |
| knots | uniform |
| solver | Moore-Penrose pseudo-inverse, `pinv` |
| `pinv_rcond` | `sqrt(machine epsilon)`, about `1.49e-8` |

The pseudo-inverse path is used because the R simulations use `MASS::ginv(...)`
rather than ridge regularization. Small singular directions below the tolerance
are treated as numerically zero instead of being inverted.

## Spline Segment Choices

Each runner's `SPECS` list specifies separate spline segment counts (`Jc`, `Jt`)
for the control and treated first-stage regressions:

| Models | Control segments | Treated segments |
| --- | ---: | ---: |
| Model1 | 16 | 16 |
| Model2-Model5 | 1 | 1 |
| Model6-Model7 | 4 | 1 |
| Model8 | 8 | 8 |
| Model9 | 1 | 4 |
| Model10 | 4 | 1 |
| Model11-Model14 | 1 | 1 |
| Model15 | 1 | 4 |

For two-dimensional designs these are tensor-product spline segments, so the
number of columns grows with dimension.

## Variance And Inference

Every CCG spec in this folder supplies a `VarianceConfig`, so the summary table
includes standard errors and coverage.

| Target | SieveVar evaluation |
| --- | --- |
| Known welfare | Sobol integration over the target support using derivative `1{h_hat(x) >= 0}`. |
| Unknown/common-support welfare | empirical common-support evaluation plus the empirical variance component for `max(h_hat(X_i), 0)`. |
| Known value | level-set approximation using `-eps < h_hat(x) < eps`, with `eps=0.005` and `v(X)=9`. |

The R-code settings use many more Sobol points for variance than the smoke
settings. The CCG-specific target supports and `v(X)=9` choice are supplied by
each runner's parameter/variance options; the variance estimator itself remains generic.

## Run Settings

Each runner declares its settings as top-level constants, matching the R scripts:

| Constant | Value |
| --- | ---: |
| `SPECS` | the runner's models with their `Jt`/`Jc` sieve dimensions |
| `ITE` | `2000` |
| `NS` | `(1500, 3000, 6000)` |
| `THETA_SOBOL` | `5000` |
| `VARIANCE_SOBOL` / `VALUE_VARIANCE_SOBOL` | `40000` / `1000000` |
| `SEED` | `2025` |
| `JOBS` | `3` |
| `LOO_DELTA0` | `0.05` |
| `LOO_METHOD` | `"band"` (analytical; set to `"central_difference"` for finite differences) |
| `ESTIMATOR` | `"sieve"` (B-spline sieve) or `"rf_ridge"` (random features); RF tuned via the `RF_*` constants |

For a tiny smoke check of one group:

```python
from opttreat.simulations.ccg2025.ccg_welfare_known.run_ccg_welfare_known import smoke_main

summary, draws = smoke_main()
```

To run the full R-code replication, run each group runner:

```bash
python -m opttreat.simulations.ccg2025.ccg_welfare_known.run_ccg_welfare_known
python -m opttreat.simulations.ccg2025.ccg_welfare_unknown.run_ccg_welfare_unknown
python -m opttreat.simulations.ccg2025.ccg_value_known.run_ccg_value_known
```

This is an expensive run: 15 models across the three groups, 3 sample sizes, and
2000 replications, each reporting plug-in and LOO estimates.

## Output Files

Each group runner writes to its own `results/` folder using the shared
simulation naming convention with a group-specific stem:

```text
<group>/results/<stem>_summary_n<sample-sizes>_rep<replications>_loo<delta0>_<method>.csv
<group>/results/<stem>_draws_n<sample-sizes>_rep<replications>_loo<delta0>_<method>.csv
```

For example, the full paper-style runs write:

```text
ccg_welfare_known/results/ccg_welfare_known_summary_n1500_3000_6000_rep2000_loo0p05_band.csv
ccg_welfare_unknown/results/ccg_welfare_unknown_summary_n1500_3000_6000_rep2000_loo0p05_band.csv
ccg_value_known/results/ccg_value_known_summary_n1500_3000_6000_rep2000_loo0p05_band.csv
```

The summary CSV contains one row per model, sample size, and estimator
(`plug_in` and `loo`):

| Column | Meaning |
| --- | --- |
| `spec` | model label, such as `Model8` |
| `n` | sample size |
| `estimator` | `plug_in` or `loo` |
| `W_true` | target truth from the model and parameter |
| `bias` | `mean(W_hat) - W_true` |
| `sd` | Monte Carlo standard deviation of `W_hat` |
| `se` | average SieveVar standard error (shared by plug-in and LOO) |
| `sd_se` | Monte Carlo standard deviation of the standard errors |
| `coverage` | empirical coverage of nominal 95 percent intervals |
| `replications` | number of Monte Carlo replications |

The draw-level CSV contains one row per replication and estimator:

| Column | Meaning |
| --- | --- |
| `spec`, `n`, `estimator` | design identifiers |
| `rep` | replication index |
| `W_hat` | estimated welfare or value target |
| `W_true` | target truth |
| `se` | SieveVar standard error for that draw |

## Reading The Results

For welfare models, lower absolute bias and an empirical `sd` close to average
`se` indicate that the estimator and SieveVar are behaving similarly to the CCG
reported tables. Coverage should be close to the nominal 0.95 level in the
large paper-style runs, with normal Monte Carlo fluctuation.

Python results are intended to be statistically comparable to the CCG R results,
not bit-for-bit identical. Random-number streams, Sobol implementations, spline
matrix construction, and pseudo-inverse details differ across Python and R.

## Notes

- `Model4` is included because OptTreat treats the main CCG design set as
  Model1-Model15, even though one local two-dimensional R workflow commented it
  out.
- `Model15` is the canonical OptTreat name for the value/area design.
- Changing each runner's `ITE` constant changes the number of Monte Carlo
  repetitions. It is separate from the sample size `n`, controlled by `NS`.
- Historical CSVs may exist in `results/`; active output names identify the
  exact sample sizes, replication count, and model set used for the run.
