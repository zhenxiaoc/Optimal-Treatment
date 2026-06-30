# OptTreat Model Specifications

This file documents the data-generating processes implemented in `opttreat/models`.
CCG `Model1` through `Model15` are implemented by the `CCGModel.DEFINITIONS` table in
`ccg_model.py`; `TaylorExpansionModel` is the separate Taylor-style
model family.

## Common Convention

Unless noted otherwise, each model follows this workflow:

1. Draw observed covariates `X` from `rF0(n)`.
2. Draw treatment `D | X ~ Bernoulli(p0(X))`.
3. Draw outcome `Y = mu0(X, D) + epsilon`, where `epsilon ~ N(0, noise_sd^2)`.
4. Use `rF(m)` or `inverse_CDF(U)` for the target/integration distribution.
5. The treatment effect is `h0(X) = mu0(X, 1) - mu0(X, 0)`.

`expit(z)` means `1 / (1 + exp(-z))`.

## Quick Index

| Model | Dimension | Observed support `rF0` | Target support | Propensity `p0(X)` | Treatment effect `h0(X)` |
| --- | ---: | --- | --- | --- | --- |
| `Model1` | 1 | `[-0.2, 1.2]` | `[0, 1]` | `expit(1 - 2*x)` | `-0.4 + 2*x^2` |
| `Model2` | 1 | `[-0.2, 1.2]` | `[0, 1]` | `expit(-0.5 + x)` | `0.5 - x^2` |
| `Model3` | 1 | `[-0.2, 1.2]` | `[0, 1]` | `expit(0.5 - x)` | `1 - x` |
| `Model4` | 2 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `0.5*x1 - 0.4*x2` |
| `Model5` | 2 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `0.3*x1 - 0.3*x2` |
| `Model6` | 2 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(1.5*x1 - 0.5*x2)` | `x1 - 0.7*x2` |
| `Model7` | 2 | `[-0.2, 1.2]^2` | `[0, 1]^2` | `expit(-0.5 + x1 + 2*x2)` | `0.5 - x2` |
| `Model8` | 1 | `[0, 1]` | `[0, 1]` | `expit(1 - 2*x)` | `-0.4 + 2*x^2` |
| `Model9` | 1 | `[0, 1]` | `[0, 1]` | `expit(-0.5 + x)` | `0.5 - x^2` |
| `Model10` | 1 | `[0, 1]` | `[0, 1]` | `expit(0.5 - x)` | `1 - x` |
| `Model11` | 2 | `[0, 1]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `0.5*x1 - 0.4*x2` |
| `Model12` | 2 | `[0, 1]^2` | `[0, 1]^2` | `expit(x1 - x2)` | `0.3*x1 - 0.3*x2` |
| `Model13` | 2 | `[0, 1]^2` | `[0, 1]^2` | `expit(1.5*x1 - 0.5*x2)` | `x1 - 0.7*x2` |
| `Model14` | 2 | `[0, 1]^2` | `[0, 1]^2` | `expit(-0.5 + x1 + 2*x2)` | `0.5 - x2` |
| `Model15` | 2 | `[-2, 2]^2` | `[-1.5, 1.5]^2` | `expit(x1 - x2)` | `(1 - x1^2 - x2^2)*(4 + sin(x1)*x2 + cos(x2))` |
| `TaylorExpansionModel` | configurable | `support0^K` | `support^K` | `0.5` | depends on `expansion` |

## Chen, Chen, and Gao (2025) Models

The main CCG 2025 simulation set contains 15 numbered models:

- `Model1` through `Model7`: Theorem 1 known-target welfare models.
- `Model8` through `Model14`: Theorem 2 unknown/common-target welfare models.
- `Model15`: Theorem 3 value-functional model. The local R file names this
  value example "Model 1", but in the M1-M15 paper workflow it is `Model15`.

The local 2D SieveVar R script comments out `Model4`; OptTreat includes it
because it appears in the paper and in the corresponding plug-in R script.

## One-Dimensional CCG 2025 Welfare Models

`Model1` through `Model3` draw observed covariates from `[-0.2, 1.2]` and
target covariates from `[0, 1]`. `Model8` through `Model10` reuse the same
outcome and propensity formulas, but draw both observed and target covariates
from `[0, 1]`.

| Model | Baseline `b(x)` | Treatment effect `h0(x)` |
| --- | --- | --- |
| `Model1`, `Model8` | `5*sin(2*pi*x)*cos(2*pi*x)` | `-0.4 + 2*x^2` |
| `Model2`, `Model9` | `0.5*abs(x)` | `0.5 - x^2` |
| `Model3`, `Model10` | `x^2` | `1 - x` |

For each model:

```text
mu0(x, d) = b(x) + d*h0(x)
```

## Two-Dimensional Models

### Model4

- Observed covariates: `X1, X2 ~ Uniform(-0.2, 1.2)`.
- Target covariates: `X1, X2 ~ Uniform(0, 1)`.
- Propensity: `p0(x) = expit(x1 - x2)`.
- Baseline outcome:

```text
b4(x) = (1 - x1^2 - x2^2) * (4 + sin(x1)*x2 + cos(x2))
```

- Treatment effect:

```text
h0(x) = 0.5*x1 - 0.4*x2
```

- Outcome mean:

```text
mu0(x, d) = b4(x) + d*h0(x)
```

### Model5

- Observed covariates: `X1, X2 ~ Uniform(-0.2, 1.2)`.
- Target covariates: `X1, X2 ~ Uniform(0, 1)`.
- Propensity: `p0(x) = expit(x1 - x2)`.
- Baseline outcome:

```text
b5(x) = (1 - x1*x2) * (3 + sin(pi*x1)*cos(pi*x2))
```

- Treatment effect:

```text
h0(x) = 0.3*x1 - 0.3*x2
```

- Outcome mean:

```text
mu0(x, d) = b5(x) + d*h0(x)
```

### Model6

- Observed covariates: `X1, X2 ~ Uniform(-0.2, 1.2)`.
- Target covariates: `X1, X2 ~ Uniform(0, 1)`.
- Propensity: `p0(x) = expit(1.5*x1 - 0.5*x2)`.
- Baseline outcome:

```text
b6(x) = log(1 + x1 + x2)
```

- Treatment effect:

```text
h0(x) = x1 - 0.7*x2
```

- Outcome mean:

```text
mu0(x, d) = b6(x) + d*h0(x)
```

### Model7

- Observed covariates: `X1, X2 ~ Uniform(-0.2, 1.2)`.
- Target covariates: `X1, X2 ~ Uniform(0, 1)`.
- Propensity: `p0(x) = expit(-0.5 + x1 + 2*x2)`.
- Baseline outcome:

```text
b7(x) = (x1^2 + x2^2) * exp(-x1 - x2)
```

- Treatment effect:

```text
h0(x) = 0.5 - x2
```

- Outcome mean:

```text
mu0(x, d) = b7(x) + d*h0(x)
```

## Unit-Support Two-Dimensional Models

`Model11` through `Model14` draw both observed and target covariates from `[0, 1]^2`.
Their propensities mirror the corresponding Theorem 1 two-dimensional designs:
`Model11` and `Model12` use `expit(x1 - x2)`, `Model13` uses
`expit(1.5*x1 - 0.5*x2)`, and `Model14` uses `expit(-0.5 + x1 + 2*x2)`.

The outcome formulas mirror `Model4` through `Model7`:

| Model | Baseline `b(x)` | Treatment effect `h0(x)` |
| --- | --- | --- |
| `Model11` | `(1 - x1^2 - x2^2) * (4 + sin(x1)*x2 + cos(x2))` | `0.5*x1 - 0.4*x2` |
| `Model12` | `(1 - x1*x2) * (3 + sin(pi*x1)*cos(pi*x2))` | `0.3*x1 - 0.3*x2` |
| `Model13` | `log(1 + x1 + x2)` | `x1 - 0.7*x2` |
| `Model14` | `(x1^2 + x2^2) * exp(-x1 - x2)` | `0.5 - x2` |

For each model:

```text
mu0(x, d) = b(x) + d*h0(x)
```

## High-D Tan2 Taylor Specifications

The high-dimensional tan2 simulations use explicit special cases of
`TaylorExpansionModel(expansion="tan2")`, not numbered model aliases. They use:

```text
p0(x) = 0.5
h0(x) = 1
mu0(x, d) = b(x) + d
```

Observed covariates are drawn from `[-0.2, 1.2]^K`, and target covariates are
drawn from `[0, 1]^K`.

`Model99`, `Model100`, `Model101`, and `Model102` are not active model names.
Use explicit `TaylorExpansionModel` configurations instead.

### K = 3

```text
TaylorExpansionModel(expansion="tan2", K=3)
```

```text
b(x) = x1^2 + (2/3)*x2^4 + (17/45)*x3^6
```

### K = 7

```text
TaylorExpansionModel(expansion="tan2", K=7)
```

```text
b(x) =
    x1^2
  + (2/3)*x2^4
  + (17/45)*x3^6
  + (62/315)*x4^8
  + (1382/14175)*x5^10
  + (21844/467775)*x6^12
  + (929669/42568525)*x7^14
```

### K = 10

```text
TaylorExpansionModel(expansion="tan2", K=10)
```

```text
b(x) =
    x1^2
  + (2/3)*x2^4
  + (17/45)*x3^6
  + (62/315)*x4^8
  + (1382/14175)*x5^10
  + (21844/467775)*x6^12
  + (929669/42568525)*x7^14
  + (6404582/638512875)*x8^16
  + (443861162/97692469875)*x9^18
  + (18888466084/9280784638125)*x10^20
```

## Model15

`Model15` is the CCG 2025 value-functional model. The local Theorem 3 R script
labels this example "Model 1"; in OptTreat and in the M1-M15 workflow it is
the canonical `Model15`.

- Observed covariates: `X1, X2 ~ Uniform(-2, 2)`.
- Target covariates: `X1, X2 ~ Uniform(-1.5, 1.5)`.
- Propensity: `p0(x) = expit(x1 - x2)`.
- Baseline outcome under control: `0`.
- Treatment effect:

```text
h0(x) = (1 - x1^2 - x2^2) * (4 + sin(x1)*x2 + cos(x2))
```

- Outcome mean:

```text
mu0(x, d) = d*h0(x)
```

- `inverse_CDF(u) = -1.5 + 3*u`, which maps Sobol points from `[0,1]^2`
  to `[-1.5,1.5]^2`.
- CCG value estimand: `9 * E[1{h0(X) > 0}] = pi`.

## TaylorExpansionModel

`TaylorExpansionModel` is configurable.

Constructor defaults:

```text
K = 10
expansion = "tan2"
support0 = (-0.2, 1.2)
support = (0.0, 1.0)
p0(x) = 0.5
```

Covariates:

- Observed data: each coordinate is drawn from `Uniform(support0[0], support0[1])`.
- Target/integration data: each coordinate is drawn from `Uniform(support[0], support[1])`.
- `inverse_CDF(u)` maps unit-cube inputs to the configured target support.

### Constant-Gap Multivariate Expansions

For `expansion in {"tan2", "sinh2", "rational"}`:

```text
mu0(x, d) = B(x1, ..., xK) + d*C
h0(x) = C
```

Currently `C = 1.0`.

The baseline `B` uses the first `K` coordinates:

- `tan2`: powers `x_j^(2j)` with coefficients from the Taylor series of `tan(z)^2`.
- `sinh2`: powers `x_j^(2j)` with coefficients from the Taylor series of `sinh(z)^2`.
- `rational`: powers `x_j^j` with coefficients `2, 3, ..., K+1`.

### Variable-Gap Univariate Expansions

For `expansion in {"hyperbolic", "exp_pm"}`, the model uses `z = X1`:

```text
mu0(x, d) = B(z) + d*C(z)
h0(x) = C(z)
```

- `hyperbolic`:
  - `A(z)` approximates `cosh(z)`.
  - `B(z)` approximates `sinh(z)`.
  - `C(z)` approximates `exp(-z)`.
- `exp_pm`:
  - `A(z)` approximates `exp(z)`.
  - `B(z)` approximates `exp(-z)`.
  - `C(z)` approximates `2*sinh(z)`.

`K` controls the Taylor truncation degree/order.
