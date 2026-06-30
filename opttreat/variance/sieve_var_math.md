# Math for `SieveVariance`

This note explains how `opttreat/variance/sieve_var.py` constructs the
sieve variance estimate and the corresponding standard error.

The estimator starts from a first-stage fit with separate treated and control
regressions. Write the fitted CATE as

```math
\hat h(x)=\hat\mu_t(x)-\hat\mu_c(x)
       =\psi_t(x)'\hat\beta_t-\psi_c(x)'\hat\beta_c.
```

For the treated sample, `Psi_t` is the matrix with rows `psi_t(X_i)'` and
residual vector `e_t`. For the control sample, `Psi_c` is the matrix with rows
`psi_c(X_i)'` and residual vector `e_c`.

The code treats the coefficient vector as

```math
\beta=(\beta_t',\beta_c')',
```

so a coefficient perturbation changes `h` through

```math
d h(x)
=
\psi_t(x)'d\beta_t-\psi_c(x)'d\beta_c.
```

This is why the integration basis used by the code is

```math
b(x)=
\begin{bmatrix}
\psi_t(x) \\
-\psi_c(x)
\end{bmatrix}.
```

In the implementation this is

```python
bases = np.hstack([Psi_t_int, -Psi_c_int])
```

where `Psi_t_int` and `Psi_c_int` are the two feature maps evaluated on the
integration rows.

## Integration Rows

The derivative of the target parameter is approximated on rows called
`X_int`.

For known target distributions, `X_int` is a Sobol grid:

```math
X_j = T(U_j), \qquad U_j \in [0,1]^d,
```

where `T` is `options["transform"]`. If no transform is supplied, the identity
map is used.

For unknown target distributions, `X_int` is not a Sobol grid. It is taken from
`estimator_output["X_eval"]` if available, otherwise from
`estimator_output["X_all"]`.

Let

```math
M = n_{\mathrm{int}}
```

be the number of integration rows and let

```math
\hat h_j = \hat h(X_j).
```

## The Derivative Vector `Bun`

The central object is `Bun`, the finite-dimensional derivative of the target
parameter with respect to the sieve coefficients. It has one block for the
treated coefficients and one block for the control coefficients:

```math
\hat B =
\begin{bmatrix}
\hat B_t \\
\hat B_c
\end{bmatrix}.
```

Because the control basis enters `h` with a minus sign, `\hat B_c` already
contains that sign.

### Welfare

For welfare, the target is

```math
W(h)=\int \max\{h(x),0\}\,dF(x).
```

The directional derivative is

```math
D W(h)[s]
=
\int 1\{h(x)\ge 0\}s(x)\,dF(x).
```

The code approximates this derivative by

```math
\hat B
=
\frac{1}{M}
\sum_{j=1}^M
1\{\hat h_j\ge 0\} b(X_j).
```

In code this is the `good = h_int >= 0.0` branch:

```python
Bun = bases[good, :].sum(axis=0) / n_int
```

If no integration row has `h_hat >= 0`, the code sets `Bun` to zero.

### Value

For value, the target is

```math
V(h)=\int 1\{h(x)>0\}m(x)\,dF(x).
```

This functional has a kink at the decision boundary `h(x)=0`. Formally, its
derivative is concentrated on the boundary. The code uses an epsilon-band
approximation:

```math
\delta_0(\hat h(x))
\approx
\frac{1}{2\epsilon}1\{|\hat h(x)| < \epsilon\}.
```

Therefore the implemented derivative vector is

```math
\hat B
=
\frac{1}{2\epsilon}
\frac{1}{M}
\sum_{j=1}^M
1\{|\hat h_j| < \epsilon\}m(X_j)b(X_j).
```

In code this is the value branch:

```python
Bun = (bases[good, :] * value_weight[good, None]).sum(axis=0) / n_int
Bun = Bun / (2.0 * eps)
```

The bandwidth follows the same `_boundary_band` convention used by the value
parameter. It is either the fixed bandwidth

```math
\epsilon = \texttt{options["loo_eps"]}
```

when `loo_eps` is supplied, or otherwise

```math
\epsilon =
\max\{
\texttt{delta}\cdot \operatorname{sd}(\hat h(X_j)),\ 10^{-12}
\},
```

with `delta` defaulting to `0.05`. The mask selects exactly $|\hat h_j| < \epsilon$.

For known-distribution value targets, the Sobol grid can be expanded until the
epsilon band has enough points. This is controlled by
`variance_expand_sobol`, `variance_min_band`, `variance_max_sobol`, and
`variance_sobol_expand_factor`.

## Solving for the Sieve Riesz Weights

After `Bun` is split into treated and control blocks, the code solves two
arm-specific linear systems. Define

```math
\hat G_t = \Psi_t'\Psi_t+\alpha I,
\qquad
\hat G_c = \Psi_c'\Psi_c+\alpha I.
```

Then

```math
\hat w_t = \hat G_t^{-1}\hat B_t,
\qquad
\hat w_c = \hat G_c^{-1}\hat B_c.
```

These are the finite-dimensional Riesz weights. In code:

```python
weights_t = gram_t_inv @ Bun_t
weights_c = gram_c_inv @ Bun_c
```

If `solver` is `pinv`, or if `alpha == 0`,
the code uses

```math
(\Psi_a'\Psi_a)^+
```

instead of the ridge inverse.

Notice the normalization: the code uses `Psi_a.T @ Psi_a`, not
`(Psi_a.T @ Psi_a) / n_a`. Because `Bun` is an average but the Gram matrix is
an unnormalized sum, the resulting weights are on the estimator standard-error
scale.

## Residual Influence Contributions

For every treated observation, the implemented influence contribution is

```math
\hat\ell_{t,i}
=
\hat e_{t,i}\,\psi_t(X_{t,i})'\hat w_t.
```

For every control observation, it is

```math
\hat\ell_{c,i}
=
\hat e_{c,i}\,\psi_c(X_{c,i})'\hat w_c.
```

The control sign is already inside `\hat B_c`, because `b(x)` used
`-\psi_c(x)` when constructing `Bun`.

In code:

```python
influence_t = e_t * (Psi_t @ weights_t)
influence_c = e_c * (Psi_c @ weights_c)
influence = np.concatenate([influence_t, influence_c])
```

## Scale of the Returned Number

The code uses unnormalized Gram matrices, such as `Psi_t.T @ Psi_t`, while
`Bun` is an average over the integration rows. As a result, the weights are of
order `1 / n_fit`, the influence contributions are on the estimator scale, and
the sum of squared contributions is of order `1 / n_fit`.

So `fit()` returns the variance used directly for the estimator's standard
error:

```math
\widehat{\operatorname{se}}(\hat\theta)
=
\sqrt{\widehat{\operatorname{Var}}(\hat\theta)}.
```

If one wants the variance of a normalized statistic like
`\sqrt{n}(\hat\theta-\theta)`, the corresponding scale is obtained by
multiplying the returned estimator-scale variance by the relevant sample size.

## Known-Distribution Variance

For a known target distribution, the returned sieve variance is just the sum of
squared residual influence contributions:

```math
\widehat{\operatorname{Var}}_{\mathrm{sieve}}(\hat\theta)
=
\sum_i \hat\ell_i^2.
```

In code:

```python
var_sieve = float(np.sum(influence ** 2))
var_total = var_sieve
```

The standard error stored by `fit()` is

```math
\widehat{\operatorname{se}}(\hat\theta)
=
\sqrt{\widehat{\operatorname{Var}}_{\mathrm{sieve}}(\hat\theta)}.
```

That is:

```python
self.var_hat_ = float(var_total)
self.se_ = float(np.sqrt(var_total))
```

## Unknown-Distribution Extra Term

For unknown target distributions, the target distribution itself is empirical.
The code therefore adds a second component for the randomness of the evaluation
covariates.

First define the empirical target value on each evaluation row:

For welfare:

```math
\hat g_j = \max\{\hat h(X_j),0\}.
```

For value:

```math
\hat g_j = 1\{\hat h(X_j)>0\}m(X_j).
```

The code estimates the empirical-distribution variance as

```math
\widehat{\operatorname{Var}}_{\mathrm{emp}}
=
\frac{1}{M}
\sum_{j=1}^M
(\hat g_j-\bar g)^2.
```

Then it combines the nuisance-estimation piece and the empirical-distribution
piece as

```math
\widehat{\operatorname{Var}}_{\mathrm{total}}
=
\widehat{\operatorname{Var}}_{\mathrm{sieve}}
+
\frac{\widehat{\operatorname{Var}}_{\mathrm{emp}}}{M}
```

when `n_fit == M`. More generally, the code allows the number of first-stage
fit observations and evaluation observations to differ and uses

```math
\widehat{\operatorname{Var}}_{\mathrm{total}}
=
\widehat{\operatorname{Var}}_{\mathrm{sieve}}
\left(\frac{n_{\mathrm{fit}}}{M}\right)
+
\frac{\widehat{\operatorname{Var}}_{\mathrm{emp}}}{M}.
```

In code:

```python
var_total = var_sieve * (n_fit / n_eval) + var_empirical / n_eval
```

where `n_eval` is `M`.

## Bootstrap Critical Value

`bootstrap_critical_value()` reuses the same `influence` vector as `fit()`.
For known target distributions, it draws multipliers `xi_i` and forms

```math
T^*
=
\left|
\frac{\sum_i \xi_i\hat\ell_i}
{\widehat{\operatorname{se}}}
\right|.
```

The critical value is the empirical `(1-alpha)` quantile of these bootstrap
statistics.

The bootstrap is intentionally not implemented for unknown target
distributions, because those targets also include the empirical-distribution
variance term above.

## Code Map

- `_known_integration_grid`: builds the Sobol integration grid for known
  target distributions.
- `_value_band_mask`: chooses the value-functional epsilon band.
- `_linearization_components`: computes `Bun`, Gram inverses, influence
  contributions, and `var_sieve`.
- `fit`: adds the unknown-distribution empirical term when needed, then stores
  `var_hat_` and `se_`.
- `bootstrap_critical_value`: builds multiplier-bootstrap critical values from
  the same influence vector.
