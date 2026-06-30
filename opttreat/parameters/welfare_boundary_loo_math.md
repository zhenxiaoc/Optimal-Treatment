# Math for `_welfare_band_loo`

This note explains the boundary-band LOO correction used by
`_welfare_band_loo`. It is selected when
`loo_method="band"` or `"boundary_band"`.

The welfare functional is

```math
W(h)
=
\int \max\{h(x),0\}\,dP(x),
```

where

```math
h(x)=\mu(x,1)-\mu(x,0).
```

The plug-in estimator is

```math
W(\hat h)
=
\int \max\{\hat h(x),0\}\,dP(x).
```

In the code this integral is represented by an average over evaluation rows:

```math
W(\hat h)
\approx
\frac{1}{N_{\mathrm{eval}}}
\sum_{j=1}^{N_{\mathrm{eval}}}
\max\{\hat h(X_j),0\}.
```

For known-distribution welfare, the rows `X_j` are Sobol integration points.
For unknown-distribution welfare, they are usually the observed covariates.

## Directions

The nuisance object is the pair

```math
\mu(x)=\{\mu(x,1),\mu(x,0)\}.
```

A perturbation of this nuisance is

```math
\nu(x)=\{\nu(x,1),\nu(x,0)\}.
```

Since

```math
h(x)=\mu(x,1)-\mu(x,0),
```

the induced perturbation of `h` is

```math
\nu(x,1)-\nu(x,0).
```

For a treated observation, the LOO direction changes only the treated arm:

```math
\nu_i(x)=(s_{i,t}(x),0),
```

so the induced direction for `h` is `s_{i,t}`. For a control observation,

```math
\nu_i(x)=(0,s_{i,c}(x)),
```

so the induced direction for `h` is `-s_{i,c}`. The second derivative is
quadratic in the direction, so

```math
D_h^2W(\hat h)[-s,-s]
=
D_h^2W(\hat h)[s,s].
```

That is why the treated and control arm corrections add.

## Exact Boundary Formula

The kink of the welfare functional occurs on the boundary

```math
\mathcal M_{\hat h}
=
\{x:\hat h(x)=0\}.
```

For one induced direction `s`, the first derivative is

```math
DW(\hat h)[s]
=
\int 1\{\hat h(x)>0\}s(x)\,dP(x).
```

The second derivative comes from differentiating the indicator. Formally,

```math
\frac{d}{dt}1\{\hat h(x)+t s(x)>0\}\bigg|_{t=0}
=
\delta(\hat h(x))s(x),
```

so

```math
D_h^2W(\hat h)[s,s]
=
\int \delta(\hat h(x))s(x)^2\,dP(x).
```

If `P` has density `p(x)` with respect to Lebesgue measure and
`\|\nabla_x\hat h(x)\|>0` on the boundary, the coarea formula gives

```math
D_h^2W(\hat h)[s,s]
=
\int_{\mathcal M_{\hat h}}
\frac{s(x)^2 p(x)}
{\|\nabla_x\hat h(x)\|}
d\mathcal H^{d-1}(x).
```

Unlike the value functional, no curvature or `grad s` term appears here. The
welfare kink is just the second derivative of the scalar map
`z -> max(z, 0)`.

## Boundary-Band Approximation

The code approximates the Dirac delta by a fixed-width band:

```math
\delta(\hat h(x))
\approx
\frac{1\{|\hat h(x)|<\epsilon\}}{2\epsilon}.
```

Therefore,

```math
D_h^2W(\hat h)[s,s]
\approx
\frac{1}{2\epsilon N_{\mathrm{eval}}}
\sum_{j:|\hat h(X_j)|<\epsilon}
s(X_j)^2.
```

This is exactly the approximation implemented by
`_welfare_second_derivative_matrix`.

The bandwidth is chosen in `_boundary_band`:

```math
\epsilon
=
\texttt{loo\_eps}
```

if `loo_eps` is supplied. Otherwise,

```math
\epsilon
=
\texttt{loo\_delta0}\cdot \operatorname{sd}\{\hat h(X_j)\},
```

with a small positive floor for numerical stability.

For known-distribution welfare, `_sobol_grid_with_boundary_points` can expand
the Sobol grid until enough rows fall inside this fixed band.

## Coefficient-Space Form

For each arm, directions are represented in a fitted feature basis:

```math
s(x)=b(x)^\top a.
```

On the retained boundary-band rows, let

```math
B_{\mathrm{band}}
=
\begin{bmatrix}
b(X_{j_1})^\top\\
\vdots\\
b(X_{j_m})^\top
\end{bmatrix},
\qquad
|\hat h(X_{j_\ell})|<\epsilon.
```

Then

```math
\frac{1}{2\epsilon N_{\mathrm{eval}}}
\sum_{\ell=1}^m
s(X_{j_\ell})^2
=
a^\top
\left[
\frac{1}{2\epsilon N_{\mathrm{eval}}}
B_{\mathrm{band}}^\top B_{\mathrm{band}}
\right]
a.
```

The code calls this matrix `A`:

```math
A
=
\frac{1}{2\epsilon N_{\mathrm{eval}}}
B_{\mathrm{band}}^\top B_{\mathrm{band}}.
```

Thus `_welfare_second_derivative_matrix` builds `A` such that

```math
a^\top A a
\approx
D_h^2W(\hat h)[s,s].
```

## Arm-Level LOO Directions

For one arm, let `Psi` be the fitted feature matrix and let
`n_arm` be the number of observations in that arm. The normalized Gram matrix
is

```math
G
=
\frac{\Psi^\top\Psi+\alpha I}{n_{\mathrm{arm}}}
```

for ridge, and

```math
G
=
\frac{\Psi^\top\Psi}{n_{\mathrm{arm}}}
```

for pinv/OLS.

The code uses the leave-one-out coefficient direction

```math
a_i
=
\frac{1}{n_{\mathrm{arm}}}
G^{-1}b(X_i).
```

Therefore the induced function direction is

```math
s_i(x)
=
b(x)^\top a_i.
```

The diagonal leverage is

```math
H_{ii}
=
\frac{1}{n_{\mathrm{arm}}}
b(X_i)^\top G^{-1}b(X_i),
```

and the leave-one-out residual is

```math
e_i^{(-i)}
=
\frac{e_i}{1-H_{ii}}.
```

For one arm, `_arm_quadratic_correction` computes

```math
\frac{1}{2}
\sum_{i\in\mathrm{arm}}
D_h^2W(\hat h)[s_i,s_i]
\left(e_i^{(-i)}\right)^2.
```

The helper evaluates the quadratic form as

```math
D_h^2W(\hat h)[s_i,s_i]
\approx
a_i^\top A a_i.
```

Chunking only controls memory use. It does not change the mathematical sum.

## Final Boundary-Band LOO Estimator

The boundary-band LOO estimator is

```math
\widehat W_{\mathrm{LOO}}
=
W(\hat h)
-
\frac{1}{2}
\sum_{i:D_i=1}
D_h^2W(\hat h)[\hat s_{i,t},\hat s_{i,t}]
\left(\hat e_{i,t}^{(-i)}\right)^2
-
\frac{1}{2}
\sum_{i:D_i=0}
D_h^2W(\hat h)[\hat s_{i,c},\hat s_{i,c}]
\left(\hat e_{i,c}^{(-i)}\right)^2.
```

The factors `1 / n_t` and `1 / n_c` are already inside the direction
coefficients `a_i`. If the directions were defined without those factors,
explicit `1 / n_t^2` and `1 / n_c^2` factors would appear in the two arm
sums.

## Relation to the Central-Difference Method

`_welfare_central_difference_loo` estimates the same diagonal second
derivative numerically by perturbing the vector `h_vals` in each LOO
direction. The boundary-band method instead uses the analytic kink formula

```math
D_h^2W(\hat h)[s,s]
\approx
\frac{1}{2\epsilon N_{\mathrm{eval}}}
\sum_{j:|\hat h(X_j)|<\epsilon}s(X_j)^2.
```

The boundary-band method is simpler and faster because it constructs one
matrix `A` per arm and then reuses it for all LOO directions.

## Code Map

- `_boundary_band`: chooses `eps` and keeps rows with
  `|h_hat(X_j)| < eps`.
- `_sobol_grid_with_boundary_points`: for known-distribution welfare, expands
  the Sobol grid until the boundary band has enough rows.
- `_welfare_second_derivative_matrix`: builds the coefficient-space matrix
  `A = B_band.T @ B_band / (2 eps N_eval)`.
- `_normalized_arm_gram_inverse`: builds `G^{-1}` for one treatment arm.
- `_arm_quadratic_correction`: sums one arm's quadratic LOO correction.
- `_welfare_band_loo`: combines the boundary band, the two arm corrections,
  and the plug-in welfare estimate.
- `_welfare_central_difference_loo`: optional finite-difference alternative.
- `WelfareKnownDist.loo`: uses the boundary-band method by default.
- `WelfareUnknownDist.loo`: uses central difference by default, but can use
  the boundary-band method when `loo_method="band"`.
