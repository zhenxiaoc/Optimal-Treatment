# Math for Value Boundary LOO Corrections

This note explains the optional LOO corrections for the value functional:
the boundary-band correction in `_loo_value_boundary_correction`, used when
`loo_method="band"` or `"boundary_band"`, and the central finite-difference
correction in `_loo_value_central_difference`, used when
`loo_method="central_difference"`.

The goal is to correct the plug-in estimator of the treatment-control value
functional

```math
\overline V(\mu)
=
\int
1\{\mu(x,1)-\mu(x,0)>0\}
m(x)\,dx.
```

The CATE is

```math
h(x)=\mu(x,1)-\mu(x,0),
```

so

```math
\overline V(\mu)=V(h),
\qquad
V(h)=\int 1\{h(x)>0\}m(x)\,dx.
```

The plug-in value is

```math
V(\hat h)
=
\int 1\{\hat h(x)>0\}m(x)\,dx.
```

In the code this integral is represented by an average over evaluation rows:

```math
V(\hat h)
\approx
\frac{1}{N_{\mathrm{eval}}}
\sum_{j=1}^{N_{\mathrm{eval}}}
1\{\hat h(X_j)>0\}m(X_j).
```

The code takes `m` to be this single target-measure weight and evaluates
$m(X_j)$ directly.

## Directions

The nuisance object is $\mu(x,d)$, not $h(x)$. A perturbation of $\mu$ is

```math
\nu(x)=\{\nu(x,1),\nu(x,0)\}.
```

Because

```math
h(x)=\mu(x,1)-\mu(x,0),
```

the induced perturbation of $h$ is

```math
\nu(x,1)-\nu(x,0).
```

Therefore,

```math
D_\mu^2\overline V(\mu)[\nu_1,\nu_2]
=
D_h^2V(h)
[
\nu_1(\cdot,1)-\nu_1(\cdot,0),
\nu_2(\cdot,1)-\nu_2(\cdot,0)
].
```

For a treated observation, the LOO direction has the form

```math
\nu_i(x)=(s_{i,t}(x),0),
```

so the induced direction for $h$ is $s_{i,t}(x)$. For a control
observation, the LOO direction has the form

```math
\nu_i(x)=(0,s_{i,c}(x)),
```

so the induced direction for $h$ is $-s_{i,c}(x)$. The LOO correction uses
the diagonal second derivative, and

```math
D_h^2V(\hat h)[-s_{i,c},-s_{i,c}]
=
D_h^2V(\hat h)[s_{i,c},s_{i,c}],
```

so the treated and control corrections add.

## Exact Surface Formula

The boundary of the value rule is

```math
\mathcal M_{\hat h}
=
\{x:\hat h(x)=0\}.
```

For one induced direction $s$, Chen and Gao's shape-calculus formula gives

```math
\begin{aligned}
D_h^2V(\hat h)[s,s]
=
&-\int_{\mathcal M_{\hat h}}
s(x)^2
\left[
\frac{1}{\|\nabla_x\hat h(x)\|}
\frac{\nabla_x\hat h(x)}{\|\nabla_x\hat h(x)\|}
\cdot
\nabla_x
\left\{
\frac{m(x)}{\|\nabla_x\hat h(x)\|}
\right\}
+
\frac{m(x)}{\|\nabla_x\hat h(x)\|^2}
\hat H(x)
\right]
d\mathcal H^{d-1}(x)
\\
&-2\int_{\mathcal M_{\hat h}}
\frac{m(x)}{\|\nabla_x\hat h(x)\|^2}
s(x)
\frac{\nabla_x\hat h(x)}{\|\nabla_x\hat h(x)\|}
\cdot
\nabla_xs(x)
d\mathcal H^{d-1}(x),
\end{aligned}
```

where

```math
\hat H(x)
=
\operatorname{div}
\left(
\frac{\nabla_x\hat h(x)}
{\|\nabla_x\hat h(x)\|}
\right).
```

Here is where the two terms come from. The first derivative at
$\hat h+ts$ is

```math
DV(\hat h+t s)[s]
=
\int_{\{\hat h+t s=0\}}
\frac{s(x)m(x)}
{\|\nabla_x(\hat h+t s)(x)\|}
d\mathcal H^{d-1}(x).
```

Differentiating this with respect to $t$ moves the boundary
$\{\hat h+t s=0\}$. The moving-boundary part produces

```math
\frac{s(x)}{\|\nabla_x\hat h(x)\|}
\frac{\nabla_x\hat h(x)}
{\|\nabla_x\hat h(x)\|}
\cdot
\nabla_x
\left\{
\frac{s(x)m(x)}
{\|\nabla_x\hat h(x)\|}
\right\}
```

and the curvature part produces

```math
\frac{s(x)m(x)}
{\|\nabla_x\hat h(x)\|}
\frac{s(x)}
{\|\nabla_x\hat h(x)\|}
\hat H(x).
```

The product rule splits

```math
\nabla_x
\left\{
\frac{s(x)m(x)}
{\|\nabla_x\hat h(x)\|}
\right\}
```

into

```math
s(x)
\nabla_x
\left\{
\frac{m(x)}
{\|\nabla_x\hat h(x)\|}
\right\}
+
\frac{m(x)}
{\|\nabla_x\hat h(x)\|}
\nabla_xs(x).
```

The first piece above, together with the curvature term, gives the first
integral in $D_h^2V(\hat h)[s,s]$. The piece involving $\nabla_xs(x)$ gives
the second integral.

## Boundary-Band Approximation

The code does not explicitly parameterize $\mathcal M_{\hat h}$. Instead it
keeps rows close to the boundary:

```math
|\hat h(X_j)|<\epsilon.
```

The coarea approximation is

```math
\int_{\mathcal M_{\hat h}}\phi(x)d\mathcal H^{d-1}(x)
\approx
\sum_{j:|\hat h(X_j)|<\epsilon}
\frac{\|\nabla_x\hat h(X_j)\|}
{2\epsilon N_{\mathrm{eval}}}
\phi(X_j).
```

This is why `_loo_value_boundary_correction` computes

```math
\nabla_x\hat h(X_j),
\qquad
\Delta_x\hat h(X_j),
\qquad
\nabla_x\hat h(X_j)^\top
\nabla_x^2\hat h(X_j)
\nabla_x\hat h(X_j)
```

on the retained rows.

## Boundary Coefficient Used In Code

Start from the first integral in the exact surface formula. This is the part
that multiplies $s(x)^2$:

```math
\begin{aligned}
&-\int_{\mathcal M_{\hat h}}
s(x)^2
\left[
\frac{1}{\|\nabla_x\hat h(x)\|}
\frac{\nabla_x\hat h(x)}{\|\nabla_x\hat h(x)\|}
\cdot
\nabla_x
\left\{
\frac{m(x)}{\|\nabla_x\hat h(x)\|}
\right\}
+
\frac{m(x)}{\|\nabla_x\hat h(x)\|^2}
\hat H(x)
\right]
d\mathcal H^{d-1}(x).
\end{aligned}
```

The code evaluates this coefficient directly, using the curvature term

```math
\hat H(x)
:=
\operatorname{div}
\left(
\frac{\nabla_x\hat h(x)}
{\|\nabla_x\hat h(x)\|}
\right)
=
\frac{\Delta_x\hat h(x)}
{\|\nabla_x\hat h(x)\|}
-
\frac{
\nabla_x\hat h(x)^\top
\nabla_x^2\hat h(x)
\nabla_x\hat h(x)
}
{\|\nabla_x\hat h(x)\|^3}.
```

The other piece is

```math
\frac{\nabla_x\hat h(x)}{\|\nabla_x\hat h(x)\|}
\cdot
\nabla_x
\left\{
\frac{m(x)}{\|\nabla_x\hat h(x)\|}
\right\}.
```

Expanding only this derivative gives

```math
\begin{aligned}
&
\frac{\nabla_x\hat h(x)}{\|\nabla_x\hat h(x)\|}
\cdot
\nabla_x
\left\{
\frac{m(x)}{\|\nabla_x\hat h(x)\|}
\right\}
\\
&=
\frac{
\nabla_x\{m(x)\}^\top\nabla_x\hat h(x)
}
{\|\nabla_x\hat h(x)\|^2}
-
\frac{
m(x)
\nabla_x\hat h(x)^\top
\nabla_x^2\hat h(x)
\nabla_x\hat h(x)
}
{\|\nabla_x\hat h(x)\|^4}.
\end{aligned}
```

The second term is the derivative of
$1/\|\nabla_x\hat h(x)\|$. This is the only norm-derivative step needed
when we use $\hat H(x)$ directly.

The coefficient multiplying $s(x)^2$ is this derivative term divided by
$\|\nabla_x\hat h(x)\|$, plus the curvature term with
$\|\nabla_x\hat h(x)\|^2$ in the denominator. Substituting the expression for
$\hat H(x)$, this simplifies to

```math
\frac{
\nabla_x\{m(x)\}^\top\nabla_x\hat h(x)
}
{\|\nabla_x\hat h(x)\|^3}
+
m(x)
\left[
\frac{\Delta_x\hat h(x)}
{\|\nabla_x\hat h(x)\|^3}
-
2
\frac{
\nabla_x\hat h(x)^\top
\nabla_x^2\hat h(x)
\nabla_x\hat h(x)
}
{\|\nabla_x\hat h(x)\|^5}
\right].
```

The code evaluates $m(X_j)$ before computing the plug-in value or the
boundary-band LOO correction.

Putting this coefficient together with the coarea weight, the code
approximates the diagonal second derivative by

```math
\begin{aligned}
D_h^2V(\hat h)[s,s]
\approx
-
\sum_{j:|\hat h(X_j)|<\epsilon}
\frac{\|\nabla_x\hat h(X_j)\|}
{2\epsilon N_{\mathrm{eval}}}
\Bigg[
&
\left[
\frac{
\nabla_x\{m(X_j)\}^\top\nabla_x\hat h(X_j)
}
{\|\nabla_x\hat h(X_j)\|^3}
+
m(X_j)
\left\{
\frac{\Delta_x\hat h(X_j)}
{\|\nabla_x\hat h(X_j)\|^3}
-
2
\frac{
\nabla_x\hat h(X_j)^\top
\nabla_x^2\hat h(X_j)
\nabla_x\hat h(X_j)
}
{\|\nabla_x\hat h(X_j)\|^5}
\right\}
\right]
s(X_j)^2
\\
&+
2s(X_j)
\frac{m(X_j)\nabla_x\hat h(X_j)}
{\|\nabla_x\hat h(X_j)\|^3}
\cdot
\nabla_xs(X_j)
\Bigg].
\end{aligned}
```

This is the mathematical quantity that `_D2_V_matrix` represents
in coefficient space.

## Coefficient-Space Form

In the fitted model, each direction has the form

```math
s(x)=b(x)^\top a.
```

The helper `_D2_V_matrix` builds a matrix $A$ such that

```math
a^\top A a
```

equals the boundary-band approximation above for $s(x)=b(x)^\top a$.

This matrix form is useful because the LOO correction needs the same quadratic
form for many different leave-one-out directions.

## Central-Difference Approximation

As an alternative to the boundary-band surface formula, the code can use
`loo_method="central_difference"`. This path does not compute gradients,
Hessians, curvature, or a boundary band. Instead, it approximates each diagonal
second derivative directly from the plug-in value functional.

For known-distribution value LOO, when the caller does not supply evaluation
rows, the central-difference path uses the same adaptive Sobol grid as the
boundary-band path. It starts from `n_sobol`, sets
$\epsilon=$ `loo_eps` when supplied and otherwise
$\epsilon=\delta_0\operatorname{sd}\{\hat h(X_j)\}$, and increases the Sobol
grid until at least `loo_min_band` rows satisfy $|\hat h(X_j)|<\epsilon$, or
until `loo_max_sobol` is reached. If the caller supplies `X`, that grid is
used as-is.

For a direction $s$, define

```math
V_{\mathrm{eval}}(h)
=
\frac{1}{N_{\mathrm{eval}}}
\sum_{j=1}^{N_{\mathrm{eval}}}
1\{h(X_j)>0\}m(X_j).
```

With finite-difference step

```math
\eta
=
\delta_0\max\{\operatorname{sd}(\hat h(X_j)),10^{-12}\},
```

the central approximation is

```math
D_h^2V(\hat h)[s,s]
\approx
\frac{
V_{\mathrm{eval}}(\hat h+\eta s)
-
2V_{\mathrm{eval}}(\hat h)
+
V_{\mathrm{eval}}(\hat h-\eta s)
}
{\eta^2}.
```

Let $\ell_i$ be the LOO direction evaluated on the integration rows, including
its arm-specific $1/n_a$ factor:

```math
\ell_i(X_j)
=
b(X_j)^\top G_a^{-1}b(X_i)/n_a.
```

The paper's LOO formula uses the sample-size-scaled direction

```math
q_i(X_j)=n\ell_i(X_j),
```

and `_loo_value_central_difference` finite-differences directly along this raw
direction:

```math
D_h^2V(\hat h)[q_i,q_i]
\approx
\frac{
V_{\mathrm{eval}}(\hat h+\eta q_i)
-
2V_{\mathrm{eval}}(\hat h)
+
V_{\mathrm{eval}}(\hat h-\eta q_i)
}
{\eta^2}.
```

Since $\ell_i=q_i/n$, the central-difference LOO estimator is

```math
\widehat{\overline V}_{\mathrm{LOO,central}}
=
\overline V(\hat\mu)
-
\frac{1}{2n^2}
\sum_{a\in\{t,c\}}
\sum_{i\in a}
D_h^2V(\hat h)[q_i,q_i]
\left(\hat e_{i,a}^{(-i)}\right)^2.
```

This is algebraically the same diagonal quadratic correction as the boundary
formula, but with the second derivative estimated by finite differences of the
discontinuous value rule on the evaluation grid.

## Arm-Level LOO Directions

For each arm, let $\Psi$ be the fitted basis matrix for that arm. The
normalized arm Gram matrix is

```math
G=\frac{\Psi^\top\Psi+\alpha I}{n_{\mathrm{arm}}}
```

for ridge, or

```math
G=\frac{\Psi^\top\Psi}{n_{\mathrm{arm}}}
```

for pinv/OLS.

The code uses the leave-one-out coefficient direction

```math
a_i=\frac{1}{n_{\mathrm{arm}}}G^{-1}b(X_i),
```

so

```math
s_i(x)=b(x)^\top a_i.
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
\sum_{i\in \mathrm{arm}}
D_h^2V(\hat h)[s_i,s_i]
\left(e_i^{(-i)}\right)^2,
```

using the matrix $A$ for the evaluation of
$D_h^2V(\hat h)[s_i,s_i]$. Chunking changes only memory use; it does not
change the mathematical sum.

## Final LOO Estimator

The resulting LOO estimator is

```math
\widehat{\overline V}_{\mathrm{LOO}}
=
\overline V(\hat\mu)
-
\frac{1}{2}
\sum_{i:D_i=1}
D_h^2V(\hat h)[\hat s_{i,t},\hat s_{i,t}]
\left(\hat e_{i,t}^{(-i)}\right)^2
-
\frac{1}{2}
\sum_{i:D_i=0}
D_h^2V(\hat h)[\hat s_{i,c},\hat s_{i,c}]
\left(\hat e_{i,c}^{(-i)}\right)^2.
```

The factors $1/n_t$ and $1/n_c$ are already inside
$\hat s_{i,t}$ and $\hat s_{i,c}$. If the directions were defined without
those factors, explicit $1/n_t^2$ and $1/n_c^2$ factors would appear in the
two arm sums.

## Code Map

- `_boundary_band`: keeps rows with $|\hat h(X_j)|<\epsilon$.
- `_h_gradient_values`: computes $\nabla_x\hat h$.
- `_m_gradient_values`: computes $\nabla_x m$.
- `_h_hessian_values`: computes $\nabla_x^2\hat h$.
- `_h_laplacian_values`: computes
  $\Delta_x\hat h=\operatorname{tr}(\nabla_x^2\hat h)$.
- `_h_hessian_quadratic_values`: computes
  $\nabla_x\hat h^\top\nabla_x^2\hat h\nabla_x\hat h$.
- `_h_surface_divergence_values`: computes
  $\hat H=\operatorname{div}(\nabla_x\hat h/\|\nabla_x\hat h\|)$.
- `_D2_V_matrix`: builds $A$ for the boundary-band approximation
  to $D_h^2V(\hat h)[s,s]$, including the basis representation of
  $(\nabla_x\hat h/\|\nabla_x\hat h\|)^\top\nabla_xs$.
- `_arm_quadratic_correction`: sums the LOO quadratic terms for one arm.
- `_loo_value_boundary_correction`: combines the boundary band, the two arm
  corrections, and the plug-in value.
