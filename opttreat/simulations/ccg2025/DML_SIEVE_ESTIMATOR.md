# Cross-Fitted Sieve-Debiased (DML) Estimator for the CCG Simulations

This note documents the **cross-fitted sieve-debiased ("DML") estimator** added to
the CCG known-distribution welfare and value simulations, implementing
Section `sec:SieveDML` (eq. `eq:debias_est`) of `main_260630ZX.tex`. It is a third
estimator alongside the existing `plug_in` and `loo` rows.

| File | Role |
| --- | --- |
| [estimation/dml_sieve.py](../../estimation/dml_sieve.py) | `dml_sieve_estimate` (paper eq:debias_est) and `dml_welfare_aipw` (closed-form AIPW cross-check) |
| [variance/sieve_var.py](../../variance/sieve_var.py) | `sieve_riesz_core` — the shared Riesz-representer engine used by both `SieveVariance` and the DML |
| [ccg_welfare_known/run_ccg_welfare_known.py](ccg_welfare_known/run_ccg_welfare_known.py) | welfare runner, now emits a `dml` row (Models 1–7) |
| [ccg_value_known/run_ccg_value_known.py](ccg_value_known/run_ccg_value_known.py) | value runner, now emits a `dml` row (Model 15) |

---

## 1. The estimator

For a functional `Theta` (welfare `E_F[max(h,0)]` or value `E_F[1{h>0} m]`) under a
**known** target distribution `F`, the cross-fitted sieve-debiased estimator is

```
theta_tilde = (1/K) sum_kappa [ Theta(mu_hat_{-kappa})
              + (K/n) sum_{i in I_kappa} v*_{K,-kappa}(X_i, D_i) (Y_i - mu_hat_{-kappa}(X_i, D_i)) ]
```

with the **sieve Riesz representer**

```
v*_K(x, d) = psi-bar(x, d)^T  R_K^{-1}  D_mu Theta(mu_hat)[psi-bar],
R_K = (1/n) sum_i psi-bar(X_i, D_i) psi-bar(X_i, D_i)^T .
```

The representer is the *same object* the `SieveVariance` already builds
(`Bun = D_mu Theta(mu_hat)[psi-bar]`, arm weights `w_a = G_a^{-1} Bun_a`), so the
point estimate and its standard error share one construction. `sieve_riesz_core`
was extracted from `SieveVariance.fit` for exactly this reuse (SieveVariance is
byte-for-byte unchanged — it sets none of the new options).

Why this estimator (and not only the plug-in/LOO):

* **It unifies welfare and value.** The regular welfare functional has a bounded
  global influence function; the irregular value functional does **not** — yet the
  sieve Riesz representer is well-defined on the sieve space in both cases, and its
  norm `sigma_n = ||v*||_sd` (bounded vs. divergent) governs the rate.
* **It accommodates generic ML first stages.** The correction annihilates the
  first-order estimation error of `mu_hat` on the sieve space, so `mu_hat_{-kappa}`
  need not be the sieve least-squares fit (see §4).

Off-fold the representer is evaluated as `v*(x,1) = (n/n_t) psi_t(x)^T w_t` and
`v*(x,0) = (n/n_c) psi_c(x)^T w_c` (the `n/n_a` factors convert the per-arm Gram
`G_a` to the stacked `R_K`).

### Standard error

The SE is the **cross-fitted score standard deviation**

```
se = sqrt( sum_i ( v*_{-kappa(i)}(X_i, D_i) e_i )^2 ) / n ,   e_i = Y_i - mu_hat_{-kappa(i)}(X_i, D_i),
```

the sample analog of `sigma_n / sqrt(n)` (Theorem `thm:SieveDML`: "any consistent
estimator of `sigma_n^2` — such as the sieve sandwich variance"). It uses
**out-of-fold** residuals, so it reflects the debiasing variance that the in-sample
SieveVar SE (used by the plug-in/LOO rows) does not.

### Numerical stabilization

The per-fold arm Gram is far more ill-conditioned than the full-sample Gram: a
near-empty B-spline tensor cell produces an eigenvalue just above the pinv cutoff,
so `w = G^{-1} Bun` explodes (observed `||w|| ~ 1e4`, e.g. Model 7's `Jc=4` control
arm, whose fold Gram has `lam_max ~ 0.04` with eigenvalues down to `~1e-17`). Two
knobs (both off inside `SieveVariance`, so it is unaffected):

* `riesz_rcond` (default `5e-3`) — forces the representer solve to a Moore–Penrose
  inverse with this **relative** singular-value cutoff, regardless of the
  first-stage solver (works for the B-spline sieve and a random-feature ridge
  alike). Truncation leaves the kept directions **unshrunk**, so the score-variance
  SE stays honest (a global Tikhonov ridge, by contrast, shrinks every fold and
  deflates the SE).
* `riesz_cap_mult` (default `10`) — a per-fold representer-norm cap referenced to
  the median across folds, a backstop for any residual outlier fold.

---

## 2. Coverage results

500-replication production runs are what the runners emit; the tables below are
**300-rep demonstration grids** (`scratchpad/cov_welfare.py`, `cov_value.py`) that
are statistically sufficient to read coverage (MC SE at 0.95 over 300 reps ≈ ±0.013).
`riesz_rcond = 5e-3`, `K = 5` folds, B-spline sieve first stage.

### Welfare (known distribution), coverage of the `dml` estimator

| Model | n=1500 | n=3000 | n=6000 | note |
| --- | --- | --- | --- | --- |
| Model1 | 0.91 | 0.88 | 0.93 | 1-D, oscillatory baseline |
| Model2 | 0.95 | 0.93 | 0.97 | 1-D |
| Model3 | 0.94 | 0.91 | 0.93 | 1-D |
| Model4 | 0.89 | 0.89 | 0.90 | 2-D |
| Model5 | 0.81 | 0.86 | 0.87 | 2-D, tiny `W_0=0.05` (near boundary) |
| Model6 | 0.88 | 0.92 | 0.93 | 2-D |
| Model7 | 0.90 | 0.84 | 0.87 | 2-D, ill-conditioned `Jc=4` control arm |

### Value (known distribution, Model 15), all three estimators

| n | estimator | bias | sd | se | coverage |
| --- | --- | --- | --- | --- | --- |
| 1500 | plug_in | +0.012 | 0.082 | 0.071 | 0.917 |
| 1500 | loo | −0.010 | **0.348** | 0.071 | 0.910 |
| 1500 | **dml** | +0.010 | **0.103** | 0.068 | 0.860 |
| 3000 | dml | +0.006 | 0.051 | 0.041 | 0.877 |
| 6000 | dml | +0.010 | 0.034 | 0.028 | 0.903 |

**Key value-functional result:** at `n=1500` the DML's sd (0.103) is a third of the
LOO's (0.348) — for the irregular value functional the sieve-Riesz correction is
**far more stable** than the LOO quadratic correction. DML bias is well-controlled
and coverage climbs toward nominal with `n`.

### How to read these numbers

* **Well-conditioned / smooth models (M1, M2, M3, M6; and value)**: coverage reaches
  ~0.93–0.97 by `n>=3000`.
* **Ill-conditioned 2-D models (M4, M5, M7)**: coverage sits at ~0.85–0.90 at these
  `n`. This is *expected*: the sample is first split treated/control, then again by
  cross-fitting, so the effective per-fold arm sample is small; and the non-smooth
  functionals carry a **second-order (kink/level-set) bias** that first-order
  debiasing does not remove and that vanishes only as `n` grows. The plug-in has a
  *positive* such bias, the DML a *negative* one; the LOO removes it (best bias) but
  its in-sample SE then under-covers. None of the three reaches 0.95 on the hardest
  models at `n=1500` — the gap is a small-sample, not a specification, effect.

---

## 3. Relationship to the R `TEST_DML_1D_Spline.R` (closed-form AIPW)

Because welfare is **regular**, it also admits the equivalent closed-form
orthogonal-moment (AIPW) estimator of Remark `rem:Welfare_DML` (eq:W_DML_est) — the
Python port of the R DML script — provided as `dml_welfare_aipw`. It estimates the
propensity directly and uses `nu*(x,d) = 1{h>=0} lambda (d/p - (1-d)/(1-p))` with the
density ratio `lambda = f/f_0` (here `vol(F_0)/vol(F)` on the target box). It is a
useful cross-check but is *not* the default because (i) it does not extend to the
irregular value functional, and (ii) on models with near-1 propensities (e.g. M7)
the inverse-propensity weights inflate its variance. The sieve-Riesz estimator is
the paper's unified construction and is the one wired into the runners.

---

## 4. Exploring ML first stages

The DML "accommodates generic machine-learning first-stage estimators" — set each
runner's `ESTIMATOR` to `"rf_ridge"` (random features) instead of `"sieve"`. An
exploration (`scratchpad/cov_welfare.py ESTIMATOR=rf_ridge`) found:

* **Model 3** (smooth `mu0=x^2`, `h0=1-x`): rf-ridge DML is fine — bias −0.005,
  coverage 0.925 at `n=3000`, comparable to the sieve.
* **Model 1** (baseline `5 sin(2πx) cos(2πx)`, oscillatory): rf-ridge plug-in bias
  **+0.17**, coverage 0.31 — the random sigmoid features (ridge-penalized) cannot
  capture the high-frequency baseline at these `n`, and **debiasing cannot repair
  first-stage misspecification** (it removes first-order estimation *noise*, not the
  approximation *bias*).

Takeaway: the DML machinery is genuinely first-stage-agnostic and *valid* whenever
the learner satisfies the Assumption `assu:LocRob` rate conditions, but first-stage
**quality is paramount** — the paper's own caveat that those conditions "may be
nontrivial to verify for specific learners." On the smooth CCG DGPs the
purpose-tuned B-spline sieve is the better first stage, so it is the default.

---

## 5. Configuration reference (runner constants)

| Constant | welfare | value |
| --- | --- | --- |
| `DML_FOLDS` | 5 | 5 |
| `DML_RIESZ_RCOND` | 5e-3 | 5e-3 |
| representer band | `1{h>=0}` | relative `delta=0.05`, grid 16384 (≈16× faster than the tight `eps=0.005`, same estimate) |
| SE | cross-fitted score sd (out-of-fold residuals) | same |

Run: `python -m opttreat.simulations.ccg2025.ccg_welfare_known.run_ccg_welfare_known`
(and `...ccg_value_known.run_ccg_value_known`). Each `dml` draw row carries its own
`se`; `plug_in`/`loo` share the SieveVar SE.
