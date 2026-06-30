rm(list = ls()) 

library(checkmate)
library(npiv)
library(mgcv)
library(qrng)
library(MASS)
library(Matrix)
source("helper.R")
source("spline.R")
set.seed(123)

#####################
## Data Generation ##
#####################
# n_obs (int) – Number of observations to simulate. Default is 200.
# p (int) – Dimension of covariates. Default is 30.
# support_size (int) – Number of relevant (confounding) covariates for determining treatment status. Default is 5.
# n_x (int) – Dimension of the heterogeneity. Can be either 1 or 2. Default is 1.
# binary_treatment (bool) – Indicates whether the treatment is binary. Default is False.
n_rep = 1000
n_obs = 1500
p <- support_size <- 5
data = list()
for (i_rep in seq_len(n_rep)) {
  data[[i_rep]] <- make_heterogeneous_data(n_obs = n_obs, p = p, 
                                           support_size = support_size,
                                           n_x = 2, binary_treatment = TRUE)
}


############
## True W ##
############
treatment_effect <- function(x) exp(2 * x[, 1]) + 3 * sin(4 * x[, 2])
M <- 1000
sobol_points <- sobol(M, d = p, randomize = "none")
W_true <- mean(pmax(treatment_effect(sobol_points), 0))
int_X <- sobol_points


#################################
## Estimate and Inference of W ##
#################################
# segments <- sieve_dim(n_obs = n_obs, p = p, 
#                       support_size = support_size,
#                       n_x = 2, binary_treatment = TRUE)

# Sieve Tuning
# Run a pilot simulation to determine sieve dimensions
# J.x.segments.c <- segments[1] + 1   # +1 for undersmooth           
# J.x.segments.t <- segments[2] + 1
J.x.segments.c <- 1   # +1 for undersmooth
J.x.segments.t <- 1
basis <- 'tensor'
knots <- 'uniform'
J.x.degree <- 3
X.min <- NULL
X.max <- NULL

# Parallel
library(future)
library(future.apply)
plan(multisession, workers = 2)
one_ite <- function(i,
                    data,
                    J.x.degree, J.x.segments.c, J.x.segments.t,
                    knots, basis) {
  
  df <- data[[i]]$data
  
  # Control + Treated
  Y <- df$y; D <- df$d
  X <- df[ , !(names(df) %in% c("d","y")), drop = FALSE]
  
  # Control
  df.c <- df[df$d == 0, , drop = FALSE]
  Y.c  <- as.numeric(df.c$y)
  X.c  <- df.c[ , !(names(df.c) %in% c("d","y")), drop = FALSE]
  
  # Treated
  df.t <- df[df$d == 1, , drop = FALSE]
  Y.t  <- as.numeric(df.t$y)
  X.t  <- df.t[ , !(names(df.t) %in% c("d","y")), drop = FALSE]
  
  # Trimmed (min-max overlap on all X columns)
  df_trim <- trim_minmax(df, d = "d", vars = NULL)$df_trim
  X_trim  <- df_trim[ , !(names(df_trim) %in% c("d","y")), drop = FALSE]
  
  
  # --- control arm spline fit
  Psi_X.c <- prodspline(
    x     = X.c,
    K     = cbind(rep(J.x.degree, NCOL(X.c)), rep(J.x.segments.c, NCOL(X.c))),
    knots = knots, basis = basis,
    x.min = NULL, x.max = NULL
  )
  # pivoted-QR fit (robust to collinearity)
  QRc <- qr(Psi_X.c, LAPACK = TRUE, tol = 1e-10)
  beta.c <- qr.coef(QRc, Y.c); beta.c[is.na(beta.c)] <- 0
  d_sieve.c <- ncol(Psi_X.c)
  
  Psi_int_X.c <- prodspline(
    x     = X.c, xeval = int_X,
    K     = cbind(rep(J.x.degree, NCOL(X.c)), rep(J.x.segments.c, NCOL(X.c))),
    knots = knots, basis = basis,
    x.min = NULL, x.max = NULL
  )
  mu0_int_X.c <- as.numeric(Psi_int_X.c %*% beta.c)
  
  # --- treated arm spline fit
  Psi_X.t <- prodspline(
    x     = X.t,
    K     = cbind(rep(J.x.degree, NCOL(X.t)), rep(J.x.segments.t, NCOL(X.t))),
    knots = knots, basis = basis,
    x.min = NULL, x.max = NULL
  )
  QRt <- qr(Psi_X.t, LAPACK = TRUE, tol = 1e-10)
  beta.t <- qr.coef(QRt, Y.t); beta.t[is.na(beta.t)] <- 0
  d_sieve.t <- ncol(Psi_X.t)
  
  Psi_int_X.t <- prodspline(
    x     = X.t, xeval = int_X,
    K     = cbind(rep(J.x.degree, NCOL(X.t)), rep(J.x.segments.t, NCOL(X.t))),
    knots = knots, basis = basis,
    x.min = NULL, x.max = NULL
  )
  mu0_int_X.t <- as.numeric(Psi_int_X.t %*% beta.t)
  
  # second stage
  W_i <- mean(pmax(mu0_int_X.t - mu0_int_X.c, 0))
  
  # Asymptotic variance
  # I(-eps < h(x) < eps)
  int_h_predict <- Psi_int_X.t %*% beta.t - Psi_int_X.c %*% beta.c
  ind_good      <- (int_h_predict >= 0)
  
  # Pathwise derivative
  term_1 <- matrix(rep(ind_good, d_sieve.t + d_sieve.c), ncol = d_sieve.t + d_sieve.c)
  term_3 <- cbind(Psi_int_X.t, -Psi_int_X.c)
  Bun    <- colMeans(term_1 * term_3)
  
  # OLS formula
  nt <- nrow(Psi_X.t); pt <- ncol(Psi_X.t)
  nc <- nrow(Psi_X.c); pc <- ncol(Psi_X.c)
  B  <- matrix(0, nrow = nt + nc, ncol = pt + pc)
  B[1:nt, 1:pt] <- Psi_X.t
  B[(nt+1):(nt+nc), (pt+1):(pt+pc)] <- Psi_X.c
  B      <- as.matrix(B)
  BBinvB <- ginv(t(B) %*% B) %*%  t(B)
  term_u <- c(Y.t, Y.c) - B %*% c(beta.t, beta.c)
  Patty  <- t(t(BBinvB) * as.numeric(term_u))%*%(t(BBinvB) * as.numeric(term_u))
  

  # helper: (X'X + λI)^{-1} X'u via sparse Cholesky
  # solve_ridge <- function(X, u, lambda = 1e-6) {
  #   X  <- as(X, "dgCMatrix")
  #   XtX <- crossprod(X) + lambda * Diagonal(ncol(X))
  #   cf  <- Cholesky(XtX, LDL = FALSE)
  #   solve(cf, crossprod(X, u))
  # }
  # 
  # nt <- nrow(Psi_X.t); pt <- ncol(Psi_X.t)
  # nc <- nrow(Psi_X.c); pc <- ncol(Psi_X.c)
  # B  <- matrix(0, nrow = nt + nc, ncol = pt + pc)
  # B[1:nt, 1:pt] <- Psi_X.t
  # B[(nt+1):(nt+nc), (pt+1):(pt+pc)] <- Psi_X.c
  # 
  # # residuals
  # u_t <- as.numeric(Y.t - Psi_X.t %*% beta.t)
  # u_c <- as.numeric(Y.c - Psi_X.c %*% beta.c)
  # 
  # term_u <- c(u_t, u_c)
  # 
  # Lt <- solve_ridge(B, term_u, lambda = 1e-6)
  # L  <- t(Lt)
  # Patty1 <- Lt %*% L
  # 
  # 
  # beta <- ginv(t(B) %*% B) %*%  t(B) %*% c(Y.t, Y.c)
  # beta.tt <- beta[1:256,]
  # beta.cc <- beta[257:512,]
  # 
  # BBinvB <- qr.Q(QRB) %*% t(qr.Q(QRB))
  # B_sparse <- as(A_dense, "sparseMatrix")                  # convert to sparse
  
  # asy.var estimate
  asy.var  <- t(Bun) %*% Patty %*% Bun
  se_i     <- sqrt(asy.var)
  
  return(c(W_i,se_i))
}
  
results <- future_sapply(
  seq_len(n_rep),
  one_ite,
  data = data,
  J.x.degree = J.x.degree,
  J.x.segments.c = J.x.segments.c,
  J.x.segments.t = J.x.segments.t,
  knots = knots,
  basis = basis,
  future.seed = TRUE
)

L <- results[1,] - 1.96*results[2,]
R <- results[1,] + 1.96*results[2,]

final_df <- data.frame()
final_df <- rbind(final_df, data.frame(
  n         = n_obs,
  W_true    = W_true,
  bias      = mean(results[1,] - W_true), 
  sd        = sd(results[1,]),
  se        = mean(results[2,]),
  sd_se     = sd(results[2,]),
  coverage  = mean((W_true < R)*(W_true > L))
))

print(final_df)
