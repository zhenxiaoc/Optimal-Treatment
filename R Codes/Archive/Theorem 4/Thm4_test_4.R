rm(list = ls())   
source("spline.R")
library(npiv)
library(mgcv)
library(future.apply)
library(future)
library(qrng)
library(MASS)
library(Matrix)


####################  Model specification #################### 
spec <-   list(
  name = "Model 1",
  rF0  = function(n) data.frame(X1 = runif(n, -2, 2), X2 = runif(n, -2, 2)),
  rF   = function(m) data.frame(X1 = runif(m, -1.9, 1.9), X2 = runif(m, -1.9, 1.9)), 
  p0   = function(x1,x2) plogis(x1 - x2),                            
  mu0  = function(x1,x2,d)  d *(1 - x1^2 -x2^2) * (4 + sin(x1)*x2 + cos(x2)),        
  noise_sd = 2                                                  
)

#################### Tuning ####################
ns             <- c(1500, 3000, 6000)        # sample size
ite            <- 1000        # number of iterations
M              <- 5000        # number of sobol points to estimate W_0
alpha          <- 0.05
basis          <- "tensor"    # c("tensor","additive","glp")
J.x.degree     <- 3           # cubic spline
J.x.segments.t <- 5           # number of segments to estimate mu_0(x,1)
J.x.segments.c <- 4           # number of segments to estimate mu_0(x,0)
knots          <- "quantiles" # c("uniform","quantiles")
X.min          <- NULL
X.max          <- NULL

m              <- 6
eps            <- 0.005       # level set approximation


#################### Parallel setup ################
plan(multisession, workers = 2)
#plan(multicore, workers = parallel::detectCores() - 1)
#plan(multicore, workers = 4)
set.seed(2025)

#################### Storage #######################
final_df <- data.frame()
draws_df <- data.frame()

#################### Loop over specs ###############
for (n in ns) {
  cat("n =", n, "\n")
  
  # Define one iteration
  one_ite <- function(dummy) {
    # DGP
    df   <- spec$rF0(n)
    df$D <- rbinom(n, 1, spec$p0(df$X1, df$X2))
    df$Y <- spec$mu0(df$X1, df$X2, df$D) + rnorm(nrow(df), 0, spec$noise_sd)
    
    # Control
    df.c <- subset(df, D == 0)
    X.c  <- cbind(df.c$X1, df.c$X2)
    Y.c  <-  as.numeric(df.c$Y) 
    
    # Treated
    df.t <- subset(df, D == 1)
    X.t  <- cbind(df.t$X1, df.t$X2)
    Y.t  <- as.numeric(df.t$Y) 
    
    # sobol points to estimate W_0
    int_lower <- rep(-1.9, 2)
    int_upper <- rep(1.9, 2)
    int_X     <- matrix(NA, nrow = M, ncol = 2)
    
    sobol_points <- sobol(M, d = 2, randomize = "none")
    for (j in 1:2) {
      int_X[,j] <- int_lower[j] +  sobol_points[,j] * (int_upper[j] - int_lower[j])
    }
    
    # First stage
    Psi_X.c <- prodspline(x = X.c,
                          K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                          knots = knots,
                          basis = basis,
                          x.min = X.min,
                          x.max = X.max)
    d_sieve.c <- ncol(Psi_X.c)
    
    Psi_int_X.c <- prodspline(x = X.c,
                              xeval = int_X,
                              K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                              knots = knots,
                              basis = basis,
                              x.min = X.min,
                              x.max = X.max)
    
    XXinvX.c <- ginv(t(Psi_X.c) %*% Psi_X.c) %*%  t(Psi_X.c)
    beta.c <- XXinvX.c %*% Y.c
    
    Psi_X.t <- prodspline(x = X.t,
                          K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                          knots = knots,
                          basis = basis,
                          x.min = X.min,
                          x.max = X.max)
    d_sieve.t <- ncol(Psi_X.t)
    
    Psi_int_X.t <- prodspline(x = X.t,
                              xeval = int_X,
                              K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                              knots = knots,
                              basis = basis,
                              x.min = X.min,
                              x.max = X.max)
    
    XXinvX.t  <- ginv(t(Psi_X.t) %*% Psi_X.t) %*%  t(Psi_X.t)
    beta.t    <- XXinvX.t %*% Y.t
    
    # Second stage
    mu0_int_X.c <- Psi_int_X.c %*% beta.c
    mu0_int_X.t <- Psi_int_X.t %*% beta.t    
    W_hat_i     <- 3.8^2 * mean((mu0_int_X.t - mu0_int_X.c >= 0)) # W_0 estimate
    
    # estimate asymptotic variance
    M2        <- 10^m
    int_lower <- rep(-1.9, 2)
    int_upper <- rep(1.9, 2)
    int_X     <- matrix(NA, nrow = M2, ncol = 2)
    
    sobol_points <- sobol(M2, d = 2, randomize = "none")
    for (j in 1:2) {
      int_X[,j] <- int_lower[j] +  sobol_points[,j] * (int_upper[j] - int_lower[j])
    }
    
    # psi_k(x) for controls
    Psi_int_X.c <- prodspline(x = X.c,
                              xeval = int_X,
                              K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                              knots = knots,
                              basis = basis,
                              x.min = X.min,
                              x.max = X.max)
    
    # psi_k(x) for treated
    Psi_int_X.t <- prodspline(x = X.t,
                              xeval = int_X,
                              K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                              knots = knots,
                              basis = basis,
                              x.min = X.min,
                              x.max = X.max)  
    
    # I(-eps < h(x) < eps)
    int_h_predict <- Psi_int_X.t %*% beta.t - Psi_int_X.c %*% beta.c
    ind_good      <- (int_h_predict > -eps) & (int_h_predict < eps)
    num_i         <- sum(ind_good)
    
    # compute (1/2*eps) * \int I(-eps < h(x) < eps) * p_0(x) * psi_k(x) * f(x) dx, and
    # compute (1/2*eps) * \int I(-eps < h(x) < eps) * (p_0(x) - 1)* psi_k(x) * f(x) dx
    # term_1: I(-eps < h(x) < eps)
    # term_2: p_0(x) or p_0(x) - 1
    # term_3: psi_k(x)
    term_1 <- matrix(rep(ind_good, d_sieve.t + d_sieve.c), ncol = d_sieve.t + d_sieve.c)
    term_3 <- cbind(Psi_int_X.t, -Psi_int_X.c)
    Bun    <- 3.8^2 * colMeans(term_1 * term_3)/(2*eps)
    # Note: I follow the formula in Sim2_Disk.R to compute Bun
    
    # OLS formula
    B      <- bdiag(Psi_X.t, Psi_X.c)
    B      <- as.matrix(B) 
    BBinvB <- ginv(t(B) %*% B) %*%  t(B)
    term_u <- c(Y.t, Y.c) - B %*% rbind(beta.t, beta.c)
    Patty  <- t(t(BBinvB) * as.numeric(term_u))%*%(t(BBinvB) * as.numeric(term_u))
    
    # asy.var estimate
    asy.var  <- t(Bun) %*% Patty %*% Bun
    sd_hat_i <- sqrt(asy.var)
    
    # Clean up
    rm(Psi_X.t, Psi_X.c, XXinvX.t, XXinvX.c, Psi_int_X.t, Psi_int_X.c, int_X, sobol_points, int_h_predict, ind_good, term_1, term_3, B)
    gc()
    
    return(c(W_hat = W_hat_i, sd_hat = sd_hat_i, num = num_i))
  }
  
  
  # Run in parallel
  res_mat <- future_replicate(ite, one_ite(1), simplify = "matrix", future.seed = TRUE)
  
  # Extract
  W_hat  <- res_mat["W_hat", ]
  sd_hat <-res_mat["sd_hat", ]
  
  # Store summary row
  final_df <- rbind(final_df, data.frame(
    spec      = spec$name,
    n         = n,
    W_true    = pi,
    bias      = mean(W_hat - pi),
    sd        = sd(W_hat),
    se        = mean(sd_hat),
    coverage  = mean(abs(W_hat - pi)/sd_hat <= 1.96)
  ))
  
  # Store all draws
  draws_df <- rbind(draws_df, data.frame(
    spec      = spec$name,
    n         = n,
    W_hat     = W_hat
  ))
  
  cat("spec =", spec$name, " n =", n,
      " | bias =", round(mean(W_hat - pi), 3),
      " | sd =", round(sd(W_hat), 3),
      " | se =", round(mean(sd_hat), 3),
      " | coverage =", round(mean(abs(W_hat - pi)/sd_hat <= 1.96), 3),"\n")
}

print(final_df)
