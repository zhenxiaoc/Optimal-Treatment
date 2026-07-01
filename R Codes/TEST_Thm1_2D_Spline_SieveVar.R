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
specs <- list(
  list(
    name = "Model 4",
    rF0  = function(n) data.frame(X1 = runif(n, -0.2, 1.2), X2 = runif(n, -0.2, 1.2)),
    rF   = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),                                # Measure F of target pop.
    lambda = function(x1, x2) 1.4^2 * (x1 > 0) * (x1 < 1) * (x2 > 0) * (x2 < 1),
    p0   = function(x1,x2) plogis(x1 - x2),                                                                 # Propensity score fun.
    mu0  = function(x1,x2,d)  (1 - x1^2 -x2^2) * (4 + sin(x1)*x2 + cos(x2))+ d * (x1*0.5 - x2*0.4),         # Regression fun.
    noise_sd = 1,
    J.x.segments.c = 1,
    J.x.segments.t = 1,
    J.x.segments   = 1
  ),
  list(
    name = "Model 5",
    rF0 = function(n) data.frame(X1 = runif(n, -0.2, 1.2), X2 = runif(n, -0.2, 1.2)),
    rF  = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),
    lambda = function(x1, x2) 1.4^2 * (x1 > 0) * (x1 < 1) * (x2 > 0) * (x2 < 1),
    p0  = function(x1, x2) plogis(x1 - x2),
    mu0 = function(x1, x2, d) (1 - x1 * x2) * (3 + sin(pi * x1) * cos(pi * x2)) + d * (0.3 * x1 - 0.3 * x2),
    noise_sd = 1,
    J.x.segments.c = 1,
    J.x.segments.t = 1,
    J.x.segments   = 1
  ),
  list(
    name = "Model 6",
    rF0 = function(n) data.frame(X1 = runif(n, -0.2, 1.2), X2 = runif(n, -0.2, 1.2)),
    rF  = function(m) data.frame(X1 = rbeta(m, 1, 1), X2 = rbeta(m, 1, 1)),
    lambda = function(x1, x2) 1.4^2 * (x1 > 0) * (x1 < 1) * (x2 > 0) * (x2 < 1),
    p0  = function(x1, x2) plogis(1.5 * x1 - 0.5 * x2),
    mu0 = function(x1, x2, d) log(1 + x1 + x2) + d * (x1 - 0.7 * x2),
    noise_sd = 1,
    J.x.segments.c = 4,
    J.x.segments.t = 1,
    J.x.segments   = 1
  ),
  list(
    name = "Model 7",
    rF0 = function(n) data.frame(X1 = runif(n, -0.2, 1.2), X2 = runif(n, -0.2, 1.2)),
    rF  = function(m) data.frame(X1 = rbeta(m, 1, 1), X2 = rbeta(m, 1, 1)),
    lambda = function(x1, x2) 1.4^2 * (x1 > 0) * (x1 < 1) * (x2 > 0) * (x2 < 1),
    p0  = function(x1, x2) plogis(-0.5 + x1 + 2 * x2),
    mu0 = function(x1, x2, d) (x1^2 + x2^2) * exp(-x1 - x2) + d * (0.5 - x2),
    noise_sd = 1,
    J.x.segments.c = 4,
    J.x.segments.t = 1,
    J.x.segments   = 1
  )
)

#################### Simulation #################### 
ite <- 2000                 
ns  <- c(1500,3000,6000) 
M   <- 5000

# Tuning
J.x.degree     <- 3              # Cubic splines
knots          <- "uniform"
basis          <- "tensor" 
X.min          <- NULL
X.max          <- NULL

#### Paralleled computing
plan(multisession, workers = parallel::detectCores() - 1)
set.seed(2025)

#################### Storage ####################
final_df <- data.frame()
draws_df <- data.frame()

#################### Loop over specs ####################
for (spec in specs) {
  # Compute W_true
  int_lower = rep(0, 2)
  int_upper = rep(1, 2)
  int_X = matrix(NA, nrow = M, ncol = 2)
  sobol_points <- sobol(M, d = 2, randomize = "none")
  for (j in 1:2) {
    int_X[,j] <- int_lower[j] +  sobol_points[,j] * (int_upper[j] - int_lower[j])
  }
  int_X <- data.frame(X1 = int_X[,1], X2 = int_X[,2])
  W_true <- mean(pmax(spec$mu0(int_X$X1, int_X$X2, 1) - spec$mu0(int_X$X1, int_X$X2, 0), 0))
  
  for (n in ns){
    cat("Running spec =", spec$name, "| n =", n, "\n")
    
    # Define one iteration
    one_ite <- function(dummy) {
      df <- spec$rF0(n)
      df$D <- rbinom(n, 1, spec$p0(df$X1, df$X2))
      noise <- rnorm(nrow(df), 0, spec$noise_sd)
      df$Y <- spec$mu0(df$X1, df$X2, df$D) + noise
      
      Y <- df$Y
      X <- cbind(df$X1, df$X2)
      D <- df$D
      
      # Control
      df.c <- subset(df, D == 0)
      X.c  <- cbind(df.c$X1, df.c$X2)
      Y.c <-  as.numeric(df.c$Y) 
      
      # Treated
      df.t <- subset(df, D == 1)
      X.t  <- cbind(df.t$X1, df.t$X2)
      Y.t <- as.numeric(df.t$Y) 
      
      df_trim <- subset(df, X1 > min(df.t$X1) & X1 < max(df.t$X1) & 
                          X1 > min(df.c$X1) & X1 < max(df.c$X1) & 
                          X2 > min(df.t$X2) & X2 < max(df.t$X2) &
                          X2 > min(df.c$X2) & X2 < max(df.c$X2))
      X_trim <- cbind(df_trim$X1, df_trim$X2)
      Y_trim <- df_trim$Y
      D_trim <- df_trim$D
      
      
      # chooseJ_result <- npiv_choose_J(Y.c,
      #                                 X.c,
      #                                 X.c,
      #                                 X.grid = NULL,
      #                                 J.x.degree = 3,
      #                                 K.w.degree = 4,
      #                                 K.w.smooth = 2,
      #                                 knots = "uniform",
      #                                 basis = "tensor",
      #                                 X.min = NULL,
      #                                 X.max = NULL,
      #                                 W.min = NULL,
      #                                 W.max = NULL,
      #                                 grid.num = 50,
      #                                 boot.num = 99,
      #                                 check.is.fullrank= FALSE,
      #                                 progress = TRUE)
      # 
      # chooseJ_result$J.x.seg
      # 
      # chooseJ_result <- npiv_choose_J(Y.t,
      #                                 X.t,
      #                                 X.t,
      #                                 X.grid = NULL,
      #                                 J.x.degree = 3,
      #                                 K.w.degree = 4,
      #                                 K.w.smooth = 2,
      #                                 knots = "uniform",
      #                                 basis = "tensor",
      #                                 X.min = NULL,
      #                                 X.max = NULL,
      #                                 W.min = NULL,
      #                                 W.max = NULL,
      #                                 grid.num = 50,
      #                                 boot.num = 99,
      #                                 check.is.fullrank= FALSE,
      #                                 progress = TRUE)
      # 
      # chooseJ_result$J.x.seg
      # 
      # chooseJ_result <- npiv_choose_J(D,
      #                                 X,
      #                                 X,
      #                                 X.grid = NULL,
      #                                 J.x.degree = 3,
      #                                 K.w.degree = 4,
      #                                 K.w.smooth = 2,
      #                                 knots = "uniform",
      #                                 basis = "tensor",
      #                                 X.min = NULL,
      #                                 X.max = NULL,
      #                                 W.min = NULL,
      #                                 W.max = NULL,
      #                                 grid.num = 50,
      #                                 boot.num = 99,
      #                                 check.is.fullrank= FALSE,
      #                                 progress = TRUE)
      # 
      # chooseJ_result$J.x.seg
      
      J.x.segments.c <- spec$J.x.segments.c                # Sieve dimension chosen by npiv_choose_J
      J.x.segments.t <- spec$J.x.segments.t  
      J.x.segments   <- spec$J.x.segments   
      
      # First stage
      # Estimating mu_0(x , 0) and evaluated it on X_trim
      Psi_X.c      <- prodspline(x = X.c,
                                 K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                                 knots = knots,
                                 basis = basis,
                                 x.min = X.min,
                                 x.max = X.max)
      beta.c       <- ginv(t(Psi_X.c) %*% Psi_X.c) %*% t(Psi_X.c) %*% Y.c
      d_sieve.c    <- ncol(Psi_X.c)
      
      Psi_int_X.c <- prodspline(x = X.c,
                                xeval = int_X,
                                K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                                knots = knots,
                                basis = basis,
                                x.min = X.min,
                                x.max = X.max)
      mu0_int_X.c <- Psi_int_X.c %*% beta.c
      
      # Estimating mu_0(x , 1) and evaluated it on X_trim
      Psi_X.t      <- prodspline(x = X.t,
                                 K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                                 knots = knots,
                                 basis = basis,
                                 x.min = X.min,
                                 x.max = X.max)
      beta.t       <- ginv(t(Psi_X.t) %*% Psi_X.t) %*% t(Psi_X.t) %*% Y.t
      d_sieve.t    <- ncol(Psi_X.t)
      
      Psi_int_X.t <- prodspline(x = X.t,
                                xeval = int_X,
                                K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                                knots = knots,
                                basis = basis,
                                x.min = X.min,
                                x.max = X.max)
      mu0_int_X.t <- Psi_int_X.t %*% beta.t
      
      # Second stage
      W_hat_i     <- mean(pmax(mu0_int_X.t- mu0_int_X.c, 0))
      
      # Asymptotic variance
      M2        <- 40000
      int_lower <- rep(0, 2)
      int_upper <- rep(1, 2)
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
      ind_good      <- (int_h_predict >= 0)
      
      
      # Pathwise derivative
      term_1 <- matrix(rep(ind_good, d_sieve.t + d_sieve.c), ncol = d_sieve.t + d_sieve.c)
      term_3 <- cbind(Psi_int_X.t, -Psi_int_X.c)
      Bun    <- colMeans(term_1 * term_3)
      
      # OLS formula
      nt <- nrow(Psi_X.t); pt <- ncol(Psi_X.t)
      nc <- nrow(Psi_X.c); pc <- ncol(Psi_X.c)
      B <- matrix(0, nrow = nt + nc, ncol = pt + pc)
      B[1:nt, 1:pt] <- Psi_X.t
      B[(nt+1):(nt+nc), (pt+1):(pt+pc)] <- Psi_X.c
      B      <- as.matrix(B) 
      BBinvB <- ginv(t(B) %*% B) %*%  t(B)
      term_u <- c(Y.t, Y.c) - B %*% rbind(beta.t, beta.c)
      Patty  <- t(t(BBinvB) * as.numeric(term_u))%*%(t(BBinvB) * as.numeric(term_u))
      
      # asy.var estimate
      asy.var  <- t(Bun) %*% Patty %*% Bun
      se_i     <- sqrt(asy.var)
      
      return(c(W_hat = W_hat_i, se = se_i))
    }
    
    # Run in parallel
    res_mat <- future_replicate(ite, one_ite(1), simplify = "matrix", future.seed = TRUE)
    
    # Extract
    W_hat  <- res_mat["W_hat",]
    se  <- res_mat["se",]
    
    # Store summary row
    final_df <- rbind(final_df, data.frame(
      spec      = spec$name,
      n         = n,
      W_true    = W_true,
      bias      = mean(W_hat - W_true), 
      sd        = sd(W_hat),
      se        = mean(se),
      sd_se     = sd(se),
      coverage  = mean((W_hat - 1.96*se <= W_true)*(W_hat + 1.96*se >= W_true))
    ))
  }
}

print(final_df)

save(final_df, res_mat, file = "sim_results_Thm1_2D_SieveVar.Rdata")
