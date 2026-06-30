rm(list = ls())        # Remove all objects from the workspace

source("spline.R")
library(npiv)
library(mgcv)
library(future.apply)
library(qrng)
library(MASS)

####################  Model specification #################### 
specs <- list(
  list(
    name = "Model 4",
    rF0  = function(n) data.frame(X1 = runif(n, -0.1, 1.1), X2 = runif(n, -0.1, 1.1)),
    rF   = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),                                # Measure F of target pop.
    fF   = function(x1,x2) dbeta(as.numeric(x1), 1, 1) * dbeta(as.numeric(x2), 1, 1) * 1.2^2,               # Density f of target pop.
    p0   = function(x1,x2) plogis(x1 - x2),                                                                 # Propensity score fun.
    mu0  = function(x1,x2,d)  (1 - x1^2 -x2^2) * (4 + sin(x1)*x2 + cos(x2))+ d * (x1*0.5 - x2*0.4),         # Regression fun.
    noise_sd = 1                                                                                            # Noise level in regression
  ),
  list(
    name = "Model 5",
    rF0 = function(n) data.frame(X1 = runif(n, -0.1, 1.1), X2 = runif(n, -0.1, 1.1)),
    rF  = function(m) data.frame(X1 = rbeta(m, 1, 1), X2 = rbeta(m, 1, 1)),
    fF  = function(x1, x2) dbeta(as.numeric(x1), 1, 1) * dbeta(as.numeric(x2), 1, 1) * 1.2^2,
    p0  = function(x1, x2) plogis(x1 - x2),
    mu0 = function(x1, x2, d) (1 - x1 * x2) * (3 + sin(pi * x1) * cos(pi * x2)) + d * (0.3 * x1 - 0.3 * x2),
    noise_sd = 1
  ),
  list(
    name = "Model 6",
    rF0 = function(n) data.frame(X1 = runif(n, -0.1, 1.1), X2 = runif(n, -0.1, 1.1)),
    rF  = function(m) data.frame(X1 = rbeta(m, 1, 1), X2 = rbeta(m, 1, 1)),
    fF  = function(x1, x2) dbeta(as.numeric(x1), 1, 1) * dbeta(as.numeric(x2), 1, 1) * 1.2^2,
    p0  = function(x1, x2) plogis(1.5 * x1 - 0.5 * x2),
    mu0 = function(x1, x2, d) log(1 + x1 + x2) + d * (x1 - 0.7 * x2),
    noise_sd = 1
  ),
  list(
    name = "Model 7",
    rF0 = function(n) data.frame(X1 = runif(n, -0.1, 1.1), X2 = runif(n, -0.1, 1.1)),
    rF  = function(m) data.frame(X1 = rbeta(m, 1, 1), X2 = rbeta(m, 1, 1)),
    fF  = function(x1, x2) dbeta(as.numeric(x1), 1, 1) * dbeta(as.numeric(x2), 1, 1) * 1.2^2,
    p0  = function(x1, x2) plogis(-0.5 + x1 + 2 * x2),
    mu0 = function(x1, x2, d) (x1^2 + x2^2) * exp(-x1 - x2) + d * (0.5 - x2),
    noise_sd = 1
  )
)

#################### Parameters #################### 
ite <- 1000                      # number of iterations
ns <- c(1500, 3000, 6000)        # sample sizes
alpha          <- 0.05
basis          <- "tensor"       # c("tensor","additive","glp"),
J.x.degree     <- 3              # cubic spline
J.x.segments.t <- 1              # number of segments for treated
J.x.segments.c <- 1              # number of segments for control
knots          <- "quantiles"    # c("uniform","quantiles")
X.min          <- NULL
X.max          <- NULL


#################### Parallel setup ################
plan(multisession, workers = parallel::detectCores() - 1)
set.seed(2025)

#################### Storage #######################
final_df <- data.frame()
draws_df <- data.frame()

#################### Loop over specs ###############
for (spec in specs) {
  # Numerically approximate W̄(μ₀)
  int_X  <- spec$rF(10000)
  W_true <- mean(pmax(spec$mu0(int_X[,1], int_X[,2], 1) - spec$mu0(int_X[,1], int_X[,2], 0),0))
  
  for (n in ns) {
    cat("Running spec =", spec$name, "| n =", n, "\n")
    
    one_ite <- function(dummy){
      # DGP
      df <- spec$rF0(n)
      df$D <- rbinom(n, 1, spec$p0(df$X1, df$X2))
      df$Y <- spec$mu0(df$X1, df$X2, df$D) + rnorm(nrow(df), 0, spec$noise_sd)
      
      df_trim <- subset(df, X1 >= 0 & X1 <= 1 & X2 >= 0 & X2 <= 1)
      X_trim <- cbind(df_trim$X1, df_trim$X2)
      
      # Control
      df.c <- subset(df, D == 0)
      X.c  <- cbind(df.c$X1, df.c$X2)
      Y.c  <-  as.numeric(df.c$Y) 

      # Treated 
      df.t <- subset(df, D == 1)
      X.t  <- cbind(df.t$X1, df.t$X2)
      Y.t  <- as.numeric(df.t$Y) 
      
      # First stage
      Psi_X.c <- prodspline(x = X.c,
                            K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                            knots = knots,
                            basis = basis,
                            x.min = X.min,
                            x.max = X.max)
      
      Psi_int_X.c <- prodspline(x = X.c,
                                xeval = int_X,
                                K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                                knots = knots,
                                basis = basis,
                                x.min = X.min,
                                x.max = X.max)
      
      XXinvX.c <- ginv(t(Psi_X.c) %*% Psi_X.c) %*%  t(Psi_X.c)
      beta.c <- XXinvX.c %*% Y.c
      mu0_int.c = Psi_int_X.c %*%beta.c  
 
      Psi_X.t <- prodspline(x = X.t,
                            K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                            knots = knots,
                            basis = basis,
                            x.min = X.min,
                            x.max = X.max)
      
      Psi_int_X.t <- prodspline(x = X.t,
                                xeval = int_X,
                                K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                                knots = knots,
                                basis = basis,
                                x.min = X.min,
                                x.max = X.max)
      
      XXinvX.t  <- ginv(t(Psi_X.t) %*% Psi_X.t) %*%  t(Psi_X.t)
      beta.t    <- XXinvX.t %*% Y.t
      mu0_int.t <- Psi_int_X.t %*%beta.t
      
      # Second Stage
      W_hat_i <- mean(pmax(mu0_int.t - mu0_int.c, 0))
      
      
      # Asymptotic variance
      Psi_X_trim.c <- prodspline(x = X.c,
                                 xeval = X_trim,
                                 K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                                 knots = knots,
                                 basis = basis,
                                 x.min = X.min,
                                 x.max = X.max)
      
      Psi_X_trim.t <- prodspline(x = X.t,
                                 xeval = X_trim,
                                 K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                                 knots = knots,
                                 basis = basis,
                                 x.min = X.min,
                                 x.max = X.max)
      
      mu0_hat.c <- Psi_X_trim.c %*% beta.c    # μ̂₀(x₁, x₂, 0) evaluated at X_trim
      mu0_hat.t <- Psi_X_trim.t %*% beta.t    # μ̂₀(x₁, x₂, 1) evaluated at X_trim
      ind       <- as.numeric((mu0_hat.t - mu0_hat.c) >= 0)
      
      resid     <- ifelse(df_trim$D == 1, df_trim$Y - mu0_hat.t, df_trim$Y - mu0_hat.c)
      sig2_mod  <- gam(I(resid^2) ~ te(X1, X2), data = df_trim, family = gaussian(), method = "REML")
      sig2_hat  <- predict(sig2_mod, newdata = df_trim, type = "response")
      lambda2   <- (1.2^2)^2
      ps_mod    <- bam(D ~ te(X1, X2), data = df, family = binomial(), method = "fREML", discrete = TRUE)
      p_hat     <- predict(ps_mod, newdata = df_trim, type = "response")
      asymp_var_hat_i <- sum(ind * lambda2 * sig2_hat / (p_hat * (1 - p_hat)), na.rm = TRUE)/n
      sd_hat_i  <- sqrt(asymp_var_hat_i/n)
      
      return(c(W_hat = W_hat_i, sd_hat = sd_hat_i))
    }
    
    # Run in parallel
    res_mat <- future_replicate(ite, one_ite(1), simplify = "matrix", future.seed = TRUE)
    
    # Extract
    W_hat     <- res_mat["W_hat", ]
    sd_hat    <- res_mat["sd_hat", ]
    
    # Store summary row
    final_df <- rbind(final_df, data.frame(
      spec      = spec$name,
      n         = n,
      W_true    = W_true,
      bias      = mean(W_hat - W_true),       
      sd        = sd(W_hat),
      se        = mean(sd_hat),
      coverage  = mean((W_hat - 1.96*sd_hat <= W_true)*(W_hat + 1.96*sd_hat >= W_true))
    ))
    
    # Store all draws
    draws_df <- rbind(draws_df, data.frame(
      spec      = spec$name,
      n         = n,
      W_hat     = W_hat
    ))
    
    cat("spec ="       , spec$name, 
        " n ="         , n,
        " | bias ="    , round(mean(W_hat - W_true), 3),
        " | sd ="      , round(sd(W_hat), 3),
        " | se ="      , round(mean(sd_hat), 3),
        " | coverage =", round(mean((W_hat - 1.96*sd_hat <= W_true)*(W_hat + 1.96*sd_hat >= W_true)), 3), "\n")
  }
}


#################### Done! ####################
print(final_df)