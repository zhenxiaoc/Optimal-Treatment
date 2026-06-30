rm(list = ls())        # Remove all objects from the workspace

library(mgcv)
library(future.apply)
library(ggplot2)
library(dplyr)
library(qrng)
source("Spline.R")

####################  Model specification #################### 
specs <- list(
  list(
    name = "Model 4",
    rF0  = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),
    rF   = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),                                # Measure F of target pop.
    lambda = function(x1, x2) 1,
    p0   = function(x1,x2) plogis(x1 - x2),                                                                 # Propensity score fun.
    mu0  = function(x1,x2,d)  (1 - x1^2 -x2^2) * (4 + sin(x1)*x2 + cos(x2))+ d * (x1*0.5 - x2*0.4),         # Regression fun.
    noise_sd = 1,
    J.x.segments.c = 1,
    J.x.segments.t = 1,
    J.x.segments   = 1
  ),
  list(
    name = "Model 5",
    rF0 = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),
    rF  = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),
    lambda = function(x1, x2) 1,
    p0  = function(x1, x2) plogis(x1 - x2),
    mu0 = function(x1, x2, d) (1 - x1 * x2) * (3 + sin(pi * x1) * cos(pi * x2)) + d * (0.3 * x1 - 0.3 * x2),
    noise_sd = 1,
    J.x.segments.c = 1,
    J.x.segments.t = 1,
    J.x.segments   = 1
  ),
  list(
    name = "Model 6",
    rF0 = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),
    rF  = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),
    lambda = function(x1, x2) 1,
    p0  = function(x1, x2) plogis(1.5 * x1 - 0.5 * x2),
    mu0 = function(x1, x2, d) log(1 + x1 + x2) + d * (x1 - 0.7 * x2),
    noise_sd = 1,
    J.x.segments.c = 1,
    J.x.segments.t = 1,
    J.x.segments   = 1
  ),
  list(
    name = "Model 7",
    rF0 = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),
    rF  = function(m) data.frame(X1 = runif(m, 0, 1), X2 = runif(m, 0, 1)),
    lambda = function(x1, x2) 1,
    p0  = function(x1, x2) plogis(-0.5 + x1 + 2 * x2),
    mu0 = function(x1, x2, d) (x1^2 + x2^2) * exp(-x1 - x2) + d * (0.5 - x2),
    noise_sd = 1,
    J.x.segments.c = 1,
    J.x.segments.t = 1,
    J.x.segments   = 1
  )
)

#################### Simulation #################### 
ite <- 2000                 
ns  <- c(1500, 3000, 6000) 
M   <- 5000

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
      # DGP
      df    <- spec$rF0(n)
      df$D  <- rbinom(n, 1, spec$p0(df$X1, df$X2))
      noise <- rnorm(nrow(df), 0, spec$noise_sd)
      df$Y  <- spec$mu0(df$X1, df$X2, df$D) + noise
      
      # Control      
      df.c <- subset(df, D == 0)
      X.c  <- cbind(df.c$X1, df.c$X2)
      Y.c  <-  as.numeric(df.c$Y) 
      
      # Treated     
      df.t <- subset(df, D == 1)
      X.t  <- cbind(df.t$X1, df.t$X2)
      Y.t  <- as.numeric(df.t$Y) 
      
      # First stage
      ps_mod   <- gam(D ~ te(X1, X2), data = df, family = binomial(), method = "REML")
      ctrl_mod <- gam(Y ~ te(X1, X2), data = df.c, family = gaussian(), method = "REML")
      trt_mod  <- gam(Y ~ te(X1, X2), data = df.t, family = gaussian(), method = "REML")
      
      # Second stage
      mu0_X.t  <- predict(trt_mod, newdata = df, type = "response")
      mu0_X.c  <- predict(ctrl_mod, newdata = df, type = "response")
      W_hat_i     <- mean(pmax(mu0_X.t - mu0_X.c, 0))
      
      # Asymptotic variance
      p_hat    <- predict(ps_mod, newdata = df, type = "response")
      
      ind      <- as.numeric((mu0_X.t - mu0_X.c) >= 0)
      resid    <- ifelse(df$D == 1, df$Y - mu0_X.t, df$Y - mu0_X.c)
      lambda2  <- spec$lambda(df$X1, df$x2)^2
      asymp_var_hat_i  <- mean(pmax(mu0_X.t - mu0_X.c, 0)^2) - mean(pmax(mu0_X.t - mu0_X.c, 0))^2 + sum(ind * lambda2 * resid^2 / (p_hat * (1 - p_hat)), na.rm = TRUE)/n
      se_i     <- sqrt(asymp_var_hat_i/n)
      
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

save(final_df, res_mat, file = "sim_results_Thm2_2D_GAM.Rdata")
