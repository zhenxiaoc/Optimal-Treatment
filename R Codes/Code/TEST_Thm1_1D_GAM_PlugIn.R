rm(list = ls())        # Remove all objects from the workspace

library(mgcv)
library(future.apply)
library(ggplot2)
library(dplyr)
library(qrng)

####################  Model specification #################### 
specs <- list(
  list(
    name   = "Model 1",
    rF0    = function(n) data.frame(X = runif(n, -0.2, 1.2)),
    rF     = function(m) data.frame(X = runif(n, 0, 1)),
    lambda = function(x) 1.4 * (x > 0) * (x < 1),
    p0     = function(x) plogis(1 - 2*x),
    mu0    = function(x,d) 5*sin(2*pi*x)*cos(2*pi*x) + d*(-0.4 + 2*x^2),
    noise_sd = 1,
    J.x.segments.c = 16,
    J.x.segments.t = 16,
    J.x.segments   = 1
  ),
  list(
    name = "Model 2",
    rF0  = function(n) data.frame(X = runif(n, -0.2, 1.2)),
    rF   = function(m) data.frame(X = runif(n, 0, 1)),
    lambda = function(x) 1.4 * (x > 0) * (x < 1),
    p0   = function(x) plogis(-0.5 + x),
    mu0  = function(x,d) 0.5*abs(x) + d*(0.5 - x^2),
    noise_sd = 1,
    J.x.segments.c = 1,
    J.x.segments.t = 1,
    J.x.segments   = 1
  ),
  list(
    name = "Model 3",
    rF0  = function(n) data.frame(X = runif(n, -0.2, 1.2)),
    rF   = function(m) data.frame(X = runif(n, 0, 1)),
    lambda = function(x) 1.4 * (x > 0) * (x < 1),
    p0   = function(x) plogis(0.5 - x),
    mu0  = function(x,d) x^2 + d*(1 - x),
    noise_sd = 1,
    J.x.segments.c = 1,
    J.x.segments.t = 1,
    J.x.segments   = 1
  )
)


#################### Parameters ####################
ite          <- 2000
ns           <- c(1500, 3000, 6000)
M            <- 5000 # number of sobol points for simulating W_0

#################### Parallel setup ################
plan(multisession, workers = parallel::detectCores() - 1)
set.seed(2025)

#################### Storage #######################
final_df <- data.frame()
draws_df <- data.frame()

#################### Loop over specs ###############
for (spec in specs) {
  # Compute W_true
  int_lower = rep(0, 1)
  int_upper = rep(1, 1)
  sobol_points <- sobol(M, d = 1, randomize = "none")
  int_X <- int_lower +  sobol_points * (int_upper - int_lower)
  int_X <- data.frame(X = int_X)
  W_true <- mean(pmax(spec$mu0(int_X$X, 1) - spec$mu0(int_X$X, 0), 0))
  
  for (n in ns) {
    cat("Running spec =", spec$name, "| n =", n, "\n")
    
    # Define one iteration
    one_ite <- function(dummy) {
      # DGP
      df    <- spec$rF0(n)
      df$D  <- rbinom(n, 1, spec$p0(df$X))
      noise <- rnorm(nrow(df), 0, spec$noise_sd)
      df$Y  <- spec$mu0(df$X, df$D) + noise
      
      # Control      
      df.c <- subset(df, D == 0)
      X.c  <- df.c$X
      Y.c  <-  as.numeric(df.c$Y) 
      
      # Treated     
      df.t <- subset(df, D == 1)
      X.t  <- df.t$X
      Y.t  <- as.numeric(df.t$Y) 
      
      # First stage
      ps_mod   <- gam(D ~ s(X, bs = "bs"), data=df, family=binomial(), method="REML")
      ctrl_mod <- gam(Y ~ s(X, bs = "bs"), data=df.c, family=gaussian(), method="REML")
      trt_mod  <- gam(Y ~ s(X, bs = "bs"), data=df.t, family=gaussian(), method="REML")
      
      # Second stage
      mu0_int_X.t <- predict(trt_mod, newdata = int_X, type = "response")
      mu0_int_X.c <- predict(ctrl_mod, newdata = int_X, type = "response")
      W_hat_i     <- mean(pmax(mu0_int_X.t - mu0_int_X.c, 0))
      
      # Asymptotic variance
      p_hat    <- predict(ps_mod, newdata = df, type = "response")
      mu0_X.t  <- predict(trt_mod, newdata = df, type = "response")
      mu0_X.c  <- predict(ctrl_mod, newdata = df, type = "response")
      
      ind      <- as.numeric((mu0_X.t - mu0_X.c) >= 0)
      resid    <- ifelse(df$D == 1, df$Y - mu0_X.t, df$Y - mu0_X.c)
      lambda2  <- spec$lambda(df$X)^2
      asymp_var_hat_i  <- sum(ind * lambda2 * resid^2 / (p_hat * (1 - p_hat)), na.rm = TRUE)/n
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

#################### Done! ####################
print(final_df)

save(final_df, res_mat, file = "sim_results_Thm1_1D_GAM.Rdata")