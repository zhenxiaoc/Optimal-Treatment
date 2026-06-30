rm(list = ls())

library(mgcv)
library(future.apply)
library(ggplot2)
library(dplyr)
library(qrng)

####################  Model specification ####################
specs <- list(
  list(
    name = "Model 1",
    rF0  = function(n) data.frame(X = runif(n, 0, 1)),
    rF   = function(m) data.frame(X = runif(n, 0, 1)),
    lambda = function(x) 1,
    p0   = function(x) plogis(1 - 2*x),
    mu0  = function(x,d) 5*sin(2*pi*x)*cos(2*pi*x) + d*(-0.4 + 2*x^2),
    noise_sd = 1
  ),
  list(
    name = "Model 2",
    rF0  = function(n) data.frame(X = runif(n, 0, 1)),
    rF   = function(m) data.frame(X = rbeta(m, 1, 1)),
    lambda = function(x) 1,
    p0   = function(x) plogis(-0.5 + x),
    mu0  = function(x,d) 0.5*sqrt(x) + d*(0.5 - x^2),
    noise_sd = 1
  ),
  list(
    name = "Model 3",
    rF0  = function(n) data.frame(X = runif(n, 0, 1)),
    rF   = function(m) data.frame(X = rbeta(m, 1, 1)),
    lambda = function(x) 1,
    p0   = function(x) plogis(0.5 - x),
    mu0  = function(x,d) x^2 + d*(1 - x),
    noise_sd = 1
  )
)

#################### Parameters ####################
ite          <- 1000
ns           <- c(1500, 3000, 6000)
M            <- 5000
gamma_eval.c <- 0.1  
gamma_eval.t <- 0.1
bs           <- "cr" # type of smoothers

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

      # df_trim <- subset(df, X >= 0 & X <= 1)
      # X_trim  <- df_trim$X
      
      # Control      
      df.c <- subset(df, D == 0)
      X.c  <- df.c$X
      Y.c  <-  as.numeric(df.c$Y) 
      
      # Treated     
      df.t <- subset(df, D == 1)
      X.t  <- df.t$X
      Y.t  <- as.numeric(df.t$Y) 
      
      # Cross Validation
      K <- 5                            
      W_hat_k <- numeric(K)   
      folds <- sample(rep(1:K, length.out = n))
      
      for (k in 1:K) {
        idx_train <- which(folds != k)   # “minus-k” sample
        idx_test  <- which(folds == k)   # fold k
        
        train <- df[idx_train, ]
        test  <- df[idx_test,  ]
        
        ps_mod   <- gam(D ~ s(X, bs = "cr"), data = train, family = binomial(), method = "REML")
        ctrl_mod <- gam(Y ~ s(X, bs = "cr"), data = train[train$D == 0, ], family = gaussian(), method = "REML", gamma = gamma_eval.c)
        trt_mod  <- gam(Y ~ s(X, bs = "cr"), data = train[train$D == 1, ], family = gaussian(), method = "REML", gamma = gamma_eval.t)
        
        p_hat    <- predict(ps_mod, newdata = test, type = "response")
        mu0_hat  <- predict(ctrl_mod, newdata = test, type = "response")
        mu1_hat  <- predict(trt_mod, newdata = test, type = "response")
        
        ind      <- as.numeric((mu1_hat - mu0_hat) >= 0)
        
        W_hat_k[k] <- mean(pmax(mu1_hat - mu0_hat, 0) + ind*(test$D/p_hat - (1 - test$D)/(1 - p_hat))*(test$Y - test$D*mu1_hat - (1 - test$D)*mu0_hat))
      }

      W_hat_i <- mean(W_hat_k)
      
      # Asymptotic variance
      ps_mod   <- gam(D ~  s(X), data = df, family = binomial()) # Default setting. No need to undersmooth
      ctrl_mod <- gam(Y ~ s(X), data = df.c, family = gaussian())
      trt_mod  <- gam(Y ~ s(X), data = df.t, family = gaussian())
      
      p_hat         <- predict(ps_mod, newdata = df, type = "response")
      mu0_X.t  <- predict(trt_mod, newdata = df, type = "response")
      mu0_X.c  <- predict(ctrl_mod, newdata = df, type = "response")
      
      ind      <- as.numeric((mu0_X.t - mu0_X.c) >= 0)
      resid    <- ifelse(df$D == 1, df$Y - mu0_X.t, df$Y - mu0_X.c)
      sig2_mod <- gam(I(resid^2) ~ s(X), data = df, family = gaussian())
      sig2_hat <- predict(sig2_mod, newdata = df, type = "response")
      lambda2  <- spec$lambda(df$X)^2
      asymp_var_hat_i <- mean(pmax(mu0_X.t - mu0_X.c, 0)^2) - mean(pmax(mu0_X.t - mu0_X.c, 0))^2 + sum(ind * lambda2 * sig2_hat / (p_hat * (1 - p_hat)), na.rm = TRUE)/n
      sd_hat_i <- sqrt(asymp_var_hat_i/n)
      
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