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
    name   = "Model 1",
    rF0    = function(n) data.frame(X = runif(n, 0, 1)),
    rF     = function(m) data.frame(X = runif(n, 0, 1)),
    lambda = function(x) 1,
    p0     = function(x) plogis(1 - 2*x),
    mu0    = function(x,d) 5*sin(2*pi*x)*cos(2*pi*x) + d*(-0.4 + 2*x^2),
    noise_sd = 1,
    J.x.segments.c = 8,
    J.x.segments.t = 8,
    J.x.segments   = 1
  ),
  list(
    name = "Model 2",
    rF0  = function(n) data.frame(X = runif(n, 0, 1)),
    rF   = function(m) data.frame(X = runif(n, 0, 1)),
    lambda = function(x) 1,
    p0   = function(x) plogis(-0.5 + x),
    mu0  = function(x,d) 0.5*abs(x) + d*(0.5 - x^2),
    noise_sd = 1,
    J.x.segments.c = 1,
    J.x.segments.t = 4,
    J.x.segments   = 1
  ),
  list(
    name = "Model 3",
    rF0  = function(n) data.frame(X = runif(n, 0, 1)),
    rF   = function(m) data.frame(X = runif(n, 0, 1)),
    lambda = function(x) 1,
    p0   = function(x) plogis(0.5 - x),
    mu0  = function(x,d) x^2 + d*(1 - x),
    noise_sd = 1,
    J.x.segments.c = 4,
    J.x.segments.t = 1,
    J.x.segments   = 1
  )
)

#################### Simulation #################### 
ite <- 2000                 
ns  <- c(1500, 3000, 6000) 
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
  int_lower = rep(0, 1)
  int_upper = rep(1, 1)
  sobol_points <- sobol(M, d = 1, randomize = "none")
  int_X <- int_lower +  sobol_points * (int_upper - int_lower)
  int_X <- data.frame(X = int_X)
  W_true <- mean(pmax(spec$mu0(int_X$X, 1) - spec$mu0(int_X$X, 0), 0))
  
  for (n in ns){
    cat("Running spec =", spec$name, "| n =", n, "\n")
    
    # Define one iteration
    one_ite <- function(dummy) {
      df <- spec$rF0(n)
      df$D <- rbinom(n, 1, spec$p0(df$X))
      noise <- rnorm(nrow(df), 0, spec$noise_sd)
      df$Y <- spec$mu0(df$X, df$D) + noise
      Y <- df$Y
      X <- df$X
      D <- df$D
      
      # Control
      df.c <- subset(df, D == 0)
      X.c  <- df.c$X
      Y.c <-  as.numeric(df.c$Y) 
      
      # Treated
      df.t <- subset(df, D == 1)
      X.t  <- df.t$X
      Y.t <- as.numeric(df.t$Y) 
      
      J.x.segments.c <- spec$J.x.segments.c              # Sieve dimension chosen by npiv_choose_J
      J.x.segments.t <- spec$J.x.segments.t
      J.x.segments   <- spec$J.x.segments
      
      # chooseJ_result <- npiv_choose_J(Y.c,
      #               X.c,
      #               X.c,
      #               X.grid = NULL,
      #               J.x.degree = 3,
      #               K.w.degree = 4,
      #               K.w.smooth = 2,
      #               knots = "uniform",
      #               basis = "tensor",
      #               X.min = NULL,
      #               X.max = NULL,
      #               W.min = NULL,
      #               W.max = NULL,
      #               grid.num = 50,
      #               boot.num = 99,
      #               check.is.fullrank= FALSE,
      #               progress = TRUE)
      # 
      # chooseJ_result$J.x.seg
      # 
      # chooseJ_result <- npiv_choose_J(Y.t,
      #               X.t,
      #               X.t,
      #               X.grid = NULL,
      #               J.x.degree = 3,
      #               K.w.degree = 4,
      #               K.w.smooth = 2,
      #               knots = "uniform",
      #               basis = "tensor",
      #               X.min = NULL,
      #               X.max = NULL,
      #               W.min = NULL,
      #               W.max = NULL,
      #               grid.num = 50,
      #               boot.num = 99,
      #               check.is.fullrank= FALSE,
      #               progress = TRUE)
      # 
      # chooseJ_result$J.x.seg
      # 
      # chooseJ_result <- npiv_choose_J(D,
      #               X,
      #               X,
      #               X.grid = NULL,
      #               J.x.degree = 3,
      #               K.w.degree = 4,
      #               K.w.smooth = 2,
      #               knots = "uniform",
      #               basis = "tensor",
      #               X.min = NULL,
      #               X.max = NULL,
      #               W.min = NULL,
      #               W.max = NULL,
      #               grid.num = 50,
      #               boot.num = 99,
      #               check.is.fullrank= FALSE,
      #               progress = TRUE)
      # 
      # chooseJ_result$J.x.seg
      
      # First stage
      # Cross Validation
      K <- 5                            
      W_hat_k <- numeric(K)   
      folds <- sample(rep(1:K, length.out = n))
      
      for (k in 1:K) {
        idx_train <- which(folds != k)   # “minus-k” sample
        idx_test  <- which(folds == k)   # fold k
        
        df_train <- df[idx_train, ]
        X_train  <- df_train$X
        Y_train  <- df_train$Y
        D_train  <- df_train$D
        
        df_test  <- df[idx_test,  ]
        
        # Control in training
        df_train.c <- subset(df_train, D == 0)
        X_train.c  <- df_train.c$X
        Y_train.c  <- as.numeric(df_train.c$Y) 
        
        # Treated in training
        df_train.t <- subset(df_train, D == 1)
        X_train.t  <- df_train.t$X
        Y_train.t  <- as.numeric(df_train.t$Y) 
        
        df_test_trim <- subset(df_test, X > min(df_train.t$X) & X < max(df_train.t$X) & X > min(df_train.c$X) & X < max(df_train.c$X))
        X_test_trim  <- df_test_trim$X
        Y_test_trim  <- df_test_trim$Y
        D_test_trim  <- df_test_trim$D
        
        # Control in training
        df_test_trim.c <- subset(df_test_trim, D == 0)
        X_test_trim.c  <- df_test_trim.c$X
        Y_test_trim.c  <- as.numeric(df_test_trim.c$Y) 
        
        # Treated in training
        df_test_trim.t <- subset(df_test_trim, D == 1)
        X_test_trim.t  <- df_test_trim.t$X
        Y_test_trim.t  <- as.numeric(df_test_trim.t$Y)  
        
        
        # First esimate propensity score. 
        # Drop obs with propensity bigger than 1 or smaller than 0.
        Psi_X_train <- prodspline(x = X_train,
                                 K = cbind(rep(J.x.degree,NCOL(X_train)),rep(J.x.segments,NCOL(X_train))),
                                 knots = knots,
                                 basis = basis,
                                 x.min = X.min,
                                 x.max = X.max)
        beta <- ginv(t(Psi_X_train) %*% Psi_X_train) %*% t(Psi_X_train) %*% D_train
        
        Psi_X_test_trim <- prodspline(x = X_train,
                                  xeval = X_test_trim,
                                  K = cbind(rep(J.x.degree,NCOL(X_train)),rep(J.x.segments,NCOL(X_train))),
                                  knots = knots,
                                  basis = basis,
                                  x.min = X.min,
                                  x.max = X.max)
        p_hat     <- as.numeric(Psi_X_test_trim %*% beta)
        
        idx <- which(p_hat > 0 & p_hat < 1)
        p_hat <- p_hat[idx]
        X_test_trim2 <- X_test_trim[idx]
        Y_test_trim2 <- Y_test_trim[idx]
        D_test_trim2 <- D_test_trim[idx]
        
        #Estimating mu_0(x , 0) and evaluated it on X_test_trim2
        Psi_X_train.c <- prodspline(x = X_train.c,
                                   K = cbind(rep(J.x.degree,NCOL(X_train.c)),rep(J.x.segments.c,NCOL(X_train.c))),
                                   knots = knots,
                                   basis = basis,
                                   x.min = X.min,
                                   x.max = X.max)
        beta.c       <- ginv(t(Psi_X_train.c) %*% Psi_X_train.c) %*% t(Psi_X_train.c) %*% Y_train.c
        d_sieve.c    <- ncol(Psi_X_train.c)
        
        Psi_X_test_trim2.c <- prodspline(x = X_train.c,
                                         xeval = X_test_trim2,
                                         K = cbind(rep(J.x.degree,NCOL(X_train.c)),rep(J.x.segments.c,NCOL(X_train.c))),
                                         knots = knots,
                                         basis = basis,
                                         x.min = X.min,
                                         x.max = X.max)
        mu0_X_test_trim2.c <- Psi_X_test_trim2.c %*% beta.c
        
        # Estimating mu_0(x , 1) and evaluated it on X_test_trim2
        Psi_X_train.t <- prodspline(x = X_train.t,
                                    K = cbind(rep(J.x.degree,NCOL(X_train.t)),rep(J.x.segments.t,NCOL(X_train.t))),
                                    knots = knots,
                                    basis = basis,
                                    x.min = X.min,
                                    x.max = X.max)
        beta.t       <- ginv(t(Psi_X_train.t) %*% Psi_X_train.t) %*% t(Psi_X_train.t) %*% Y_train.t
        d_sieve.t    <- ncol(Psi_X_train.t)
        
        Psi_X_test_trim2.t <- prodspline(x = X_train.t,
                                         xeval = X_test_trim2,
                                         K = cbind(rep(J.x.degree,NCOL(X_train.t)),rep(J.x.segments.t,NCOL(X_train.t))),
                                         knots = knots,
                                         basis = basis,
                                         x.min = X.min,
                                         x.max = X.max)
        mu0_X_test_trim2.t <- Psi_X_test_trim2.t %*% beta.t

        
        ind      <- as.numeric((mu0_X_test_trim2.t - mu0_X_test_trim2.c) >= 0)
        
        W_hat_k[k] <- mean(pmax(mu0_X_test_trim2.t - mu0_X_test_trim2.c, 0) + 
                             ind * 
                             (D_test_trim2/p_hat - (1 - D_test_trim2)/(1 - p_hat)) * 
                             (Y_test_trim2 - D_test_trim2*mu0_X_test_trim2.t - (1 - D_test_trim2)*mu0_X_test_trim2.c))
      }
      
      W_hat_i <- mean(W_hat_k)
      
      #### Aysmptotic Variance
      df_trim <- subset(df, X > min(df.t$X) & X < max(df.t$X) & X > min(df.c$X) & X < max(df.c$X))
      X_trim <- df_trim$X
      Y_trim <- df_trim$Y
      D_trim <- df_trim$D
      
      Psi_X_trim <- prodspline(x = X_trim,
                               K = cbind(rep(J.x.degree,NCOL(X_trim)),rep(J.x.segments,NCOL(X_trim))),
                               knots = knots,
                               basis = basis,
                               x.min = X.min,
                               x.max = X.max)
      beta <- ginv(t(Psi_X_trim) %*% Psi_X_trim) %*% t(Psi_X_trim) %*% D_trim
      p_hat     <- as.numeric(Psi_X_trim %*% beta)
      
      idx <- which(p_hat > 0 & p_hat < 1)
      p_hat <- p_hat[idx]
      X_trim2 <- X_trim[idx]
      Y_trim2 <- Y_trim[idx]
      D_trim2 <- D_trim[idx]
      
      # Estimating mu_0(x , 0) and evaluated it on X_trim2
      Psi_X.c <- prodspline(x = X.c,
                            K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                            knots = knots,
                            basis = basis,
                            x.min = X.min,
                            x.max = X.max)
      beta.c <- ginv(t(Psi_X.c) %*% Psi_X.c) %*% t(Psi_X.c) %*% Y.c
      
      Psi_X_trim2.c <- prodspline(x = X.c,
                                  xeval = X_trim2,
                                  K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                                  knots = knots,
                                  basis = basis,
                                  x.min = X.min,
                                  x.max = X.max)
      mu0_X_trim2.c <- Psi_X_trim2.c %*% beta.c
      
      # Estimating mu_0(x , 1) and evaluated it on X_trim2
      Psi_X.t <- prodspline(x = X.t,
                            K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                            knots = knots,
                            basis = basis,
                            x.min = X.min,
                            x.max = X.max)
      beta.t <- ginv(t(Psi_X.t) %*% Psi_X.t) %*% t(Psi_X.t) %*% Y.t
      
      Psi_X_trim2.t <- prodspline(x = X.t,
                                  xeval = X_trim2,
                                  K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                                  knots = knots,
                                  basis = basis,
                                  x.min = X.min,
                                  x.max = X.max)
      mu0_X_trim2.t <- Psi_X_trim2.t %*% beta.t
      
      # Estimate propensity score
      ind      <- as.numeric((mu0_X_trim2.t - mu0_X_trim2.c) >= 0)
      resid    <- ifelse(D_trim2 == 1, Y_trim2 - mu0_X_trim2.t, Y_trim2 - mu0_X_trim2.c)
      lambda2   <- 1
      asymp_var_hat_i <- mean(pmax(mu0_X_trim2.t - mu0_X_trim2.c, 0)^2) - 
        mean(pmax(mu0_X_trim2.t - mu0_X_trim2.c, 0))^2 + 
        sum(ind * lambda2 * resid^2 / (p_hat * (1 - p_hat)), na.rm = TRUE)/length(Y_trim2)
      se_i <- sqrt(asymp_var_hat_i/length(Y_trim2))
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
save(final_df, res_mat, file = "sim_results_DML_1D_Spline.Rdata")
