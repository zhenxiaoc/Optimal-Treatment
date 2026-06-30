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
    name   = "Model 8",
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
    name = "Model 9",
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
    name = "Model 10",
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
      
      df_trim <- subset(df, X > min(df.t$X) & X < max(df.t$X) & X > min(df.c$X) & X < max(df.c$X))
      X_trim <- df_trim$X
      Y_trim <- df_trim$Y
      D_trim <- df_trim$D
      
      df_trim.c <- subset(df_trim, D == 0)
      Y_trim.c  <- df_trim.c$Y
      
      df_trim.t <- subset(df_trim, D == 1)
      Y_trim.t  <- df_trim.t$Y
      
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
      # Estimating mu_0(x , 0) and evaluated it on X_trim
      Psi_X.c      <- prodspline(x = X.c,
                                 K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                                 knots = knots,
                                 basis = basis,
                                 x.min = X.min,
                                 x.max = X.max)
      beta.c       <- ginv(t(Psi_X.c) %*% Psi_X.c) %*% t(Psi_X.c) %*% Y.c
      d_sieve.c    <- ncol(Psi_X.c)
      Psi_X_trim.c <- prodspline(x = X.c,
                                 xeval = X_trim,
                                 K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                                 knots = knots,
                                 basis = basis,
                                 x.min = X.min,
                                 x.max = X.max)
      mu0_X_trim.c <- Psi_X_trim.c %*% beta.c
      
      # Estimating mu_0(x , 1) and evaluated it on X_trim
      Psi_X.t      <- prodspline(x = X.t,
                                 K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                                 knots = knots,
                                 basis = basis,
                                 x.min = X.min,
                                 x.max = X.max)
      beta.t       <- ginv(t(Psi_X.t) %*% Psi_X.t) %*% t(Psi_X.t) %*% Y.t
      d_sieve.t    <- ncol(Psi_X.t)
      Psi_X_trim.t <- prodspline(x = X.t,
                                 xeval = X_trim,
                                 K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                                 knots = knots,
                                 basis = basis,
                                 x.min = X.min,
                                 x.max = X.max)
      mu0_X_trim.t <- Psi_X_trim.t %*% beta.t
      
      # Second stage
      W_hat_i     <- mean(pmax(mu0_X_trim.t- mu0_X_trim.c, 0))
      
      # Asymptotic variance
      # I(-eps < h(x) < eps)
      int_h_predict <- Psi_X_trim.t %*% beta.t - Psi_X_trim.c %*% beta.c
      ind_good      <- (int_h_predict >= 0)
      
      
      # Pathwise derivative
      term_1 <- matrix(rep(ind_good, d_sieve.t + d_sieve.c), ncol = d_sieve.t + d_sieve.c)
      term_3 <- cbind(Psi_X_trim.t, -Psi_X_trim.c)
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
      asy.var  <- mean(pmax(mu0_X_trim.t - mu0_X_trim.c, 0)^2) - 
                  mean(pmax(mu0_X_trim.t - mu0_X_trim.c, 0))^2 +
                  t(Bun) %*% Patty %*% Bun * (nt + nc)
      se_i     <- sqrt(asy.var/length(Y_trim))
      
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

save(final_df, res_mat, file = "sim_results_Thm2_1D_SieveVar.Rdata")