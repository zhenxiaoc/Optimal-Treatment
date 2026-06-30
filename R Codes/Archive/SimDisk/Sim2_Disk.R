
sim_int_disk = function(N = 1000, 
                        B = 1, 
                        M = 1000) {
  source("spline.R")
  library(npiv)
  library(mgcv)
  library(MASS)
  
  
  d = 2
  Est_B = rep(NA, B)
  CI_B = matrix(NA, nrow = B, ncol = 2)
  #Est_db_B = rep(NA, B)
  #CI_db_B = matrix(NA, nrow = B, ncol = 2)
  CI_btqt_B = matrix(NA, nrow = B, ncol = 2)
  #CI_btse_B = matrix(NA, nrow = B, ncol = 2)
  
  for (b in 1:B) {  
    X = matrix(runif(N * d, -2, 2), nrow = N, ncol = d)
    W = X
    Xnorm2 = X[,1]^2 + X[,2]^2
    h0_X = (1 - Xnorm2) * (4 + sin(X[,1])*X[,2] + cos(X[,2]))
    Y = h0_X + rnorm(N,0,1)
    
    alpha = 0.05
    basis = "tensor" #c("tensor","additive","glp"),
    J.x.degree = 3
    J.x.segments = 5
    knots= "quantiles" #c("uniform","quantiles"),
    
    X.min = NULL
    X.max= NULL
    
    # # Data-Driven Choice of Sieve Dimension
    # 
    # chooseJ_result <- npiv_choose_J(Y,
    #               X,
    #               W,
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
    # J.x.segments = chooseJ_result$J.x.seg
    
    #M = 1000
    int_lower = rep(-1.9, d)
    int_upper = rep(1.9, d)
    int_X = matrix(NA, nrow = M, ncol = 2)
    
    # Sobol Sequence
    
    library(qrng)
    sobol_points <- sobol(M, d = 2, randomize = "none")
    for (j in 1:d) {
      int_X[,j] <- int_lower[j] +  sobol_points[,j] * (int_upper[j] - int_lower[j])
    }
    
    
    Psi_X <- prodspline(x = X,
                        K = cbind(rep(J.x.degree,NCOL(X)),rep(J.x.segments,NCOL(X))),
                        knots = knots,
                        basis = basis,
                        x.min = X.min,
                        x.max = X.max)
    d_sieve <- ncol(Psi_X)
    
    Psi_int_X <- prodspline(x = X,
                            xeval = int_X,
                            K = cbind(rep(J.x.degree,NCOL(X)),rep(J.x.segments,NCOL(X))),
                            knots = knots,
                            basis = basis,
                            x.min = X.min,
                            x.max = X.max)
    
    
    # Series regression
    
    XXinvX <- ginv(t(Psi_X) %*% Psi_X) %*%  t(Psi_X)
    beta <- XXinvX %*% Y
    h_X <- Psi_X %*% beta
    
    int_h_predict = Psi_int_X %*%beta
    int_h = 3.8^2 * mean((int_h_predict >= 0))
    
    # Compute true regression function values on numerical integration points
    int_Xnorm2 = int_X[,1]^2 + int_X[,2]^2
    int_h0 = 3.8^2 * mean((int_Xnorm2 <= 1))
    
    # ## Bootstrap CI
    # 
    # B_boot = 500
    # int_h_boot = rep(NA, B_boot)
    # 
    # for (bb in 1:B_boot) {
    #   
    #   ind_bb = sample(c(1:N), N, replace = T)
    #   X_bb = X[ind_bb,]
    #   Y_bb = Y[ind_bb]
    #   Psi_X_bb <- prodspline(x = X,
    #                          xeval = X_bb,
    #                          K = cbind(rep(J.x.degree,NCOL(X)),rep(J.x.segments+1,NCOL(X))),
    #                          knots = knots,
    #                          basis = basis,
    #                          x.min = X.min,
    #                          x.max = X.max)
    #   
    #   Psi_int_X_bb <- prodspline(x = X,
    #                              xeval = int_X,
    #                              K = cbind(rep(J.x.degree,NCOL(X)),rep(J.x.segments+1,NCOL(X))),
    #                              knots = knots,
    #                              basis = basis,
    #                              x.min = X.min,
    #                              x.max = X.max)
    #   
    #   XXinvX_bb <- ginv(t(Psi_X_bb) %*% Psi_X_bb) %*%  t(Psi_X_bb)
    #   beta_bb <- XXinvX_bb %*% Y_bb
    #   int_h_predict_bb = Psi_int_X_bb %*% beta_bb
    #   int_h_boot[bb] = 3.8^2 * mean((int_h_predict_bb >= 0))
    # }
    # 
    # CI_btqt_B[b,] = c(quantile(int_h_boot, alpha/2, na.rm = T),
    #                   quantile(int_h_boot, 1 - alpha/2, na.rm = T))
    
    # # Bootstrap se
    # se_boot = sd(int_h_boot)
    # z_alpha = qnorm(1 - alpha/2)
    # MoE_boot = z_alpha * se_boot
    # CI_btse_B[b,] = c(int_h - MoE_boot, int_h + MoE_boot)
    
    
    # Analytical CI
    # Compute Df(h)[Psi]
    
    M2 = 100000
    eps = 0.001
    int_X = matrix(NA, nrow = M2, ncol = 2)
    sobol_points <- sobol(M2, d = 2, randomize = "none")
    for (j in 1:d) {
      int_X[,j] <- int_lower[j] +  sobol_points[,j] * (int_upper[j] - int_lower[j])
    }
    
    Psi_int_X <- prodspline(x = X,
                            xeval = int_X,
                            K = cbind(rep(J.x.degree,NCOL(X)),rep(J.x.segments,NCOL(X))),
                            knots = knots,
                            basis = basis,
                            x.min = X.min,
                            x.max = X.max)
    
    int_h_predict = Psi_int_X %*%beta
    ind_good = (int_h_predict > -eps) & (int_h_predict < eps)
    Psi_h_pred = matrix(rep(ind_good, d_sieve), nrow = M2, ncol = d_sieve) * Psi_int_X
    int_Psi = 3.8^2 *colMeans(Psi_h_pred)/(2*eps)
    
    ## Compute asymptotic standard errors
    term_u <- Y - Psi_X %*% beta
    term_sandwich <- t(t(XXinvX) * as.numeric(term_u))%*%(t(XXinvX) * as.numeric(term_u ))
    asy.Var <- t(int_Psi) %*% term_sandwich %*%int_Psi
    asy.se <- sqrt(asy.Var)
    
    
    # Construct confidence interval
    z_alpha = qnorm(1 - alpha/2)
    MoE = z_alpha * asy.se
    CI = c(int_h - MoE, int_h + MoE)
    
    # Save results
    Est_B[b] = int_h
    CI_B[b,] = CI
  }  
  
  return(list(
    J_seg = J.x.segments,
    d_sieve = d_sieve,
    Est = Est_B,
    CI = CI_B,
    true_val = pi,
    eps = eps,
    CI_btqt = CI_btqt_B
  ))
  
}

if (T) {
  
  N = 500  # Sample size
  B = 1000   # Number of simulations
  M = 5000  # Number of Sobol points for numerical integration
  
  set.seed(123)
  sim_results <- sim_int_disk(N = N, B = B, M = M)
  
  # Performance measures
  Error = sim_results$Est - sim_results$true_val
  bias = mean(Error)
  stdev  = sd(sim_results$Est)
  rMSE = sqrt(mean(Error^2))
  CI = sim_results$CI
  CI_cover_B = (CI[,1] <= pi & pi <= CI[,2])
  CI_cover = mean(CI_cover_B)
  CI_lower_mean = mean(CI[,1])
  Est_mean = mean(sim_results$Est)
  CI_upper_mean = mean(CI[,2])
  CI_length_mean = CI_upper_mean - CI_lower_mean
  
  CI_btqt = sim_results$CI_btqt
  CI_btqt_cover = mean((CI_btqt[,1] <= pi & pi <= CI_btqt[,2]))
  
  sum_results <- list(J_seg = sim_results$J_seg,
                      d_sieve = sim_results$d_sieve,
                      rMSE = rMSE,
                      bias = bias,
                      stdev = stdev,
                      CI_cover = CI_cover, 
                      CI_lower_mean = CI_lower_mean,
                      Est_mean  = Est_mean,
                      CI_upper_mean = CI_upper_mean,
                      CI_length_mean = CI_length_mean,
                      CI_btqt_cover = CI_btqt_cover)
  
  filename = paste0("Sim_int_disk_N", N, ".RData")
  save.image(filename)
  
  filename = paste0("Sim_int_disk_N", N, ".txt")
  sink(filename)
  cat("N = ", N, "\n")
  cat("M = ", M, "\n")
  cat("B = ", B, "\n")
  cat("True = ", pi , "\n")
  cat("\n")
  print(sum_results)
  sink()
}

