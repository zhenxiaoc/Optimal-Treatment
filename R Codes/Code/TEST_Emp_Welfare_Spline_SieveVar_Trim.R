rm(list = ls())   
source("spline.R")
library(npiv)
library(mgcv)
library(qrng)
library(MASS)
library(Matrix)
################################# Configurations ############################### 
cost_values <- c(FALSE, TRUE)
reports_df  <- data.frame()
for (cost in cost_values) {                     # If cost = F, Y = the 30-month earnings
                                                 # If cost = T, Y = the 30-month earnings - 
                                                 #                  the average cost of services per treatment assignment

################################### Data #######################################
df   <- read.csv("KT_Data1.csv")                 # Untrimmed data
if (cost == T){
  df$earnings <- df$earnings - df$D * 774
}
X    <- cbind(df$prevearn, df$edu)
D    <- df$D

df.c <- subset(df, D == 0)                       # Control group data
Y.c  <- df.c$earnings
X.c  <- cbind(df.c$prevearn, df.c$edu)

df.t <- subset(df, D == 1)                       # Treated group data
Y.t  <- df.t$earnings
X.t  <- cbind(df.t$prevearn, df.t$edu)

df$ind <- (df$prevearn < max(df.c$prevearn)) *   # An indicator for whether the data is 
  (df$prevearn > min(df.c$prevearn)) *           # in the overlap of the treated
  (df$prevearn < max(df.t$prevearn)) *           # and the control data. ind = 1 if yes. 
  (df$prevearn > min(df.t$prevearn)) *
  (df$edu < max(df.c$edu)) *
  (df$edu > min(df.c$edu)) *
  (df$edu < max(df.t$edu)) *
  (df$edu > min(df.t$edu)) 

df_trim <- subset(df, ind == 1)                 # Trimmed data
X_trim  <- cbind(df_trim$prevearn, df_trim$edu)

df_trim.c <- subset(df_trim, D == 0)            # Trimmed control group data
Y_trim.c  <- df_trim.c$earnings
X_trim.c  <- cbind(df_trim.c$prevearn, df_trim.c$edu)

df_trim.t <- subset(df_trim, D == 1)            # Trimmed treated group data
Y_trim.t  <- df_trim.t$earnings
X_trim.t  <- cbind(df_trim.t$prevearn, df_trim.t$edu)


######################## Welfare Functional Estimation ######################### 
# Tuning
J.x.degree     <- 3              # Cubic splines
J.x.segments.c <- 1              # Sieve dimension chosen by npiv_choose_J
J.x.segments.t <- 4              # Sieve dimension chosen by npiv_choose_J
knots          <- "uniform"
basis          <- "tensor" 
X.min          <- NULL
X.max          <- NULL

# chooseJ_result.c <- npiv_choose_J(Y.c,
#                                   X.c,
#                                   X.c,
#                                   X.grid = NULL,
#                                   J.x.degree = 3,
#                                   K.w.degree = 4,
#                                   K.w.smooth = 2,
#                                   knots = "quantile",
#                                   basis = "tensor",
#                                   X.min = NULL,
#                                   X.max = NULL,
#                                   W.min = NULL,
#                                   W.max = NULL,
#                                   grid.num = 50,
#                                   boot.num = 99,
#                                   check.is.fullrank= FALSE,
#                                   progress = TRUE)
# J.x.segments.c    <- chooseJ_result.c$J.x.seg
# 
# chooseJ_result.t <- npiv_choose_J(Y.t,
#                                   X.t,
#                                   X.t,
#                                   X.grid = NULL,
#                                   J.x.degree = 3,
#                                   K.w.degree = 4,
#                                   K.w.smooth = 2,
#                                   knots = "uniform",
#                                   basis = "tensor",
#                                   X.min = NULL,
#                                   X.max = NULL,
#                                   W.min = NULL,
#                                   W.max = NULL,
#                                   grid.num = 50,
#                                   boot.num = 99,
#                                   check.is.fullrank= FALSE,
#                                   progress = TRUE)
# J.x.segments.t    <- chooseJ_result.t$J.x.seg

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

# Welfare estimate
W_hat   <- mean(pmax(mu0_X_trim.t - mu0_X_trim.c, 0)) 

################################ Inference ##################################### 
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

asymp_var_hat_i  <- mean(pmax(mu0_X_trim.t - mu0_X_trim.c, 0)^2) - 
                    mean(pmax(mu0_X_trim.t - mu0_X_trim.c, 0))^2 +
                    t(Bun) %*% Patty %*% Bun * (nt + nc)
se   <- sqrt(asymp_var_hat_i/length(df_trim$earnings))



################################# Done! ######################################## 
report <- data.frame(
  W_hat   = W_hat,
  SE      = se,
  CI_low  = W_hat - 1.96 * se,
  CI_high = W_hat + 1.96 * se
)

report <- data.frame(
  cost       = cost,
  cost_label = ifelse(cost, "cost_774", "no_cost"),
  report
)
reports_df <- rbind(reports_df, report)
print(report)
}

dir.create("results", showWarnings = FALSE)
write.csv(reports_df,
          file = file.path("results", "TEST_Emp_Welfare_Spline_SieveVar_Trim_results.csv"),
          row.names = FALSE)
save(reports_df,
     file = file.path("results", "TEST_Emp_Welfare_Spline_SieveVar_Trim_results.Rdata"))
