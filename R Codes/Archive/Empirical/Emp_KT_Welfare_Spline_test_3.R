rm(list = ls())

source("spline.R")
library(npiv)
library(mgcv)
library(MASS)
################################# Configurations ############################### 
cost        <- T                                 # If cost = F, Y = the 30-month earnings
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

df_trim.c <- subset(df_trim, ind == 0)          # Trimmed control group data
Y_trim.c  <- df_trim.c$earnings
X_trim.c  <- cbind(df_trim.c$prevearn, df_trim.c$edu)

df_trim.t <- subset(df_trim, ind == 1)          # Trimmed treated group data
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
J.x.segments   <- 8 #Sieve dimension chosen by npiv_choose_J

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
# J.x.segments   <- chooseJ_result$J.x.seg

Psi_X <- prodspline(x = X,
                    K = cbind(rep(J.x.degree,NCOL(X)),rep(J.x.segments,NCOL(X))),
                    knots = knots,
                    basis = basis,
                    x.min = X.min,
                    x.max = X.max)
beta <- ginv(t(Psi_X) %*% Psi_X) %*% t(Psi_X) %*% D


Psi_X_trim <- prodspline(x = X,
                         xeval = X_trim,
                         K = cbind(rep(J.x.degree,NCOL(X)),rep(J.x.segments,NCOL(X))),
                         knots = knots,
                         basis = basis,
                         x.min = X.min,
                         x.max = X.max)

p_hat     <- as.numeric(Psi_X_trim %*% beta)
ind       <- as.numeric((mu0_X_trim.t - mu0_X_trim.c) >= 0)
resid     <- ifelse(df_trim$D == 1, df_trim$earnings - mu0_X_trim.t, df_trim$earnings - mu0_X_trim.c)
lambda2   <- 1
asy.var   <- var(pmax(mu0_X_trim.t - mu0_X_trim.c, 0)) + 
             mean(ind * lambda2 * resid^2 / (p_hat * (1 - p_hat)))
sd_hat    <- sqrt(asy.var/nrow(df_trim))


################################# Done! ######################################## 
report <- data.frame(
  W_hat   = W_hat,
  SE      = sd_hat,
  CI_low  = W_hat - 1.96 * sd_hat,
  CI_high = W_hat + 1.96 * sd_hat
)

print(report)

