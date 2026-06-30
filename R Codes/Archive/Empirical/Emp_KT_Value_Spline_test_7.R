rm(list = ls())

source("spline.R")
library(npiv)
library(mgcv)
library(MASS)

################################# Configurations ############################### 
cost        <- F                                 # If cost = F, Y = the 30-month earnings
# If cost = T, Y = the 30-month earnings - 
#                  the average cost of services per treatment assignment

if_sobol    <- T                                 # If if_sobol = T, draw Sobol points and reweigh them for numerical integral in the inference part
# If if_sobol = F, sample from a fine gird based on estimated density for the numerical integral

if_boundary <- F                                 # If if_boundary = T, do boundary kernel density estimation
# If if_boundary = F, do naive Gaussian density estimation
smoothing   <- 100                               # Control how smooth the density plot looks like

eta         <- 0.01


################################### Data #######################################
df   <- read.csv("KT_Data1.csv")                 # Untrimmed data
if (cost == T){
  df$earnings <- df$earnings - df$D * 774
}
X    <- cbind(df$prevearn, df$edu)

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

df_trim.c <- subset(df_trim, D == 0)          # Trimmed control group data
Y_trim.c  <- df_trim.c$earnings
X_trim.c  <- cbind(df_trim.c$prevearn, df_trim.c$edu)                                 # Bug one: tr and co in X_trim

df_trim.t <- subset(df_trim, D == 1)          # Trimmed treated group data
Y_trim.t  <- df_trim.t$earnings
X_trim.t  <- cbind(df_trim.t$prevearn, df_trim.t$edu)


######################### Value Functional Estimation ######################### 
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

# Value functional estimate, where v_0(x) = 1
V_hat        <- mean((mu0_X_trim.t - mu0_X_trim.c > 0)) 

###################### Distribution of h(x) over X_trim ######################## 
h_X_trim <- Psi_X_trim.t %*% beta.t - Psi_X_trim.c %*% beta.c

# Eliminated outliers
rng      <- c(q1 <- quantile(h_X_trim, 0.25) - 1.5*IQR(h_X_trim),
              quantile(h_X_trim, 0.75) + 1.5*IQR(h_X_trim))
h_X_trim <- h_X_trim[h_X_trim >= rng[1] & h_X_trim <= rng[2]]

# Define eps according to the scale of h over X_trim
eps      <- eta*sd(h_X_trim)

######################### Density Estimation  #########################          # Bug: Find density over X_trim
library(ks)
H      <- Hpi(X_trim)
#H      <- Hscv(X_trim)

if (if_boundary == T){
  fhat <- ks::kde.boundary(x = X_trim,
                           xmin = c(min(X_trim[, 1]), min(X_trim[, 2])),     # compact support defined by max and min
                           xmax = c(max(X_trim[, 1]), max(X_trim[, 2])),
                           boundary.kernel="beta",
                           H = H*smoothing)                                  # H*smoothing to provide sufficient smoothing    
}else{
  fhat <- ks::kde(x = X_trim, H = H*smoothing)     
}

######################### Visualization Tools ######################### 
library(rgl)

# build evaluation grid
nx <- 50
gx1 <- seq(min(X_trim[,1]), max(X_trim[,1]), length.out = nx)
gx2 <- seq(min(X_trim[,2]), max(X_trim[,2]), length.out = nx)
G   <- as.matrix(expand.grid(gx1, gx2))
z   <- matrix(predict(fhat, x = G), nrow = nx, byrow = FALSE)

# interactive surface
persp3d(gx1, gx2, z,
        col = "lightblue", alpha = 0.7,
        xlab = "X1", ylab = "X2", zlab = "Density")


################################ Inference ##################################### 
set.seed(123)

if (if_sobol == T){
  # Draw Sobol points from the compact support
  library(qrng)
  M            <- 1000000
  X_samp       <- matrix(NA, nrow = M, ncol = 2)
  sobol_points <- sobol(M, d = 2, randomize = "none")
  int_lower    <- c(min(X_trim[, 1]), min(X_trim[, 2]))
  int_upper    <- c(max(X_trim[, 1]), max(X_trim[, 2]))
  for (j in 1:2) {
    X_samp[,j] <- int_lower[j] +  sobol_points[,j] * (int_upper[j] - int_lower[j])
  }
  
  # psi^(K0)(x) for controls
  Psi_X_samp.c <- prodspline(x = X.c,
                             xeval = X_samp,
                             K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                             knots = knots,
                             basis = basis,
                             x.min = X.min,
                             x.max = X.max)
  
  # psi^(K1)(x) for treated
  Psi_X_samp.t <- prodspline(x = X.t,
                             xeval = X_samp,
                             K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                             knots = knots,
                             basis = basis,
                             x.min = X.min,
                             x.max = X.max)  
  
  h_X_samp    <- Psi_X_samp.t %*% beta.t - Psi_X_samp.c %*% beta.c
  ind_good    <- (h_X_samp > -eps) & (h_X_samp < eps)
  num         <- sum(ind_good)
  
  fx_scaled   <- predict(fhat, x = X_samp) *       # Rescale based on Randon-Nikodym
    (max(X_trim[, 1]) - min(X_trim[, 1])) * (max(X_trim[, 2]) - min(X_trim[, 2]))
  
  term_1      <- matrix(rep(ind_good, d_sieve.t + d_sieve.c), ncol = d_sieve.t + d_sieve.c)
  term_2      <- cbind(Psi_X_samp.t, -Psi_X_samp.c)
  term_3      <- matrix(rep(fx_scaled,  d_sieve.t + d_sieve.c), ncol = d_sieve.t + d_sieve.c)
  
  Bun         <- 1/(2*eps)  * colMeans(term_1 * term_2 * term_3)
  
}else{
  # Creating a fine grid on the compact support
  x1           <- seq(min(X_trim[,1]), max(X_trim[,1]), length.out = 1000) 
  x2           <- seq(min(X_trim[,2]), max(X_trim[,2]), length.out = 1000)
  G            <- expand.grid(x1, x2)
  
  # Sampling from the grid
  fx           <- predict(fhat, x = G) 
  fx_norm      <- fx / sum(fx)
  M            <- 1000000
  idx          <- sample(seq_along(fx_norm), size = M, replace = T, prob = fx_norm)
  X_samp       <- G[idx, ]
  
  # psi^(K0)(x) for controls
  Psi_X_samp.c <- prodspline(x = X.c,
                             xeval = X_samp,
                             K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                             knots = knots,
                             basis = basis,
                             x.min = X.min,
                             x.max = X.max)
  
  # psi^(K1)(x) for treated
  Psi_X_samp.t <- prodspline(x = X.t,
                             xeval = X_samp,
                             K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                             knots = knots,
                             basis = basis,
                             x.min = X.min,
                             x.max = X.max)  
  
  h_predict    <- Psi_X_samp.t %*% beta.t - Psi_X_samp.c %*% beta.c
  ind_good     <- (h_predict > -eps) & (h_predict < eps)
  num          <- sum(ind_good)
  
  term_1       <- matrix(rep(ind_good, d_sieve.t + d_sieve.c), ncol = d_sieve.t + d_sieve.c)
  term_2       <- cbind(Psi_X_samp.t, -Psi_X_samp.c)
  
  Bun          <- 1/(2*eps)  * colMeans(term_1 * term_2)
}


# OLS formula
# potential problem in 7
library(Matrix)
# Psi_X_trim.cc <- prodspline(x = X.c,
#                             xeval = X_trim.c,
#                             K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
#                             knots = knots,
#                             basis = basis,
#                             x.min = X.min,
#                             x.max = X.max)
# 
# Psi_X_trim.tt <- prodspline(x = X.t,
#                             xeval = X_trim.t,
#                             K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
#                             knots = knots,
#                             basis = basis,
#                             x.min = X.min,
#                             x.max = X.max)
# 
# B      <- bdiag(Psi_X_trim.tt, Psi_X_trim.cc)
# B      <- as.matrix(B)
# BBinvB <- ginv(t(B) %*% B) %*%  t(B)
# term_u <- c(Y_trim.t, Y_trim.c) - B %*% rbind(beta.t, beta.c)
# Patty  <- t(t(BBinvB) * as.numeric(term_u)) %*% (t(BBinvB) * as.numeric(term_u))

B      <- bdiag(Psi_X.t, Psi_X.c)
B      <- as.matrix(B)
BBinvB <- ginv(t(B) %*% B) %*%  t(B)
term_u <- c(Y.t, Y.c) - B %*% rbind(beta.t, beta.c)
Patty  <- t(t(BBinvB) * as.numeric(term_u)) %*% (t(BBinvB) * as.numeric(term_u))

asy.var  <- t(Bun) %*% Patty %*% Bun
sd_hat <- sqrt(asy.var)


################################# Done! ######################################## 
report <- data.frame(
  V_hat   = V_hat,
  SE      = sd_hat,
  CI_low  = V_hat - 1.96 * sd_hat,
  CI_high = V_hat + 1.96 * sd_hat,
  eps     = eps,
  num     = num
)

print(report)

