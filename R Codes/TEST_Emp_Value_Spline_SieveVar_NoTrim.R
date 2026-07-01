rm(list = ls())

source("spline.R")
library(npiv)
library(mgcv)
library(MASS)

################################# Configurations ############################### 
cost_values <- c(FALSE, TRUE)
reports_df  <- data.frame()
for (cost in cost_values) {

smoothing   <- 3                            

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
# J.x.segments.c   <- chooseJ_result.c$J.x.seg
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
# J.x.segments.t  <- chooseJ_result.t$J.x.seg

# Estimating mu_0(x , 0) and evaluated it on X_trim
Psi_X.cc      <- prodspline(x = X.c,
                           K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                           knots = knots,
                           basis = basis,
                           x.min = X.min,
                           x.max = X.max)
beta.c       <- ginv(t(Psi_X.cc) %*% Psi_X.cc) %*% t(Psi_X.cc) %*% Y.c
d_sieve.c    <- ncol(Psi_X.cc)

Psi_X.c      <- prodspline(x = X.c,
                           xeval = X,
                           K = cbind(rep(J.x.degree,NCOL(X.c)),rep(J.x.segments.c,NCOL(X.c))),
                           knots = knots,
                           basis = basis,
                           x.min = X.min,
                           x.max = X.max)
mu0_X.c      <- Psi_X.c %*% beta.c

# Estimating mu_0(x , 1) and evaluated it on X_trim
Psi_X.tt      <- prodspline(x = X.t,
                           K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                           knots = knots,
                           basis = basis,
                           x.min = X.min,
                           x.max = X.max)
beta.t       <- ginv(t(Psi_X.tt) %*% Psi_X.tt) %*% t(Psi_X.tt) %*% Y.t
d_sieve.t    <- ncol(Psi_X.tt)

Psi_X.t      <- prodspline(x = X.t,
                           xeval = X,
                           K = cbind(rep(J.x.degree,NCOL(X.t)),rep(J.x.segments.t,NCOL(X.t))),
                           knots = knots,
                           basis = basis,
                           x.min = X.min,
                           x.max = X.max)
mu0_X.t      <- Psi_X.t %*% beta.t

# Value functional estimate, where v_0(x) = 1
V_hat        <- mean((mu0_X.t - mu0_X.c > 0)) 

###################### Distribution of h(x) over X_trim ######################## 
h_X      <- Psi_X.t %*% beta.t - Psi_X.c %*% beta.c

# Eliminated outliers
rng      <- c(q1 <- quantile(h_X, 0.25) - 1.5*IQR(h_X),
              quantile(h_X, 0.75) + 1.5*IQR(h_X))
h_X      <- h_X[h_X >= rng[1] & h_X <= rng[2]]

# Define eps according to the scale of h over X_trim
eps      <- eta*sd(h_X)

############################### Density Estimation  ############################  
library(ks)
H        <- Hscv(X)
fhat     <- ks::kde(x = X, H = H*smoothing)  

############################### Visualization Tools ############################ 
library(rgl)

# build evaluation grid
nx  <- 50
gx1 <- seq(min(X[,1]), max(X[,1]), length.out = nx)
gx2 <- seq(min(X[,2]), max(X[,2]), length.out = nx)
G   <- as.matrix(expand.grid(gx1, gx2))
z   <- matrix(predict(fhat, x = G), nrow = nx, byrow = FALSE)

# interactive surface
persp3d(gx1, gx2, z,
        col = "lightblue", alpha = 0.7,
        xlab = "X1", ylab = "X2", zlab = "Density")

################################ Inference ##################################### 
set.seed(123)

# Draw Sobol points from the compact support
library(qrng)
M            <- 1000000
X_samp       <- matrix(NA, nrow = M, ncol = 2)
sobol_points <- sobol(M, d = 2, randomize = "none")
int_lower    <- c(min(X[, 1]), min(X[, 2]))
int_upper    <- c(max(X[, 1]), max(X[, 2]))
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

fx_scaled   <- predict(fhat, x = X_samp) *       
               (max(X[, 1]) - min(X[, 1])) * (max(X[, 2]) - min(X[, 2]))    # Rescale based on Randon-Nikodym

term_1      <- matrix(rep(ind_good, d_sieve.t + d_sieve.c), ncol = d_sieve.t + d_sieve.c)
term_2      <- cbind(Psi_X_samp.t, -Psi_X_samp.c)
term_3      <- matrix(rep(fx_scaled,  d_sieve.t + d_sieve.c), ncol = d_sieve.t + d_sieve.c)

Bun         <- 1/(2*eps)  * colMeans(term_1 * term_2 * term_3)


# OLS formula
library(Matrix)
nt <- nrow(Psi_X.tt); pt <- ncol(Psi_X.tt)
nc <- nrow(Psi_X.cc); pc <- ncol(Psi_X.cc)
B <- matrix(0, nrow = nt + nc, ncol = pt + pc)
B[1:nt, 1:pt] <- Psi_X.tt
B[(nt+1):(nt+nc), (pt+1):(pt+pc)] <- Psi_X.cc
B      <- as.matrix(B)
BBinvB <- ginv(t(B) %*% B) %*%  t(B)
term_u <- c(Y.t, Y.c) - B %*% rbind(beta.t, beta.c)
Patty  <- t(t(BBinvB) * as.numeric(term_u))%*%(t(BBinvB) * as.numeric(term_u))

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
          file = file.path("results", "TEST_Emp_Value_Spline_SieveVar_NoTrim_results.csv"),
          row.names = FALSE)
save(reports_df,
     file = file.path("results", "TEST_Emp_Value_Spline_SieveVar_NoTrim_results.Rdata"))
