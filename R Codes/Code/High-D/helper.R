make_heterogeneous_data <- function(n_obs = 200, p = 30, support_size = 5,
                                    n_x = 1, binary_treatment = TRUE) {
  # --- argument checks ---
  assert_count(n_obs, positive = TRUE)
  assert_count(p, positive = TRUE)
  assert_count(support_size, positive = TRUE)
  assert_true(support_size <= p)
  assert_choice(n_x, choices = c(1, 2))
  assert_flag(binary_treatment)
  
  # --- define heterogeneous treatment effect θ₀(x) ---
  treatment_effect <- if (n_x == 1) {
    function(x) exp(2 * x[, 1]) + 3 * sin(4 * x[, 1])
  } else {
    function(x) exp(2 * x[, 1]) + 3 * sin(4 * x[, 2])
  }
  
  # --- supports and coefficients ---
  support_y <- sample.int(p, size = support_size, replace = FALSE)
  coefs_y   <- runif(support_size, min = 0, max = 1)
  support_d <- support_y
  coefs_d   <- runif(support_size, min = 0, max = 0.3)
  
  # --- noise terms ---
  epsilon <- runif(n_obs, min = -1, max = 1)
  eta     <- runif(n_obs, min = -1, max = 1)
  
  # --- covariates ---
  x <- matrix(runif(n_obs * p, min = -0.2, max = 1.2), nrow = n_obs, ncol = p)
  
  # --- heterogeneous treatment effects and treatment ---
  te <- treatment_effect(x)
  linpred <- drop(x[, support_d, drop = FALSE] %*% coefs_d)
  
  if (binary_treatment) {
    d <- as.numeric(linpred >= eta)
  } else {
    d <- linpred + eta
  }
  
  # --- outcome ---
  y <- te * d + drop(x[, support_y, drop = FALSE] %*% coefs_y) + epsilon
  
  # --- assemble and return ---
  data <- data.frame(y = y, d = d, x)
  list(
    data = data,
    effects = te,
    treatment_effect = treatment_effect
  )
}


trim_minmax <- function(df, d = "d", vars = NULL) {
  if (is.null(vars)) vars <- setdiff(names(df), c(d, "y"))  # covariates
  X <- as.matrix(df[, vars, drop = FALSE])
  
  is_t <- df[[d]] == 1
  is_c <- !is_t
  
  # compute lower and upper bounds for each variable
  lower <- pmax(
    sapply(vars, function(v) min(df[[v]][is_t], na.rm = TRUE)),
    sapply(vars, function(v) min(df[[v]][is_c], na.rm = TRUE))
  )
  upper <- pmin(
    sapply(vars, function(v) max(df[[v]][is_t], na.rm = TRUE)),
    sapply(vars, function(v) max(df[[v]][is_c], na.rm = TRUE))
  )
  
  # keep observations within [lower_j, upper_j] for all j
  keep <- rowSums(
    sweep(X, 2, lower, `>=`) & sweep(X, 2, upper, `<=`)
  ) == length(vars)
  
  df_trim <- df[keep, , drop = FALSE]
  list(df_trim = df_trim,
       keep = keep,
       bounds = data.frame(var = vars, lower = lower, upper = upper))
}


sieve_dim <- function(n_obs = 200, p = 30, support_size = 5,
                      n_x = 1, binary_treatment = TRUE){
  
  df_list <- make_heterogeneous_data(n_obs, p, 
                                support_size, 
                                n_x, 
                                binary_treatment)
  
  df <- df_list$data
  # Control + Treated
  Y <- df$y; D <- df$d
  X <- df[ , !(names(df) %in% c("d","y")), drop = FALSE]
  
  # Control
  df.c <- df[df$d == 0, , drop = FALSE]
  Y.c  <- as.numeric(df.c$y)
  X.c  <- df.c[ , !(names(df.c) %in% c("d","y")), drop = FALSE]
  
  # Treated
  df.t <- df[df$d == 1, , drop = FALSE]
  Y.t  <- as.numeric(df.t$y)
  X.t  <- df.t[ , !(names(df.t) %in% c("d","y")), drop = FALSE]
  
  # Trimmed (min-max overlap on all X columns)
  df_trim <- trim_minmax(df, d = "d", vars = NULL)$df_trim
  X_trim  <- df_trim[ , !(names(df_trim) %in% c("d","y")), drop = FALSE]
  
  chooseJ_result <- npiv_choose_J(Y.c,
                                  X.c,
                                  X.c,
                                  X.grid = NULL,
                                  J.x.degree = 3,
                                  K.w.degree = 4,
                                  K.w.smooth = 2,
                                  knots = "uniform",
                                  basis = "tensor",
                                  X.min = NULL,
                                  X.max = NULL,
                                  W.min = NULL,
                                  W.max = NULL,
                                  grid.num = 50,
                                  boot.num = 99,
                                  check.is.fullrank= FALSE,
                                  progress = TRUE)
  
  J.x.segments.c <- chooseJ_result$J.x.seg
  
  chooseJ_result <- npiv_choose_J(Y.t,
                                  X.t,
                                  X.t,
                                  X.grid = NULL,
                                  J.x.degree = 3,
                                  K.w.degree = 4,
                                  K.w.smooth = 2,
                                  knots = "uniform",
                                  basis = "tensor",
                                  X.min = NULL,
                                  X.max = NULL,
                                  W.min = NULL,
                                  W.max = NULL,
                                  grid.num = 50,
                                  boot.num = 99,
                                  check.is.fullrank= FALSE,
                                  progress = TRUE)
  
  J.x.segments.t <- chooseJ_result$J.x.seg
  
  return(c(J.x.segments.c = J.x.segments.c, J.x.segments.t = J.x.segments.t))
}