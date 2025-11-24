import numpy as np
from scipy.stats.qmc import Sobol
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import RBFSampler
from doubleml.datasets import make_heterogeneous_data
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator

np.random.seed(123)

# ---------------------------
# SIMULATION PARAMETERS
# ---------------------------
rep = 500
M = 1024
p = 5
n = 1500
support_size = 5
gamma = 1.0
alpha = 1e-5
n_components = 100

# ---------------------------
# TRUE CATE FUNCTION
# ---------------------------
def true_tau(x):
    return np.exp(2 * x[:, 0]) + 3 * np.sin(4 * x[:, 1])

# Sobol grid
sobol_points = Sobol(d=p, scramble=False).random(M)
tau_sobol = true_tau(sobol_points)
W_true = np.mean(np.maximum(0, tau_sobol))

# ---------------------------
# ONE ITERATION
# ---------------------------
def compute_optimal_welfare(n_obs=n, p=p, support_size=support_size,
                            gamma=gamma, alpha=alpha,
                            sobol_points=sobol_points,
                            n_components=n_components):
    
    data_dict = make_heterogeneous_data(
        n_obs=n_obs,
        p=p,
        support_size=support_size,
        n_x=1,
        binary_treatment=True,
    )
    data = data_dict["data"]

    # -------------------- split --------------------
    df_t = data[data["d"] == 1]
    df_c = data[data["d"] == 0]

    Y_t = df_t["y"].to_numpy().ravel()
    X_t = df_t.filter(like="X").to_numpy()

    Y_c = df_c["y"].to_numpy().ravel()
    X_c = df_c.filter(like="X").to_numpy()

    # ----------------- RFF -------------------------
    rff = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)

    B_all = rff.fit_transform(np.vstack([X_t, X_c]))
    B_t = B_all[: len(X_t)]
    B_c = B_all[len(X_t) :]

    # ----------------- linear ridge = approx KRR ---
    krr_t = KernelRidge(alpha=alpha, kernel="linear").fit(B_t, Y_t)
    krr_c = KernelRidge(alpha=alpha, kernel="linear").fit(B_c, Y_c)

    e_t = Y_t - krr_t.predict(B_t)
    e_c = Y_c - krr_c.predict(B_c)     

    # ---------------- Sobol evaluation ---------------
    B_int = rff.transform(sobol_points)
    h_int = krr_t.predict(B_int) - krr_c.predict(B_int)

    W_hat = np.mean(np.maximum(0, h_int))
    
    # -------------- Asymptotic Variance --------------
    ind_good = (h_int >= 0)
    bases    = np.hstack([B_int, -B_int])
    Bun      = bases[ind_good, :].sum(axis=0) / len(bases)
    Bun_t    = Bun[:B_int.shape[1]]
    Bun_c    = Bun[B_int.shape[1]:]
    
    R_t = np.linalg.inv(B_t.T @ B_t + alpha * np.eye(B_t.shape[1])) # R matrix: (B'B + λI)^(-1)
    R_c = np.linalg.inv(B_c.T @ B_c + alpha * np.eye(B_c.shape[1])) 

    Patty_t = R_t @ (B_t.T @  np.diag(e_t**2) @ B_t) @ R_t
    Patty_c = R_c @ (B_c.T @  np.diag(e_c**2) @ B_c) @ R_c
    
    asy_var_t = Bun_t.T @ Patty_t @ Bun_t
    asy_var_c = Bun_c.T @ Patty_c @ Bun_c
    
    asy_var = float(asy_var_t + asy_var_c)
    se      = np.sqrt(asy_var)
    
    return W_hat, se

# ---------------------------
# MONTE CARLO LOOP
# ---------------------------
results = np.zeros((rep, 2))

for r in range(rep):
    results[r, :] = compute_optimal_welfare()
    if (r+1) % 50 == 0:
        print("Completed:", r+1)

# Extract columns
W_hats = results[:, 0]
SE_hats = results[:, 1]

bias        = W_hats.mean() - W_true
sampling_sd = W_hats.std(ddof=1)
avg_se      = SE_hats.mean()
lower       = W_hats - 1.96 * SE_hats
upper       = W_hats + 1.96 * SE_hats
coverage    = np.mean((lower <= W_true) & (W_true <= upper))

print("\n===== Monte Carlo Results =====")
print("True optimal welfare =", W_true)
print(f"Estimator Mean (W_hat)      = {W_hats.mean():.6f}")
print(f"True Value (W_true)         = {W_true:.6f}")
print(f"Bias (mean - W_true)        = {bias:.6f}")
print(f"Sampling SD                 = {sampling_sd:.6f}")
print(f"Average Estimated SE        = {avg_se:.6f}")
print(f"Coverage (95% CI)           = {coverage:.4f}")
