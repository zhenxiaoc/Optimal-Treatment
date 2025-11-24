import os
import numpy as np
from scipy.stats.qmc import Sobol
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import RBFSampler
from doubleml.datasets import make_heterogeneous_data
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from scipy.special import expit
from model7 import Model7
from model4 import Model4
from model5 import Model5
from RandomFeature_zx import RandomFeatureGenerator_zx

"""
permitted_activation_functions = [
        "cos",
        "sin",
        "exp",
        "arctan",
        "tanh",
        "ReLu",
        "Elu",
        "SoftPlus",
        "cos_and_sin",
    ]

"""
# ---------------------------
# SIMULATION PARAMETERS
# ---------------------------
rep = 500
M = 1024
p = 2
n = 1500
support_size = 2
gamma = 1.0
alpha = 1e-5
n_components = 100
distribution = "standard_normal"
distribution_parameters = []
activation= "tanh"
bias_distribution = None
bias_distribution_parameters = None


# ---------------------------
# TRUE CATE FUNCTION
# ---------------------------
def true_tau(x):
    return np.exp(2 * x[:, 0]) + 3 * np.sin(4 * x[:, 1])

# Sobol grid
np.random.seed(123)
# sobol_points = Sobol(d=p, scramble=False).random(M)
# tau_sobol = true_tau(sobol_points)
# W_true = np.mean(np.maximum(0, tau_sobol))


sobol_points = Sobol(d=p, scramble=False).random(100000)
model_7 = Model7()
tau_sobol = model_7.mu0(sobol_points[:,0], sobol_points[:,1], d=1) - model_7.mu0(sobol_points[:,0], sobol_points[:,1], d=0)
W_true = np.mean(np.maximum(0, tau_sobol))

# model_4 = Model4()
# tau_sobol = model_4.mu0(sobol_points[:,0], sobol_points[:,1], d=1) - model_4.mu0(sobol_points[:,0], sobol_points[:,1], d=0)
# W_true = np.mean(np.maximum(0, tau_sobol))

# ---------------------------
# ONE ITERATION
# ---------------------------
def compute_optimal_welfare(n_obs=n, p=p, support_size=support_size,
                            gamma=gamma, alpha=alpha,
                            sobol_points=sobol_points,
                            n_components=n_components):
    
    # data_dict = make_heterogeneous_data(
    #     n_obs=n_obs,
    #     p=p,
    #     support_size=support_size,
    #     n_x=1,
    #     binary_treatment=True,
    # )
    # data = data_dict["data"]

    data = model_7.generate_data(n)
    
    #data = model_4.generate_data(n)

    # -------------------- split --------------------
    df_t = data[data["d"] == 1]
    df_c = data[data["d"] == 0]

    Y_t = df_t["y"].to_numpy().ravel()
    X_t = df_t.filter(like="X").to_numpy()

    Y_c = df_c["y"].to_numpy().ravel()
    X_c = df_c.filter(like="X").to_numpy()

    # ------- Renerate_random_neuron_features -----------
    features = np.vstack([X_t, X_c])
    random_seed = np.random.randint(10**9)
    number_features = n_components
    
    # B_all = RandomFeaturesGenerator.generate_random_neuron_features(
    #     features,
    #     random_seed,
    #     distribution,
    #     distribution_parameters,
    #     activation,
    #     number_features,
    # )
    
    B_all = RandomFeatureGenerator_zx.generate_random_neuron_features(
        features,
        random_seed,
        distribution,
        distribution_parameters,
        activation,
        number_features,
    )
    B_t = B_all[: len(X_t)]
    B_c = B_all[len(X_t) :]

    # ----------------- linear ridge = approx KRR ---
    krr_t = KernelRidge(alpha=alpha, kernel="linear").fit(B_t, Y_t)
    krr_c = KernelRidge(alpha=alpha, kernel="linear").fit(B_c, Y_c)

    e_t = Y_t - krr_t.predict(B_t)
    e_c = Y_c - krr_c.predict(B_c)     

    # ---------------- Sobol evaluation ---------------
    # B_int = RandomFeaturesGenerator.generate_random_neuron_features(
    #     sobol_points,
    #     random_seed,
    #     distribution,
    #     distribution_parameters,
    #     activation,
    #     number_features,
    #     bias_distribution,
    #     bias_distribution_parameters,
    # )
    B_int = RandomFeatureGenerator_zx.generate_random_neuron_features(
        sobol_points,
        random_seed,
        distribution,
        distribution_parameters,
        activation,
        number_features,
    )
    h_int = krr_t.predict(B_int) - krr_c.predict(B_int)

    W_hat = np.mean(np.maximum(0, h_int))
    
    # -------------- Asymptotic Variance --------------
    ind_good = (h_int >= 0)
    bases    = np.hstack([B_int, -B_int])
    Bun      = bases[ind_good, :].sum(axis=0) / bases.shape[0]
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
print(f"Bias (mean - W_true)        = {bias:.6f}")
print(f"Sampling SD                 = {sampling_sd:.6f}")
print(f"Average Estimated SE        = {avg_se:.6f}")
print(f"Coverage (95% CI)           = {coverage:.6f}")

