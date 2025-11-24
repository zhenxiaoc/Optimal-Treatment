import numpy as np
import pandas as pd
from scipy.special import expit  # logistic function

class Model4:
    def __init__(self, noise_sd=1.0):
        self.noise_sd = noise_sd

    # ----------------------------------------------
    # rF0: generate initial covariates (super-space)
    # ----------------------------------------------
    def rF0(self, n):
        X1 = np.random.uniform(-0.2, 1.2, size=n)
        X2 = np.random.uniform(-0.2, 1.2, size=n)
        return pd.DataFrame({"X1": X1, "X2": X2})

    # ----------------------------------------------
    # rF: generate covariates supported in [0,1]^2
    # ----------------------------------------------
    def rF(self, m):
        X1 = np.random.beta(1, 1, size=m)   # same as uniform(0,1)
        X2 = np.random.beta(1, 1, size=m)
        return pd.DataFrame({"X1": X1, "X2": X2})

    # ----------------------------------------------
    # lambda(x1, x2): treatment effect
    # ----------------------------------------------
    def lambda_fn(self, x1, x2):
        return (1.4 ** 2) * (x1 > 0) * (x1 < 1) * (x2 > 0) * (x2 < 1)

    # ----------------------------------------------
    # p0(x1, x2): propensity score
    # ----------------------------------------------
    def p0(self, x1, x2):
        return expit(x1 - x2)

    # ----------------------------------------------
    # mu0(x1, x2, d): outcome regression
    # ----------------------------------------------
    def mu0(self, x1, x2, d):
        return (1 - x1**2 - x2**2)*(4 + np.sin(x1)*x2 - np.cos(x2)) + d*(x1*0.5 - x2*0.4)
    
    # ----------------------------------------------
    # generate a dataset of size n
    # ----------------------------------------------
    def generate_data(self, n):
        df = self.rF0(n)
        x1 = df["X1"].values
        x2 = df["X2"].values

        # Propensity score
        p = self.p0(x1, x2)

        # Treatment assignment
        d = np.random.binomial(1, p, size=n)

        # Outcome
        y = self.mu0(x1, x2, d) + np.random.normal(0, self.noise_sd, size=n)

        df["d"] = d
        df["y"] = y
        df["tau"] = self.lambda_fn(x1, x2)

        return df

