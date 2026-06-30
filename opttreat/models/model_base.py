"""Base class and helpers for simulation data-generating processes."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from opttreat.data import ensure_2d_features, ensure_vector


class ModelBase(ABC):
    """
    Base class for OptTreat simulation models.

    Models produce observed data from a full support rF0, integration points
    from target support rF, a propensity p0, an outcome mean mu0, and a true
    treatment effect h0.
    """

    def __init__(self, noise_sd: float = 1.0):
        self.noise_sd = noise_sd

    @staticmethod
    def as_features(x: np.ndarray, *, name: str = "x") -> np.ndarray:
        return ensure_2d_features(x, name=name)

    @staticmethod
    def as_vector(values: np.ndarray, n: int | None = None, *, name: str = "values") -> np.ndarray:
        return ensure_vector(values, n=n, name=name)

    @staticmethod
    def constant(n: int, value: float) -> np.ndarray:
        return np.full(n, value, dtype=float)

    def _draw_treatment(self, X: np.ndarray) -> np.ndarray:
        p = ensure_vector(self.p0(X), n=X.shape[0], name="p0(X)")
        return np.random.binomial(1, p, size=X.shape[0])

    def _add_outcome(self, df: pd.DataFrame, X: np.ndarray, d: np.ndarray) -> pd.DataFrame:
        y_mean = ensure_vector(self.mu0(X, d), n=X.shape[0], name="mu0(X,d)")
        df["d"] = d
        df["y"] = y_mean + np.random.normal(0, self.noise_sd, size=X.shape[0])
        return df

    @abstractmethod
    def rF0(self, n: int) -> pd.DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def rF(self, m: int) -> pd.DataFrame:
        raise NotImplementedError

    # def lambda_fn(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    #     """Treatment effect λ(x1, x2)."""
    #     return (1.4 ** 2) * (x1 > 0) * (x1 < 1) * (x2 > 0) * (x2 < 1)

    @abstractmethod
    def p0(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def mu0(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def h0(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def inverse_CDF(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def generate_data(self, n: int) -> pd.DataFrame:
        raise NotImplementedError

# Backward-compatible alias for any external code that imported the old name.
model_base = ModelBase
