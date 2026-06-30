"""Dense-index RF-sieve simulation DGP (CodeWG 260610)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opttreat.models.model_base import ModelBase


class DenseIndexDGP(ModelBase):
    """Dense-index DGP on Uniform([0, 1]^dim) with two centered linear indices.

    Draws ``X ~ Uniform([0, 1]^dim)`` and ``Y = baseline(X) + D * tau(X) + eps``
    with ``D | X ~ Bernoulli(p0(X))``. The treatment effect, baseline, and
    propensity are built from two centered indices ``a1 . X`` and ``a2 . X``
    (alternating-sign and all-ones directions). Requires ``dim >= 2`` because both
    indices are used.
    """

    dgp_name = "dense_index"

    def __init__(
        self,
        dim: int = 50,
        *,
        tau_scale: float = 3.0,
        shift: float = -0.70,
        overlap: float = 1.0,
        heteroskedastic: bool = False,
        noise_sd: float = 1.0,
    ):
        super().__init__(noise_sd=noise_sd)
        if dim < 2:
            raise ValueError("DenseIndexDGP requires dim >= 2 (the indices use a1.X and a2.X).")
        self.dim = int(dim)
        self.tau_scale = float(tau_scale)
        self.shift = float(shift)
        self.overlap = float(overlap)
        self.heteroskedastic = bool(heteroskedastic)

        self._a1 = np.array([(-1.0) ** j for j in range(self.dim)], dtype=float) / np.sqrt(self.dim)
        self._a2 = np.ones(self.dim, dtype=float) / np.sqrt(self.dim)
        self._m1 = 0.5 * self._a1.sum()
        self._m2 = 0.5 * self._a2.sum()

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @property
    def feature_columns(self) -> list[str]:
        return [f"X{j}" for j in range(1, self.dim + 1)]

    def _features(self, x: np.ndarray, *, name: str = "x") -> np.ndarray:
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, self.dim) if X.shape[0] == self.dim else X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"{self.dgp_name}.{name} must be 1D or 2D, got shape {X.shape}.")
        if X.shape[1] != self.dim:
            raise ValueError(f"{self.dgp_name} expects x with {self.dim} columns, got shape {X.shape}.")
        return X

    def _indices(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Two centered linear indices a1.X - m1 and a2.X - m2."""
        return X @ self._a1 - self._m1, X @ self._a2 - self._m2

    def rF0(self, n: int) -> pd.DataFrame:
        X = np.random.uniform(0.0, 1.0, size=(int(n), self.dim))
        return pd.DataFrame(X, columns=self.feature_columns)

    def rF(self, m: int) -> pd.DataFrame:
        return self.rF0(m)

    def inverse_CDF(self, u: np.ndarray) -> np.ndarray:
        return self._features(u, name="u")

    def p0(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        c1, c2 = self._indices(X)
        propensity = self._sigmoid(self.overlap * (1.2 * c1 + 0.4 * c2))
        return self.as_vector(propensity, n=X.shape[0], name="p0(X)")

    def baseline(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        c1, c2 = self._indices(X)
        return self.as_vector(2.0 * np.tanh(2.0 * c2) + np.sin(2.0 * np.pi * c1), n=X.shape[0], name="baseline(X)")

    def h0(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        c1, c2 = self._indices(X)
        tau = self.tau_scale * (self._sigmoid(3.0 * c1) + 0.5 * self._sigmoid(3.0 * c2) + self.shift)
        return self.as_vector(tau, n=X.shape[0], name="h0(X)")

    def mu0(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        X = self._features(x)
        d_vec = self.as_vector(d, n=X.shape[0], name="d")
        return self.baseline(X) + d_vec * self.h0(X)

    def generate_data(self, n: int) -> pd.DataFrame:
        df = self.rF0(n)
        X = df[self.feature_columns].to_numpy()
        d = self._draw_treatment(X).astype(float)
        y_mean = self.mu0(X, d)
        sd = np.where(d == 1.0, 1.25, 0.75) if self.heteroskedastic else self.noise_sd
        df["d"] = d
        df["y"] = y_mean + sd * np.random.normal(size=X.shape[0])
        return df


__all__ = ["DenseIndexDGP"]
