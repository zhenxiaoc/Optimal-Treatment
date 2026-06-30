"""Chen, Chen, and Gao (2025) DGPs as one configurable model class."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from opttreat.models.model_base import ModelBase


class CCGModel(ModelBase):
    """CCG 2025 model selected by name from a class-level definition table."""

    DEFINITIONS = {
        "Model1": {
            "dim": 1,
            "observed_low": (-0.2,),
            "observed_high": (1.2,),
            "target_low": (0.0,),
            "target_high": (1.0,),
            "propensity": lambda X: expit(1.0 - 2.0 * X[:, 0]),
            "baseline": lambda X: 5.0 * np.sin(2.0 * np.pi * X[:, 0]) * np.cos(2.0 * np.pi * X[:, 0]),
            "effect": lambda X: -0.4 + 2.0 * X[:, 0] ** 2,
        },
        "Model2": {
            "dim": 1,
            "observed_low": (-0.2,),
            "observed_high": (1.2,),
            "target_low": (0.0,),
            "target_high": (1.0,),
            "propensity": lambda X: expit(-0.5 + X[:, 0]),
            "baseline": lambda X: 0.5 * np.abs(X[:, 0]),
            "effect": lambda X: 0.5 - X[:, 0] ** 2,
        },
        "Model3": {
            "dim": 1,
            "observed_low": (-0.2,),
            "observed_high": (1.2,),
            "target_low": (0.0,),
            "target_high": (1.0,),
            "propensity": lambda X: expit(0.5 - X[:, 0]),
            "baseline": lambda X: X[:, 0] ** 2,
            "effect": lambda X: 1.0 - X[:, 0],
        },
        "Model4": {
            "dim": 2,
            "observed_low": (-0.2, -0.2),
            "observed_high": (1.2, 1.2),
            "target_low": (0.0, 0.0),
            "target_high": (1.0, 1.0),
            "propensity": lambda X: expit(X[:, 0] - X[:, 1]),
            "baseline": lambda X: (1.0 - X[:, 0] ** 2 - X[:, 1] ** 2)
            * (4.0 + np.sin(X[:, 0]) * X[:, 1] + np.cos(X[:, 1])),
            "effect": lambda X: 0.5 * X[:, 0] - 0.4 * X[:, 1],
        },
        "Model5": {
            "dim": 2,
            "observed_low": (-0.2, -0.2),
            "observed_high": (1.2, 1.2),
            "target_low": (0.0, 0.0),
            "target_high": (1.0, 1.0),
            "propensity": lambda X: expit(X[:, 0] - X[:, 1]),
            "baseline": lambda X: (1.0 - X[:, 0] * X[:, 1])
            * (3.0 + np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])),
            "effect": lambda X: 0.3 * X[:, 0] - 0.3 * X[:, 1],
        },
        "Model6": {
            "dim": 2,
            "observed_low": (-0.2, -0.2),
            "observed_high": (1.2, 1.2),
            "target_low": (0.0, 0.0),
            "target_high": (1.0, 1.0),
            "propensity": lambda X: expit(1.5 * X[:, 0] - 0.5 * X[:, 1]),
            "baseline": lambda X: np.log(1.0 + X[:, 0] + X[:, 1]),
            "effect": lambda X: X[:, 0] - 0.7 * X[:, 1],
        },
        "Model7": {
            "dim": 2,
            "observed_low": (-0.2, -0.2),
            "observed_high": (1.2, 1.2),
            "target_low": (0.0, 0.0),
            "target_high": (1.0, 1.0),
            "propensity": lambda X: expit(-0.5 + X[:, 0] + 2.0 * X[:, 1]),
            "baseline": lambda X: (X[:, 0] ** 2 + X[:, 1] ** 2) * np.exp(-X[:, 0] - X[:, 1]),
            "effect": lambda X: 0.5 - X[:, 1],
        },
        "Model8": {
            "dim": 1,
            "observed_low": (0.0,),
            "observed_high": (1.0,),
            "target_low": (0.0,),
            "target_high": (1.0,),
            "propensity": lambda X: expit(1.0 - 2.0 * X[:, 0]),
            "baseline": lambda X: 5.0 * np.sin(2.0 * np.pi * X[:, 0]) * np.cos(2.0 * np.pi * X[:, 0]),
            "effect": lambda X: -0.4 + 2.0 * X[:, 0] ** 2,
        },
        "Model9": {
            "dim": 1,
            "observed_low": (0.0,),
            "observed_high": (1.0,),
            "target_low": (0.0,),
            "target_high": (1.0,),
            "propensity": lambda X: expit(-0.5 + X[:, 0]),
            "baseline": lambda X: 0.5 * np.abs(X[:, 0]),
            "effect": lambda X: 0.5 - X[:, 0] ** 2,
        },
        "Model10": {
            "dim": 1,
            "observed_low": (0.0,),
            "observed_high": (1.0,),
            "target_low": (0.0,),
            "target_high": (1.0,),
            "propensity": lambda X: expit(0.5 - X[:, 0]),
            "baseline": lambda X: X[:, 0] ** 2,
            "effect": lambda X: 1.0 - X[:, 0],
        },
        "Model11": {
            "dim": 2,
            "observed_low": (0.0, 0.0),
            "observed_high": (1.0, 1.0),
            "target_low": (0.0, 0.0),
            "target_high": (1.0, 1.0),
            "propensity": lambda X: expit(X[:, 0] - X[:, 1]),
            "baseline": lambda X: (1.0 - X[:, 0] ** 2 - X[:, 1] ** 2)
            * (4.0 + np.sin(X[:, 0]) * X[:, 1] + np.cos(X[:, 1])),
            "effect": lambda X: 0.5 * X[:, 0] - 0.4 * X[:, 1],
        },
        "Model12": {
            "dim": 2,
            "observed_low": (0.0, 0.0),
            "observed_high": (1.0, 1.0),
            "target_low": (0.0, 0.0),
            "target_high": (1.0, 1.0),
            "propensity": lambda X: expit(X[:, 0] - X[:, 1]),
            "baseline": lambda X: (1.0 - X[:, 0] * X[:, 1])
            * (3.0 + np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])),
            "effect": lambda X: 0.3 * X[:, 0] - 0.3 * X[:, 1],
        },
        "Model13": {
            "dim": 2,
            "observed_low": (0.0, 0.0),
            "observed_high": (1.0, 1.0),
            "target_low": (0.0, 0.0),
            "target_high": (1.0, 1.0),
            "propensity": lambda X: expit(1.5 * X[:, 0] - 0.5 * X[:, 1]),
            "baseline": lambda X: np.log(1.0 + X[:, 0] + X[:, 1]),
            "effect": lambda X: X[:, 0] - 0.7 * X[:, 1],
        },
        "Model14": {
            "dim": 2,
            "observed_low": (0.0, 0.0),
            "observed_high": (1.0, 1.0),
            "target_low": (0.0, 0.0),
            "target_high": (1.0, 1.0),
            "propensity": lambda X: expit(-0.5 + X[:, 0] + 2.0 * X[:, 1]),
            "baseline": lambda X: (X[:, 0] ** 2 + X[:, 1] ** 2) * np.exp(-X[:, 0] - X[:, 1]),
            "effect": lambda X: 0.5 - X[:, 1],
        },
        "Model15": {
            "dim": 2,
            "observed_low": (-2.0, -2.0),
            "observed_high": (2.0, 2.0),
            "target_low": (-1.5, -1.5),
            "target_high": (1.5, 1.5),
            "propensity": lambda X: expit(X[:, 0] - X[:, 1]),
            "baseline": lambda X: np.zeros(X.shape[0], dtype=float),
            "effect": lambda X: (1.0 - X[:, 0] ** 2 - X[:, 1] ** 2)
            * (4.0 + np.sin(X[:, 0]) * X[:, 1] + np.cos(X[:, 1])),
        },
    }

    def __init__(self, name: str, noise_sd: float = 1.0):
        super().__init__(noise_sd=noise_sd)
        if name not in self.DEFINITIONS:
            raise ValueError(f"Unknown CCG model {name!r}. Known models: {sorted(self.DEFINITIONS)}")
        self.name = name
        self.definition = self.DEFINITIONS[name]

    @property
    def dim(self) -> int:
        return int(self.definition["dim"])

    @property
    def observed_low(self) -> tuple[float, ...]:
        return self.definition["observed_low"]

    @property
    def observed_high(self) -> tuple[float, ...]:
        return self.definition["observed_high"]

    @property
    def target_low(self) -> tuple[float, ...]:
        return self.definition["target_low"]

    @property
    def target_high(self) -> tuple[float, ...]:
        return self.definition["target_high"]

    @property
    def observed_support(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return self.observed_low, self.observed_high

    @property
    def target_support(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return self.target_low, self.target_high

    @property
    def feature_columns(self) -> list[str]:
        return ["X"] if self.dim == 1 else [f"X{j}" for j in range(1, self.dim + 1)]

    def _features(self, x: np.ndarray, *, name: str = "x") -> np.ndarray:
        X = self.as_features(x, name=f"{self.name}.{name}")
        if X.shape[1] != self.dim:
            raise ValueError(f"{self.name} expects x of shape (n, {self.dim}).")
        return X

    def _draw_uniform(self, n: int, low: tuple[float, ...], high: tuple[float, ...]) -> pd.DataFrame:
        lower = np.asarray(low, dtype=float)
        upper = np.asarray(high, dtype=float)
        X = lower + np.random.uniform(size=(n, self.dim)) * (upper - lower)
        return pd.DataFrame(X, columns=self.feature_columns)

    def rF0(self, n: int) -> pd.DataFrame:
        return self._draw_uniform(n, self.observed_low, self.observed_high)

    def rF(self, m: int) -> pd.DataFrame:
        return self._draw_uniform(m, self.target_low, self.target_high)

    def inverse_CDF(self, u: np.ndarray) -> np.ndarray:
        U = self._features(u, name="u")
        lower = np.asarray(self.target_low, dtype=float)
        upper = np.asarray(self.target_high, dtype=float)
        return lower + U * (upper - lower)

    def p0(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        propensity = self.definition["propensity"]
        return self.as_vector(propensity(X), n=X.shape[0], name=f"{self.name}.p0")

    def baseline(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        baseline = self.definition["baseline"]
        return self.as_vector(baseline(X), n=X.shape[0], name=f"{self.name}.baseline")

    def h0(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        effect = self.definition["effect"]
        return self.as_vector(effect(X), n=X.shape[0], name=f"{self.name}.h0")

    def mu0(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        X = self._features(x)
        d_vec = self.as_vector(d, n=X.shape[0], name="d")
        custom_mu0 = self.definition.get("custom_mu0")
        if custom_mu0 is not None:
            return self.as_vector(custom_mu0(X, d_vec), n=X.shape[0], name=f"{self.name}.mu0")
        return self.baseline(X) + d_vec * self.h0(X)

    def generate_data(self, n: int) -> pd.DataFrame:
        df = self.rF0(n)
        X = df[self.feature_columns].to_numpy()
        d = self._draw_treatment(X)
        return self._add_outcome(df, X, d)


__all__ = [
    "CCGModel",
]
