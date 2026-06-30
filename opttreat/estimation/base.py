"""Shared estimator interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import numpy as np


EstimatorOutput = Dict[str, Any]

class Estimator(ABC):
    """
    Base class for first-stage estimators of h(x) = mu_t(x) - mu_c(x).

    Concrete estimators fit separate treated/control nuisance models and expose
    the fitted treatment-effect function plus the matrices needed by variance
    estimators through get_output().
    """

    def __init__(self, name: str, options: Dict[str, Any] | None = None):
        self.name = name
        self.options = options or {}

    @abstractmethod
    def fit(self, data: Any) -> Any:
        pass

    def get_output(self) -> EstimatorOutput:
        """Return fitted objects in the common variance/parameter format."""
        required_attrs = [
            "h_hat",
            "Psi_t_",
            "Psi_c_",
            "e_t_",
            "e_c_",
            "feature_map_t",
            "feature_map_c",
            "X_all_",
            "alpha_",
        ]
        missing = [name for name in required_attrs if getattr(self, name, None) is None]
        if missing:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted or is missing: {missing}."
            )

        h_hat: Callable[[np.ndarray], np.ndarray] = self.h_hat  # type: ignore[assignment]
        return {
            "h_hat": h_hat,
            "mu_hat_t": getattr(self, "mu_hat_t", None),
            "mu_hat_c": getattr(self, "mu_hat_c", None),
            "Psi_t": self.Psi_t_,
            "Psi_c": self.Psi_c_,
            "e_t": self.e_t_,
            "e_c": self.e_c_,
            "beta_t": getattr(self, "beta_t_", None),
            "beta_c": getattr(self, "beta_c_", None),
            "feature_map_t": self.feature_map_t,
            "feature_map_c": self.feature_map_c,
            "X_all": self.X_all_,
            "alpha": self.alpha_,
            "solver": getattr(self, "solver_", None),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, options={self.options})"
