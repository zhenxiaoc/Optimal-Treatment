# variance/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class VarianceEstimator(ABC):
    """
    Abstract base class for variance estimators.

    Design philosophy:
    - A VarianceEstimator takes whatever the first-stage Estimator produced
      (usually a dict with Psi_t, Psi_c, residuals, h_hat, etc.) and returns
      an estimate of the asymptotic variance of the parameter of interest.
    """

    def __init__(self, name: str, options: Dict[str, Any] | None = None):
        """
        Parameters
        ----------
        name : str
            Short label, e.g. 'sieve_var', 'plugin_var'.
        options : dict, optional
            Free-form configuration dictionary, interpreted by subclasses.
        """
        self.name = name
        self.options = options or {}

    @abstractmethod
    def fit(self, estimator_output: Dict[str, Any]) -> float:
        """
        Given the estimator's output, compute the asymptotic variance estimate.

        Parameters
        ----------
        estimator_output : dict
            Typically contains model matrices, residuals, h_hat, etc.

        Returns
        -------
        float
            Estimated asymptotic variance.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, options={self.options})"
