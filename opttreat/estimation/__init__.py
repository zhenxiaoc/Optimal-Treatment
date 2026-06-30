"""Estimator factory and first-stage estimator classes."""

from .base import Estimator, EstimatorOutput
from .sieve import SieveEstimator
from .rf_ridge import RFRidgeEstimator

from opttreat.config import EstimatorConfig

__all__ = [
    "Estimator",
    "EstimatorOutput",
    "SieveEstimator",
    "RFRidgeEstimator",
    "get_estimator",
]


def get_estimator(cfg: EstimatorConfig) -> Estimator:
    """
    Factory that turns an EstimatorConfig into a concrete Estimator.

    cfg.method is a short string like:
        - 'sieve'
        - 'rf_ridge'
    """
    method = cfg.method.lower()

    if method == "sieve":
        return SieveEstimator(
            name="sieve",
            options=cfg.options,
        )

    if method in ("rf_ridge", "rf-ridge"):
        return RFRidgeEstimator(
            name="rf_ridge",
            options=cfg.options,
        )

    raise ValueError(f"Unknown Estimator method: {cfg.method!r}")
