from .base import VarianceEstimator
from .sieve_var import SieveVariance
from .welfare_plugin_var import WelfarePlugInVariance
from opttreat.config import VarianceConfig

__all__ = [
    "VarianceEstimator",
    "SieveVariance",
    "WelfarePlugInVariance",
    "get_variance_estimator",
]


def get_variance_estimator(cfg: VarianceConfig | None) -> VarianceEstimator | None:
    """
    Factory that turns a VarianceConfig into a concrete VarianceEstimator.

    If cfg is None, returns None (no variance estimator).
    """
    if cfg is None:
        return None

    method = cfg.method.lower()

    if method in ("sieve", "sieve_var"):
        # SieveVariance.__init__ only takes `options`
        return SieveVariance(options=cfg.options)

    if method in ("welfare_plugin", "welfare_plugin_var", "plugin_welfare"):
        return WelfarePlugInVariance(options=cfg.options)

    raise ValueError(f"Unknown variance method: {cfg.method!r}")
