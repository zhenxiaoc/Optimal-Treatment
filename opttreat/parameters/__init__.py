# parameters/__init__.py

from __future__ import annotations

from typing import Any

from .base import Parameter
from .welfare import WelfareKnownDist, WelfareUnknownDist
from .value import ValueKnownDist, ValueUnknownDist

from opttreat.config import ParameterConfig

__all__ = [
    "Parameter",
    "WelfareKnownDist",
    "WelfareUnknownDist",
    "ValueKnownDist",
    "ValueUnknownDist",
    "get_parameter",
]


def get_parameter(cfg: ParameterConfig) -> Parameter:
    """
    Factory that turns a ParameterConfig into a concrete Parameter object.
    """
    pt = cfg.param_type

    if pt == "welfare_known":
        return WelfareKnownDist(
            name="welfare_known",
            options=cfg.options,
        )

    if pt == "welfare_unknown":
        return WelfareUnknownDist(
            name="welfare_unknown",
            options=cfg.options,
        )

    if pt == "value_known":
        return ValueKnownDist(
            name="value_known",
            options=cfg.options,
        )

    if pt == "value_unknown":
        return ValueUnknownDist(
            name="value_unknown",
            options=cfg.options,
        )

    raise ValueError(f"Unknown param_type: {pt!r}")
