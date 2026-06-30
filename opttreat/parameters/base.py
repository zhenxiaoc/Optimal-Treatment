# parameters/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict

from opttreat.models.model_base import ModelBase


class Parameter(ABC):
    """
    Abstract base class for a parameter of interest,

    """

    def __init__(self, name: str, options: Dict[str, Any] | None = None):
        self.name = name
        self.options = options or {}

    @abstractmethod
    def plug_in(self, h: Any, *args: Any) -> float:
        pass

    @abstractmethod
    def loo(self, estimator_output: dict[str, Any], *args: Any, **kwargs: Any) -> float:
        pass

    @abstractmethod
    def get_true_value(self, model: ModelBase) -> float:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, options={self.options})"
