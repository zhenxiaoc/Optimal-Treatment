"""Simulation data-generating processes used by OptTreat."""

from __future__ import annotations

from .ccg_model import CCGModel
from .dense_index_model import DenseIndexDGP
from .model_base import ModelBase
from .taylor_expansion_model import TaylorExpansionModel


def _ccg_factory(model_name: str):
    """Return a small factory function for one CCG model."""

    def factory(**kwargs):
        return CCGModel(model_name, **kwargs)

    factory.__name__ = model_name
    factory.__qualname__ = model_name
    factory.__doc__ = f"Return CCGModel('{model_name}')."
    return factory


Model1 = _ccg_factory("Model1")
Model2 = _ccg_factory("Model2")
Model3 = _ccg_factory("Model3")
Model4 = _ccg_factory("Model4")
Model5 = _ccg_factory("Model5")
Model6 = _ccg_factory("Model6")
Model7 = _ccg_factory("Model7")
Model8 = _ccg_factory("Model8")
Model9 = _ccg_factory("Model9")
Model10 = _ccg_factory("Model10")
Model11 = _ccg_factory("Model11")
Model12 = _ccg_factory("Model12")
Model13 = _ccg_factory("Model13")
Model14 = _ccg_factory("Model14")
Model15 = _ccg_factory("Model15")


MODEL_REGISTRY = {
    "Model1": Model1,
    "Model2": Model2,
    "Model3": Model3,
    "Model4": Model4,
    "Model5": Model5,
    "Model6": Model6,
    "Model7": Model7,
    "Model8": Model8,
    "Model9": Model9,
    "Model10": Model10,
    "Model11": Model11,
    "Model12": Model12,
    "Model13": Model13,
    "Model14": Model14,
    "Model15": Model15,
    "TaylorExpansionModel": TaylorExpansionModel,
}


def get_model(model_name: str, **kwargs):
    """Instantiate a registered model by class name."""
    try:
        model_factory = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {model_name!r}. Known models: {sorted(MODEL_REGISTRY)}"
        ) from exc
    return model_factory(**kwargs)


__all__ = [
    "CCGModel",
    "DenseIndexDGP",
    "ModelBase",
    "Model1",
    "Model2",
    "Model3",
    "Model4",
    "Model5",
    "Model6",
    "Model7",
    "Model8",
    "Model9",
    "Model10",
    "Model11",
    "Model12",
    "Model13",
    "Model14",
    "Model15",
    "TaylorExpansionModel",
    "MODEL_REGISTRY",
    "get_model",
]
