import dataclasses
from typing import Any, Dict

import torch

__all__ = ["asdict_without_copy", "asdict_as_float"]


def asdict_without_copy(obj: Any) -> Dict[str, Any]:
    assert dataclasses.is_dataclass(obj)
    fields = dataclasses.fields(obj)
    return {field.name: getattr(obj, field.name) for field in fields}


def asdict_as_float(obj: Any) -> Dict[str, float]:
    assert dataclasses.is_dataclass(obj)
    fields = dataclasses.fields(obj)
    ret: Dict[str, float] = {}
    for field in fields:
        value = getattr(obj, field.name)
        if isinstance(value, torch.Tensor):
            assert (
                value.ndim == 0
            ), f"{field.name} needs to be scalar. {value.shape}."
            ret[field.name] = float(value.cpu().detach().numpy())
        else:
            ret[field.name] = float(value)
    return ret
