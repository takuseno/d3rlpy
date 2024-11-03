import dataclasses
from typing import Any

import torch

__all__ = ["asdict_without_copy", "asdict_as_float"]


def asdict_without_copy(obj: Any) -> dict[str, Any]:
    assert dataclasses.is_dataclass(obj)
    fields = dataclasses.fields(obj)
    return {field.name: getattr(obj, field.name) for field in fields}


def asdict_as_float(obj: Any) -> dict[str, float]:
    assert dataclasses.is_dataclass(obj)
    fields = dataclasses.fields(obj)
    ret: dict[str, float] = {}
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
