import dataclasses
from typing import Any, Dict

__all__ = ["asdict_without_copy"]


def asdict_without_copy(obj: Any) -> Dict[str, Any]:
    assert dataclasses.is_dataclass(obj)
    fields = dataclasses.fields(obj)
    return {field.name: getattr(obj, field.name) for field in fields}
