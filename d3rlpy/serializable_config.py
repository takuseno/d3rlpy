import dataclasses
from typing import Any, Dict, Type, TypeVar

from dataclasses_json import dataclass_json

TConfig = TypeVar("TConfig", bound="SerializableConfig")


@dataclass_json
@dataclasses.dataclass(frozen=True)
class SerializableConfig:
    def serialize(self) -> str:
        return self.to_json()  # type: ignore

    def serialize_to_dict(self) -> Dict[str, Any]:
        return self.to_dict()  # type: ignore

    @classmethod
    def deserialize(cls: Type[TConfig], serialized_config: str) -> TConfig:
        return cls.from_json(serialized_config)  # type: ignore

    @classmethod
    def deserialize_from_dict(
        cls: Type[TConfig], dict_config: Dict[str, Any]
    ) -> TConfig:
        return cls.from_dict(dict_config)  # type: ignore

    @classmethod
    def deserialize_from_file(cls: Type[TConfig], path: str) -> TConfig:
        with open(path, "r") as f:
            return cls.deserialize(f.read())
