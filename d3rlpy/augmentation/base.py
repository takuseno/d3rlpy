import torch

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict


class Augmentation(metaclass=ABCMeta):

    TYPE: ClassVar[str] = "none"

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_type(self) -> str:
        """Returns augmentation type.

        Returns:
            str: augmentation type.

        """
        return self.TYPE

    @abstractmethod
    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        pass
