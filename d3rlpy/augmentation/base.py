from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict

import torch


class Augmentation(metaclass=ABCMeta):

    TYPE: ClassVar[str] = "none"

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns augmented observation.

        Args:
            x: observation.

        Returns:
            augmented observation.

        """

    def get_type(self) -> str:
        """Returns augmentation type.

        Returns:
            augmentation type.

        """
        return self.TYPE

    @abstractmethod
    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns augmentation parameters.

        Args:
            deep: flag to copy parameters.

        Returns:
            augmentation parameters.

        """
