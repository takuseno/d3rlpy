from typing import Any, ClassVar, Dict

import torch

from ..decorators import pretty_repr


@pretty_repr
class Augmentation:

    TYPE: ClassVar[str] = "none"

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns augmented observation.

        Args:
            x: observation.

        Returns:
            augmented observation.

        """
        raise NotImplementedError

    def get_type(self) -> str:
        """Returns augmentation type.

        Returns:
            augmentation type.

        """
        return self.TYPE

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns augmentation parameters.

        Args:
            deep: flag to copy parameters.

        Returns:
            augmentation parameters.

        """
        raise NotImplementedError
