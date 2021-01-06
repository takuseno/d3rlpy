from typing import Any, ClassVar, Dict

import torch

from .base import Augmentation


class SingleAmplitudeScaling(Augmentation):
    r"""Single Amplitude Scaling augmentation.

    .. math::

        x' = x + z

    where :math:`z \sim \text{Unif}(minimum, maximum)`.

    References:
        * `Laskin et al., Reinforcement Learning with Augmented Data.
          <https://arxiv.org/abs/2004.14990>`_

    Args:
        minimum (float): minimum amplitude scale.
        maximum (float): maximum amplitude scale.

    """

    TYPE: ClassVar[str] = "single_amplitude_scaling"
    _minimum: float
    _maximum: float

    def __init__(self, minimum: float = 0.8, maximum: float = 1.2):
        self._minimum = minimum
        self._maximum = maximum

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scaled observation.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        z = torch.empty(x.shape[0], 1, device=x.device)
        z.uniform_(self._minimum, self._maximum)
        return x * z

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: augmentation parameters.

        """
        return {"minimum": self._minimum, "maximum": self._maximum}


class MultipleAmplitudeScaling(SingleAmplitudeScaling):
    r"""Multiple Amplitude Scaling augmentation.

    .. math::

        x' = x + z

    where :math:`z \sim \text{Unif}(minimum, maximum)` and :math:`z`
    is a vector with different amplitude scale on each.

    References:
        * `Laskin et al., Reinforcement Learning with Augmented Data.
          <https://arxiv.org/abs/2004.14990>`_

    Args:
        minimum (float): minimum amplitude scale.
        maximum (float): maximum amplitude scale.

    """

    TYPE: ClassVar[str] = "multiple_amplitude_scaling"

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.empty(*x.shape, device=x.device)
        z.uniform_(self._minimum, self._maximum)
        return x * z
