import torch

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from .base import Augmentation


class AugmentationPipeline(metaclass=ABCMeta):
    _augmentations: List[Augmentation]

    def __init__(self, augmentations: List[Augmentation]):
        self._augmentations = augmentations

    def append(self, augmentation: Augmentation) -> None:
        """Append augmentation to pipeline.

        Args:
            augmentation (d3rlpy.augmentation.base.Augmentation): augmentation.

        """
        self._augmentations.append(augmentation)

    def get_augmentation_types(self) -> List[str]:
        """Returns augmentation types.

        Returns:
            list(str): list of augmentation types.

        """
        return [aug.get_type() for aug in self._augmentations]

    def get_augmentation_params(self) -> List[Dict[str, Any]]:
        """Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            list(dict): list of augmentation parameters.

        """
        return [aug.get_params() for aug in self._augmentations]

    @abstractmethod
    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns pipeline parameters.

        Returns:
            dict: piple parameters.

        """
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns observation processed by all augmentations.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        if not self._augmentations:
            return x

        for augmentation in self._augmentations:
            x = augmentation.transform(x)

        return x

    @abstractmethod
    def process(
        self,
        func: Callable[..., torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        targets: List[str],
    ) -> torch.Tensor:
        """Runs a given function while augmenting inputs.

        Args:
            func (callable): function to compute.
            inputs (dict): inputs to the func.
            target (list(str)): list of argument names to augment.

        Returns:
            torch.Tensor: the computation result.

        """
        pass

    @property
    def augmentations(self) -> List[Augmentation]:
        return self._augmentations


class DrQPipeline(AugmentationPipeline):
    """Data-reguralized Q augmentation pipeline.

    References:
        * `Kostrikov et al., Image Augmentation Is All You Need: Regularizing
          Deep Reinforcement Learning from Pixels.
          <https://arxiv.org/abs/2004.13649>`_

    Args:
        augmentations (list(d3rlpy.augmentation.base.Augmentation or str)):
            list of augmentations or augmentation types.
        n_mean (int): the number of computations to average

    Attributes:
        augmentations (list(d3rlpy.augmentation.base.Augmentation)):
            list of augmentations.
        n_mean (int): the number of computations to average

    """

    _n_mean: int

    def __init__(
        self,
        augmentations: Optional[List[Augmentation]] = None,
        n_mean: int = 1,
    ):
        if augmentations is None:
            augmentations = []
        super().__init__(augmentations)
        self._n_mean = n_mean

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"n_mean": self._n_mean}

    def process(
        self,
        func: Callable[..., torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        targets: List[str],
    ) -> torch.Tensor:
        device = list(inputs.values())[0].device
        shape = list(inputs.values())[0].shape
        ret = torch.zeros(shape, dtype=torch.float32, device=device)
        for _ in range(self._n_mean):
            kwargs = dict(inputs)
            for target in targets:
                kwargs[target] = self.transform(kwargs[target])
            ret += func(**kwargs)
        return ret / self._n_mean
