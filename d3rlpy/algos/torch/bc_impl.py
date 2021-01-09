from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer

from ...models.torch import Imitator
from ...models.torch import DiscreteImitator, DeterministicRegressor
from ...models.torch import create_deterministic_regressor
from ...models.torch import create_discrete_imitator
from ...optimizers import OptimizerFactory
from ...encoders import EncoderFactory
from ...gpu import Device
from ...preprocessing import Scaler
from ...augmentation import AugmentationPipeline
from ...torch_utility import torch_api, train_api, augmentation_api
from .base import TorchImplBase


class BCBaseImpl(TorchImplBase, metaclass=ABCMeta):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _use_gpu: Optional[Device]
    _imitator: Optional[Imitator]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(observation_shape, action_size, scaler, augmentation)
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._use_gpu = use_gpu

        # initialized in build
        self._imitator = None
        self._optim = None

    def build(self) -> None:
        self._build_network()

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        self._build_optim()

    @abstractmethod
    def _build_network(self) -> None:
        pass

    def _build_optim(self) -> None:
        assert self._imitator is not None
        self._optim = self._optim_factory.create(
            self._imitator.parameters(), lr=self._learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["obs_t"])
    def update_imitator(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> np.ndarray:
        assert self._optim is not None

        self._optim.zero_grad()

        loss = self.compute_loss(obs_t, act_t)

        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()

    @augmentation_api(targets=["obs_t"])
    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator.compute_error(obs_t, act_t)

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator(x)

    def predict_value(
        self, x: np.ndarray, action: np.ndarray, with_std: bool
    ) -> np.ndarray:
        raise NotImplementedError("BC does not support value estimation")

    def sample_action(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("BC does not support sampling action")


class BCImpl(BCBaseImpl):

    _imitator: Optional[DeterministicRegressor]

    def _build_network(self) -> None:
        self._imitator = create_deterministic_regressor(
            self._observation_shape, self._action_size, self._encoder_factory
        )


class DiscreteBCImpl(BCBaseImpl):

    _beta: float
    _imitator: Optional[DiscreteImitator]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        beta: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            use_gpu=use_gpu,
            scaler=scaler,
            augmentation=augmentation,
        )
        self._beta = beta

    def _build_network(self) -> None:
        self._imitator = create_discrete_imitator(
            self._observation_shape,
            self._action_size,
            self._beta,
            self._encoder_factory,
        )

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator(x).argmax(dim=1)

    @augmentation_api(targets=["obs_t"])
    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator.compute_error(obs_t, act_t.long())
