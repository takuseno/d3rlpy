from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_probablistic_dynamics
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.torch import EnsembleDynamics
from ...preprocessing import ActionScaler, Scaler
from ...torch_utility import torch_api, train_api
from .base import TorchImplBase


class MOPOImpl(TorchImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _n_ensembles: int
    _lam: float
    _discrete_action: bool
    _use_gpu: Optional[Device]
    _dynamics: Optional[EnsembleDynamics]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        n_ensembles: int,
        lam: float,
        discrete_action: bool,
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        use_gpu: Optional[Device],
    ):
        super().__init__(observation_shape, action_size, scaler, action_scaler)
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._n_ensembles = n_ensembles
        self._lam = lam
        self._discrete_action = discrete_action
        self._use_gpu = use_gpu

        # initialized in build
        self._dynamics = None
        self._optim = None

    def build(self) -> None:
        self._build_dynamics()

        self.to_cpu()
        if self._use_gpu:
            self.to_gpu(self._use_gpu)

        self._build_optim()

    def _build_dynamics(self) -> None:
        self._dynamics = create_probablistic_dynamics(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            n_ensembles=self._n_ensembles,
            discrete_action=self._discrete_action,
        )

    def _build_optim(self) -> None:
        assert self._dynamics is not None
        self._optim = self._optim_factory.create(
            self._dynamics.parameters(), lr=self._learning_rate
        )

    def _predict(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._dynamics is not None
        return self._dynamics.predict_with_variance(x, action, "max")

    def _generate(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._dynamics is not None
        observations, rewards, variances = self._dynamics.predict_with_variance(
            x, action, "max"
        )
        return observations, rewards - self._lam * variances

    @train_api
    @torch_api(
        scaler_targets=["obs_t", "obs_tp1"], action_scaler_targets=["act_t"]
    )
    def update(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        obs_tp1: torch.Tensor,
    ) -> np.ndarray:
        assert self._dynamics is not None
        assert self._optim is not None

        loss = self._dynamics.compute_error(obs_t, act_t, rew_tp1, obs_tp1)

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()
