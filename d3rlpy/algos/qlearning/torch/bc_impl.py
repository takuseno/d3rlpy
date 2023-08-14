from abc import ABCMeta, abstractmethod
from typing import Union

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    CategoricalPolicy,
    DeterministicPolicy,
    NormalPolicy,
    Policy,
    compute_deterministic_imitation_loss,
    compute_discrete_imitation_loss,
    compute_stochastic_imitation_loss,
)
from ....torch_utility import Checkpointer, TorchMiniBatch, train_api
from ..base import QLearningAlgoImplBase

__all__ = ["BCImpl", "DiscreteBCImpl"]


class BCBaseImpl(QLearningAlgoImplBase, metaclass=ABCMeta):
    _learning_rate: float
    _optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        optim: Optimizer,
        checkpointer: Checkpointer,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            checkpointer=checkpointer,
            device=device,
        )
        self._optim = optim

    @train_api
    def update_imitator(self, batch: TorchMiniBatch) -> float:
        self._optim.zero_grad()

        loss = self.compute_loss(batch.observations, batch.actions)

        loss.backward()
        self._optim.step()

        return float(loss.cpu().detach().numpy())

    @abstractmethod
    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        pass

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("BC does not support value estimation")


class BCImpl(BCBaseImpl):
    _imitator: Union[DeterministicPolicy, NormalPolicy]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        imitator: Union[DeterministicPolicy, NormalPolicy],
        optim: Optimizer,
        checkpointer: Checkpointer,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            optim=optim,
            checkpointer=checkpointer,
            device=device,
        )
        self._imitator = imitator

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._imitator(x).squashed_mu

    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self._imitator, DeterministicPolicy):
            return compute_deterministic_imitation_loss(
                self._imitator, obs_t, act_t
            )
        else:
            return compute_stochastic_imitation_loss(
                self._imitator, obs_t, act_t
            )

    @property
    def policy(self) -> Policy:
        return self._imitator

    @property
    def policy_optim(self) -> Optimizer:
        return self._optim


class DiscreteBCImpl(BCBaseImpl):
    _beta: float
    _imitator: CategoricalPolicy

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        imitator: CategoricalPolicy,
        optim: Optimizer,
        beta: float,
        checkpointer: Checkpointer,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            optim=optim,
            checkpointer=checkpointer,
            device=device,
        )
        self._imitator = imitator
        self._beta = beta

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._imitator(x).logits.argmax(dim=1)

    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        return compute_discrete_imitation_loss(
            policy=self._imitator, x=obs_t, action=act_t.long(), beta=self._beta
        )
