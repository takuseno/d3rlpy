import copy
from abc import ABCMeta, abstractmethod
from typing import Dict

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    EnsembleContinuousQFunction,
    EnsembleQFunction,
    Policy,
)
from ....torch_utility import TorchMiniBatch, soft_sync, train_api
from ..base import QLearningAlgoImplBase
from .utility import ContinuousQFunctionMixin

__all__ = ["DDPGImpl"]


class DDPGBaseImpl(
    ContinuousQFunctionMixin, QLearningAlgoImplBase, metaclass=ABCMeta
):
    _gamma: float
    _tau: float
    _q_func: EnsembleContinuousQFunction
    _policy: Policy
    _targ_q_func: EnsembleContinuousQFunction
    _targ_policy: Policy
    _actor_optim: Optimizer
    _critic_optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: Policy,
        q_func: EnsembleContinuousQFunction,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        gamma: float,
        tau: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
        )
        self._gamma = gamma
        self._tau = tau
        self._policy = policy
        self._q_func = q_func
        self._actor_optim = actor_optim
        self._critic_optim = critic_optim
        self._targ_q_func = copy.deepcopy(q_func)
        self._targ_policy = copy.deepcopy(policy)

    @train_api
    def update_critic(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._critic_optim.step()

        return {"critic_loss": float(loss.cpu().detach().numpy())}

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    @train_api
    def update_actor(self, batch: TorchMiniBatch) -> Dict[str, float]:
        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        return {"actor_loss": float(loss.cpu().detach().numpy())}

    @abstractmethod
    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy(x).squashed_mu

    @abstractmethod
    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def update_critic_target(self) -> None:
        soft_sync(self._targ_q_func, self._q_func, self._tau)

    def update_actor_target(self) -> None:
        soft_sync(self._targ_policy, self._policy, self._tau)

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def policy_optim(self) -> Optimizer:
        return self._actor_optim

    @property
    def q_function(self) -> EnsembleQFunction:
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        return self._critic_optim


class DDPGImpl(DDPGBaseImpl):
    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action.squashed_mu, "none")[0]
        return -q_t.mean()

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action.squashed_mu.clamp(-1.0, 1.0),
                reduction="min",
            )

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)
