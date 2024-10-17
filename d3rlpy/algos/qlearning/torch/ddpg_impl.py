import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Dict

import torch
from torch import nn
from torch.optim import Optimizer

from ....dataclass_utils import asdict_as_float
from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    Policy,
)
from ....torch_utility import Modules, TorchMiniBatch, hard_sync, soft_sync
from ....types import Shape, TorchObservation
from ..base import QLearningAlgoImplBase
from .utility import ContinuousQFunctionMixin, DiscreteQFunctionMixin

__all__ = [
    "DDPGImpl",
    "DDPGBaseImpl",
    "DDPGBaseModules",
    "DDPGModules",
    "DDPGBaseActorLoss",
    "DDPGBaseCriticLoss",
]


@dataclasses.dataclass(frozen=True)
class DDPGBaseModules(Modules):
    policy: Policy
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    actor_optim: Optimizer
    critic_optim: Optimizer


@dataclasses.dataclass(frozen=True)
class DDPGBaseActorLoss:
    actor_loss: torch.Tensor


@dataclasses.dataclass(frozen=True)
class DDPGBaseCriticLoss:
    critic_loss: torch.Tensor


class DDPGBaseImpl(
    ContinuousQFunctionMixin, QLearningAlgoImplBase, metaclass=ABCMeta
):
    _modules: DDPGBaseModules
    _gamma: float
    _tau: float
    _q_func_forwarder: ContinuousEnsembleQFunctionForwarder
    _targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DDPGBaseModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._gamma = gamma
        self._tau = tau
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

    def update_critic(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.critic_optim.zero_grad()
        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)
        loss.critic_loss.backward()
        self._modules.critic_optim.step()
        return asdict_as_float(loss)

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> DDPGBaseCriticLoss:
        loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        return DDPGBaseCriticLoss(loss)

    def update_actor(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> Dict[str, float]:
        # Q function should be inference mode for stability
        self._modules.q_funcs.eval()
        self._modules.actor_optim.zero_grad()
        loss = self.compute_actor_loss(batch, action)
        loss.actor_loss.backward()
        self._modules.actor_optim.step()
        return asdict_as_float(loss)

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = {}
        action = self._modules.policy(batch.observations)
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch, action))
        self.update_critic_target()
        return metrics

    @abstractmethod
    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> DDPGBaseActorLoss:
        pass

    @abstractmethod
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self._modules.policy(x).squashed_mu

    @abstractmethod
    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        pass

    def update_critic_target(self) -> None:
        soft_sync(self._modules.targ_q_funcs, self._modules.q_funcs, self._tau)

    @property
    def policy(self) -> Policy:
        return self._modules.policy

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.actor_optim

    @property
    def q_function(self) -> nn.ModuleList:
        return self._modules.q_funcs

    @property
    def q_function_optim(self) -> Optimizer:
        return self._modules.critic_optim


class DiscreteDDPGBaseImpl(
    DiscreteQFunctionMixin, QLearningAlgoImplBase, metaclass=ABCMeta
):
    _modules: DDPGBaseModules
    _gamma: float
    _tau: float
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DDPGBaseModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._gamma = gamma
        self._tau = tau
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

    def update_critic(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.critic_optim.zero_grad()
        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)
        loss.critic_loss.backward()
        self._modules.critic_optim.step()
        return asdict_as_float(loss)

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> DDPGBaseCriticLoss:
        loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        return DDPGBaseCriticLoss(loss)

    def update_actor(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> Dict[str, float]:
        # Q function should be inference mode for stability
        self._modules.q_funcs.eval()
        self._modules.actor_optim.zero_grad()
        loss = self.compute_actor_loss(batch, None)
        loss.actor_loss.backward()
        self._modules.actor_optim.step()
        return asdict_as_float(loss)

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = {}
        action = self._modules.policy(batch.observations)
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch, action))
        self.update_critic_target()
        return metrics

    @abstractmethod
    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: None
    ) -> DDPGBaseActorLoss:
        pass

    @abstractmethod
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return torch.argmax(self._modules.policy(x).probs).unsqueeze(0)

    @abstractmethod
    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        pass

    def update_critic_target(self) -> None:
        soft_sync(self._modules.targ_q_funcs, self._modules.q_funcs, self._tau)

    @property
    def policy(self) -> Policy:
        return self._modules.policy

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.actor_optim

    @property
    def q_function(self) -> nn.ModuleList:
        return self._modules.q_funcs

    @property
    def q_function_optim(self) -> Optimizer:
        return self._modules.critic_optim


@dataclasses.dataclass(frozen=True)
class DDPGModules(DDPGBaseModules):
    targ_policy: Policy


class DDPGImpl(DDPGBaseImpl):
    _modules: DDPGModules

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DDPGModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            device=device,
        )
        hard_sync(self._modules.targ_policy, self._modules.policy)

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> DDPGBaseActorLoss:
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )[0]
        return DDPGBaseActorLoss(-q_t.mean())

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self._modules.targ_policy(batch.next_observations)
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action.squashed_mu.clamp(-1.0, 1.0),
                reduction="min",
            )

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def update_actor_target(self) -> None:
        soft_sync(self._modules.targ_policy, self._modules.policy, self._tau)

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = super().inner_update(batch, grad_step)
        self.update_actor_target()
        return metrics
