import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Dict

import torch
from torch import nn
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import ContinuousEnsembleQFunctionForwarder, Policy
from ....torch_utility import Modules, TorchMiniBatch, hard_sync, soft_sync
from ..base import QLearningAlgoImplBase
from .utility import ContinuousQFunctionMixin

__all__ = ["DDPGImpl", "DDPGBaseImpl", "DDPGBaseModules", "DDPGModules"]


@dataclasses.dataclass(frozen=True)
class DDPGBaseModules(Modules):
    policy: Policy
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    actor_optim: Optimizer
    critic_optim: Optimizer


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

        loss.backward()
        self._modules.critic_optim.step()

        return {"critic_loss": float(loss.cpu().detach().numpy())}

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def update_actor(self, batch: TorchMiniBatch) -> Dict[str, float]:
        # Q function should be inference mode for stability
        self._modules.q_funcs.eval()

        self._modules.actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._modules.actor_optim.step()

        return {"actor_loss": float(loss.cpu().detach().numpy())}

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = {}
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))
        self.update_critic_target()
        return metrics

    @abstractmethod
    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._modules.policy(x).squashed_mu

    @abstractmethod
    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
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

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        action = self._modules.policy(batch.observations)
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )[0]
        return -q_t.mean()

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self._modules.targ_policy(batch.next_observations)
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action.squashed_mu.clamp(-1.0, 1.0),
                reduction="min",
            )

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def update_actor_target(self) -> None:
        soft_sync(self._modules.targ_policy, self._modules.policy, self._tau)

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = super().inner_update(batch, grad_step)
        self.update_actor_target()
        return metrics
