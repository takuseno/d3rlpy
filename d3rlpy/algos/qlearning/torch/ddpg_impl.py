import dataclasses
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from ....dataclass_utils import asdict_as_float
from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    Policy,
)
from ....optimizers.optimizers import OptimizerWrapper
from ....torch_utility import (
    CudaGraphWrapper,
    Modules,
    TorchMiniBatch,
    soft_sync,
)
from ....types import TorchObservation
from ..functional import Updater, ValuePredictor

__all__ = [
    "DDPGBaseModules",
    "DDPGModules",
    "DDPGBaseActorLoss",
    "DDPGBaseCriticLoss",
    "DDPGBaseCriticLossFn",
    "DDPGBaseActorLossFn",
    "DDPGCriticLossFn",
    "DDPGActorLossFn",
    "DDPGValuePredictor",
    "DDPGBaseUpdater",
    "DDPGUpdater",
]


@dataclasses.dataclass(frozen=True)
class DDPGBaseModules(Modules):
    policy: Policy
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    actor_optim: OptimizerWrapper
    critic_optim: OptimizerWrapper


@dataclasses.dataclass(frozen=True)
class DDPGBaseActorLoss:
    actor_loss: torch.Tensor


@dataclasses.dataclass(frozen=True)
class DDPGBaseCriticLoss:
    critic_loss: torch.Tensor


class DDPGBaseCriticLossFn(metaclass=ABCMeta):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
    ):
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._gamma = gamma

    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseCriticLoss:
        q_tpn = self.compute_target(batch)
        loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        return DDPGBaseCriticLoss(critic_loss=loss)

    @abstractmethod
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        raise NotImplementedError


class DDPGBaseActorLossFn(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        raise NotImplementedError


class DDPGCriticLossFn(DDPGBaseCriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_policy: Policy,
        gamma: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
        )
        self._targ_policy = targ_policy

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action.squashed_mu.clamp(-1.0, 1.0),
                reduction="min",
            )


class DDPGActorLossFn(DDPGBaseActorLossFn):
    def __init__(self, q_func_forwarder: ContinuousEnsembleQFunctionForwarder, policy: Policy):
        self._q_func_forwarder = q_func_forwarder
        self._policy = policy

    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        action = self._policy(batch.observations)
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )[0]
        return DDPGBaseActorLoss(-q_t.mean())


class DDPGBaseUpdater(Updater):
    def __init__(
        self,
        critic_optim: OptimizerWrapper,
        actor_optim: OptimizerWrapper,
        critic_loss_fn: DDPGBaseCriticLossFn,
        actor_loss_fn: DDPGBaseActorLossFn,
        compiled: bool,
    ):
        self._critic_optim = critic_optim
        self._actor_optim = actor_optim
        self._critic_loss_fn = critic_loss_fn
        self._actor_loss_fn = actor_loss_fn
        self._compute_critic_grad = CudaGraphWrapper(self.compute_critic_grad) if compiled else self.compute_critic_grad
        self._compute_actor_grad = CudaGraphWrapper(self.compute_actor_grad) if compiled else self.compute_actor_grad

    def compute_critic_grad(self, batch: TorchMiniBatch) -> DDPGBaseCriticLoss:
        self._critic_optim.zero_grad()
        loss = self._critic_loss_fn(batch)
        loss.critic_loss.backward()
        return loss

    def compute_actor_grad(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        self._actor_optim.zero_grad()
        loss = self._actor_loss_fn(batch)
        loss.actor_loss.backward()
        return loss

    def __call__(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        metrics = {}

        # update critic
        critic_loss = self._compute_critic_grad(batch)
        self._critic_optim.step()
        metrics.update(asdict_as_float(critic_loss))

        # update actor
        actor_loss = self._compute_actor_grad(batch)
        self._actor_optim.step()
        metrics.update(asdict_as_float(actor_loss))

        # update target networks
        self.update_target()

        return metrics

    @abstractmethod
    def update_target(self) -> None:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class DDPGModules(DDPGBaseModules):
    targ_policy: Policy


class DDPGValuePredictor(ValuePredictor):
    def __init__(self, q_func_forwarder: ContinuousEnsembleQFunctionForwarder):
        self._q_func_forwarder = q_func_forwarder

    def __call__(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        return self._q_func_forwarder.compute_expected_q(
            x, action, reduction="mean"
        ).reshape(-1)


class DDPGUpdater(DDPGBaseUpdater):
    def __init__(
        self,
        q_funcs: nn.ModuleList,
        targ_q_funcs: nn.ModuleList,
        policy: Policy,
        targ_policy: Policy,
        critic_optim: OptimizerWrapper,
        actor_optim: OptimizerWrapper,
        critic_loss_fn: DDPGBaseCriticLossFn,
        actor_loss_fn: DDPGBaseActorLossFn,
        tau: float,
        compiled: bool,
    ):
        super().__init__(
            critic_optim=critic_optim,
            actor_optim=actor_optim,
            critic_loss_fn=critic_loss_fn,
            actor_loss_fn=actor_loss_fn,
            compiled=compiled,
        )
        self._q_funcs = q_funcs
        self._targ_q_funcs = targ_q_funcs
        self._policy = policy
        self._targ_policy = targ_policy
        self._tau = tau

    def update_target(self) -> None:
        soft_sync(self._targ_q_funcs, self._q_funcs, self._tau)
        soft_sync(self._targ_policy, self._policy, self._tau)
