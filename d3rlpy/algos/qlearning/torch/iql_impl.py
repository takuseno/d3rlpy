import dataclasses

import torch
from torch import nn

from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    Policy,
    ValueFunction,
    build_gaussian_distribution,
)
from ....optimizers.optimizers import OptimizerWrapper
from ....torch_utility import TorchMiniBatch, soft_sync
from .ddpg_impl import (
    DDPGBaseActorLoss,
    DDPGBaseActorLossFn,
    DDPGBaseCriticLoss,
    DDPGBaseCriticLossFn,
    DDPGBaseModules,
    DDPGBaseUpdater,
)

__all__ = [
    "IQLCriticLossFn",
    "IQLActorLossFn",
    "IQLCriticLoss",
    "IQLModules",
    "IQLUpdater",
]


@dataclasses.dataclass(frozen=True)
class IQLModules(DDPGBaseModules):
    policy: NormalPolicy
    value_func: ValueFunction


@dataclasses.dataclass(frozen=True)
class IQLCriticLoss(DDPGBaseCriticLoss):
    q_loss: torch.Tensor
    v_loss: torch.Tensor


class IQLCriticLossFn(DDPGBaseCriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        value_func: ValueFunction,
        gamma: float,
        expectile: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
        )
        self._value_func = value_func
        self._expectile = expectile

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._value_func(batch.next_observations)

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions, "min"
        )
        v_t = self._value_func(batch.observations)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    def __call__(self, batch: TorchMiniBatch) -> IQLCriticLoss:
        q_tpn = self.compute_target(batch)
        q_loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        v_loss = self.compute_value_loss(batch)
        return IQLCriticLoss(
            critic_loss=q_loss + v_loss,
            q_loss=q_loss,
            v_loss=v_loss,
        )


class IQLActorLossFn(DDPGBaseActorLossFn):
    def __init__(
        self,
        policy: Policy,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        value_func: ValueFunction,
        weight_temp: float,
        max_weight: float,
    ):
        self._policy = policy
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._value_func = value_func
        self._weight_temp = weight_temp
        self._max_weight = max_weight

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions, "min"
        )
        v_t = self._value_func(batch.observations)
        adv = q_t - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        # compute log probability
        action = self._policy(batch.observations)
        dist = build_gaussian_distribution(action)
        log_probs = dist.log_prob(batch.actions)
        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)
        return DDPGBaseActorLoss(-(weight * log_probs).mean())


class IQLUpdater(DDPGBaseUpdater):
    def __init__(
        self,
        q_funcs: nn.ModuleList,
        targ_q_funcs: nn.ModuleList,
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
        self._tau = tau

    def update_target(self) -> None:
        soft_sync(self._targ_q_funcs, self._q_funcs, self._tau)
