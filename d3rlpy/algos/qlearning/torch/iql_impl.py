import dataclasses

import torch
import torch.nn.functional as F

from ....models.torch import (
    ActionOutput,
    CategoricalPolicy,
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    NormalPolicy,
    ValueFunction,
    build_gaussian_distribution,
)
from ....torch_utility import TorchMiniBatch
from ....types import Shape, TorchObservation
from .ddpg_impl import (
    DDPGBaseActorLoss,
    DDPGBaseCriticLoss,
    DDPGBaseImpl,
    DDPGBaseModules,
    DiscreteDDPGBaseImpl,
)

__all__ = ["IQLImpl", "IQLModules", "DiscreteIQLImpl", "DiscreteIQLModules"]


@dataclasses.dataclass(frozen=True)
class IQLModules(DDPGBaseModules):
    policy: NormalPolicy
    value_func: ValueFunction


@dataclasses.dataclass(frozen=True)
class IQLCriticLoss(DDPGBaseCriticLoss):
    q_loss: torch.Tensor
    v_loss: torch.Tensor


class IQLImpl(DDPGBaseImpl):
    _modules: IQLModules
    _expectile: float
    _weight_temp: float
    _max_weight: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: IQLModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        expectile: float,
        weight_temp: float,
        max_weight: float,
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
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> IQLCriticLoss:
        q_loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
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

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._modules.value_func(batch.next_observations)

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> DDPGBaseActorLoss:
        # compute log probability
        dist = build_gaussian_distribution(action)
        log_probs = dist.log_prob(batch.actions)
        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)
        return DDPGBaseActorLoss(-(weight * log_probs).mean())

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions, "min"
        )
        v_t = self._modules.value_func(batch.observations)
        adv = q_t - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions, "min"
        )
        v_t = self._modules.value_func(batch.observations)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = build_gaussian_distribution(self._modules.policy(x))
        return dist.sample()


@dataclasses.dataclass(frozen=True)
class DiscreteIQLModules(DDPGBaseModules):
    policy: CategoricalPolicy
    value_func: ValueFunction


@dataclasses.dataclass(frozen=True)
class DiscreteIQLCriticLoss(DDPGBaseCriticLoss):
    q_loss: torch.Tensor
    v_loss: torch.Tensor


class DiscreteIQLImpl(DiscreteDDPGBaseImpl):
    _modules: DiscreteIQLModules
    _expectile: float
    _weight_temp: float
    _max_weight: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteIQLModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        expectile: float,
        weight_temp: float,
        max_weight: float,
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
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> IQLCriticLoss:
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

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._modules.value_func(batch.next_observations)

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: None
    ) -> DDPGBaseActorLoss:
        assert self._modules.policy
        # compute weight
        with torch.no_grad():
            v = self._modules.value_func(batch.observations)
            min_Q = self._targ_q_func_forwarder.compute_target(
                batch.observations, reduction="min"
            ).gather(1, batch.actions.long())

        exp_a = torch.exp((min_Q - v) * self._weight_temp).clamp(
            max=self._max_weight
        )
        # compute log probability
        dist = self._modules.policy(batch.observations)
        log_probs = dist.log_prob(batch.actions.squeeze(-1)).unsqueeze(1)

        return DDPGBaseActorLoss(-(exp_a * log_probs).mean())

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(batch.observations)
        one_hot = F.one_hot(
            batch.actions.long().view(-1), num_classes=self.action_size
        )
        q_t = (q_t * one_hot).sum(dim=1, keepdim=True)

        v_t = self._modules.value_func(batch.observations)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = self._modules.policy(x)
        return dist.sample()
