from typing import Tuple

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    EnsembleContinuousQFunction,
    NonSquashedNormalPolicy,
    ValueFunction,
)
from ....torch_utility import TorchMiniBatch, train_api
from .ddpg_impl import DDPGBaseImpl

__all__ = ["IQLImpl"]


class IQLImpl(DDPGBaseImpl):
    _policy: NonSquashedNormalPolicy
    _value_func: ValueFunction
    _expectile: float
    _weight_temp: float
    _max_weight: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: NonSquashedNormalPolicy,
        q_func: EnsembleContinuousQFunction,
        value_func: ValueFunction,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
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
            policy=policy,
            q_func=q_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            gamma=gamma,
            tau=tau,
            device=device,
        )
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._value_func = value_func

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

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._value_func(batch.next_observations)

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)

        return -(weight * log_probs).mean()

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        adv = q_t - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    @train_api
    def update_critic_and_state_value(
        self, batch: TorchMiniBatch
    ) -> Tuple[float, float]:
        self._critic_optim.zero_grad()

        # compute Q-function loss
        q_tpn = self.compute_target(batch)
        q_loss = self.compute_critic_loss(batch, q_tpn)

        # compute value function loss
        v_loss = self.compute_value_loss(batch)

        loss = q_loss + v_loss

        loss.backward()
        self._critic_optim.step()

        return float(q_loss.cpu().detach().numpy()), float(
            v_loss.cpu().detach().numpy()
        )
