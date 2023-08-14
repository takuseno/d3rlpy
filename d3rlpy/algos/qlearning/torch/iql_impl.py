import dataclasses
from typing import Dict

import torch

from ....dataset import Shape
from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    ValueFunction,
    build_gaussian_distribution,
)
from ....torch_utility import TorchMiniBatch, train_api
from .ddpg_impl import DDPGBaseImpl, DDPGBaseModules

__all__ = ["IQLImpl", "IQLModules"]


@dataclasses.dataclass(frozen=True)
class IQLModules(DDPGBaseModules):
    policy: NormalPolicy
    value_func: ValueFunction


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
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._modules.value_func(batch.next_observations)

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        # compute log probability
        dist = build_gaussian_distribution(
            self._modules.policy(batch.observations)
        )
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)

        return -(weight * log_probs).mean()

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

    @train_api
    def update_critic_and_state_value(
        self, batch: TorchMiniBatch
    ) -> Dict[str, float]:
        self._modules.critic_optim.zero_grad()

        # compute Q-function loss
        q_tpn = self.compute_target(batch)
        q_loss = self.compute_critic_loss(batch, q_tpn)

        # compute value function loss
        v_loss = self.compute_value_loss(batch)

        loss = q_loss + v_loss

        loss.backward()
        self._modules.critic_optim.step()

        return {
            "critic_loss": float(q_loss.cpu().detach().numpy()),
            "v_loss": float(v_loss.cpu().detach().numpy()),
        }

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        dist = build_gaussian_distribution(self._modules.policy(x))
        return dist.sample()
