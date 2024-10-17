import dataclasses
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from ....models.torch import (
    ActionOutput,
    CategoricalPolicy,
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    NormalPolicy,
    Parameter,
    Policy,
    build_squashed_gaussian_distribution,
    get_parameter,
)
from ....torch_utility import Modules, TorchMiniBatch, hard_sync
from ....types import Shape, TorchObservation
from ..base import QLearningAlgoImplBase
from .ddpg_impl import DDPGBaseActorLoss, DDPGBaseImpl, DDPGBaseModules
from .utility import DiscreteQFunctionMixin

__all__ = [
    "SACImpl",
    "DiscreteSACImpl",
    "SACModules",
    "DiscreteSACModules",
    "SACActorLoss",
]


@dataclasses.dataclass(frozen=True)
class SACModules(DDPGBaseModules):
    policy: NormalPolicy
    log_temp: Parameter
    temp_optim: Optional[Optimizer]


@dataclasses.dataclass(frozen=True)
class SACActorLoss(DDPGBaseActorLoss):
    temp: torch.Tensor
    temp_loss: torch.Tensor


class SACImpl(DDPGBaseImpl):
    _modules: SACModules

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: SACModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        device: str,
        clip_gradient_norm: Optional[float],
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
            clip_gradient_norm=clip_gradient_norm,
        )

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> SACActorLoss:
        dist = build_squashed_gaussian_distribution(action)
        sampled_action, log_prob = dist.sample_with_log_prob()

        if self._modules.temp_optim:
            temp_loss = self.update_temp(log_prob)
        else:
            temp_loss = torch.tensor(
                0.0, dtype=torch.float32, device=sampled_action.device
            )

        entropy = get_parameter(self._modules.log_temp).exp() * log_prob
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, sampled_action, "min"
        )
        return SACActorLoss(
            actor_loss=(entropy - q_t).mean(),
            temp_loss=temp_loss,
            temp=get_parameter(self._modules.log_temp).exp()[0][0],
        )

    def update_temp(self, log_prob: torch.Tensor) -> torch.Tensor:
        assert self._modules.temp_optim
        self._modules.temp_optim.zero_grad()
        with torch.no_grad():
            targ_temp = log_prob - self._action_size
        loss = -(get_parameter(self._modules.log_temp).exp() * targ_temp).mean()
        loss.backward()
        self._modules.temp_optim.step()
        return loss

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            dist = build_squashed_gaussian_distribution(
                self._modules.policy(batch.next_observations)
            )
            action, log_prob = dist.sample_with_log_prob()
            entropy = get_parameter(self._modules.log_temp).exp() * log_prob
            target = self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
            return target - entropy

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = build_squashed_gaussian_distribution(self._modules.policy(x))
        return dist.sample()


@dataclasses.dataclass(frozen=True)
class DiscreteSACModules(Modules):
    policy: CategoricalPolicy
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    log_temp: Optional[Parameter]
    actor_optim: Optimizer
    critic_optim: Optimizer
    temp_optim: Optional[Optimizer]


class DiscreteSACImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _modules: DiscreteSACModules
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _target_update_interval: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteSACModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        target_update_interval: int,
        gamma: float,
        device: str,
        clip_gradient_norm: Optional[float],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
            clip_grad_norm=clip_gradient_norm,
        )
        self._gamma = gamma
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._target_update_interval = target_update_interval
        hard_sync(modules.targ_q_funcs, modules.q_funcs)

    def update_critic(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self.clip_gradients()
        self._modules.critic_optim.step()

        return {"critic_loss": float(loss.cpu().detach().numpy())}

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            dist = self._modules.policy(batch.next_observations)
            log_probs = dist.logits
            probs = dist.probs
            if self._modules.log_temp is None:
                temp = torch.zeros_like(log_probs)
            else:
                temp = get_parameter(self._modules.log_temp).exp()
            entropy = temp * log_probs
            target = self._targ_q_func_forwarder.compute_target(
                batch.next_observations
            )
            keepdims = True
            if target.dim() == 3:
                entropy = entropy.unsqueeze(-1)
                probs = probs.unsqueeze(-1)
                keepdims = False
            return (probs * (target - entropy)).sum(dim=1, keepdim=keepdims)

    def compute_critic_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
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
        self.clip_gradients()
        self._modules.actor_optim.step()

        return {"actor_loss": float(loss.cpu().detach().numpy())}

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            q_t = self._q_func_forwarder.compute_expected_q(
                batch.observations, reduction="min"
            )
        dist = self._modules.policy(batch.observations)
        log_probs = dist.logits
        probs = dist.probs
        if self._modules.log_temp is None:
            temp = torch.zeros_like(log_probs)
        else:
            temp = get_parameter(self._modules.log_temp).exp()
        entropy = temp * log_probs
        return (probs * (entropy - q_t)).sum(dim=1).mean()

    def update_temp(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._modules.temp_optim
        assert self._modules.log_temp is not None
        self._modules.temp_optim.zero_grad()

        with torch.no_grad():
            dist = self._modules.policy(batch.observations)
            log_probs = F.log_softmax(dist.logits, dim=1)
            probs = dist.probs
            expct_log_probs = (probs * log_probs).sum(dim=1, keepdim=True)
            entropy_target = 0.98 * (-math.log(1 / self.action_size))
            targ_temp = expct_log_probs + entropy_target

        loss = -(get_parameter(self._modules.log_temp).exp() * targ_temp).mean()

        loss.backward()
        self._modules.temp_optim.step()

        # current temperature value
        log_temp = get_parameter(self._modules.log_temp)
        cur_temp = log_temp.exp().cpu().detach().numpy()[0][0]

        return {
            "temp_loss": float(loss.cpu().detach().numpy()),
            "temp": float(cur_temp),
        }

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = {}

        # lagrangian parameter update for SAC temeprature
        if self._modules.temp_optim:
            metrics.update(self.update_temp(batch))
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))

        if grad_step % self._target_update_interval == 0:
            self.update_target()

        return metrics

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        dist = self._modules.policy(x)
        return dist.probs.argmax(dim=1)

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = self._modules.policy(x)
        return dist.sample()

    def update_target(self) -> None:
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

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
