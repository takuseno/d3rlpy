import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    CategoricalPolicy,
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    Parameter,
    Policy,
    build_squashed_gaussian_distribution,
)
from ....torch_utility import Checkpointer, TorchMiniBatch, hard_sync, train_api
from ..base import QLearningAlgoImplBase
from .ddpg_impl import DDPGBaseImpl
from .utility import DiscreteQFunctionMixin

__all__ = ["SACImpl", "DiscreteSACImpl"]


class SACImpl(DDPGBaseImpl):
    _log_temp: Parameter
    _temp_optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: Policy,
        q_funcs: nn.ModuleList,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_funcs: nn.ModuleList,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        log_temp: Parameter,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        temp_optim: Optimizer,
        gamma: float,
        tau: float,
        checkpointer: Checkpointer,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_funcs=q_funcs,
            q_func_forwarder=q_func_forwarder,
            targ_q_funcs=targ_q_funcs,
            targ_q_func_forwarder=targ_q_func_forwarder,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            gamma=gamma,
            tau=tau,
            checkpointer=checkpointer,
            device=device,
        )
        self._log_temp = log_temp
        self._temp_optim = temp_optim

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        dist = build_squashed_gaussian_distribution(
            self._policy(batch.observations)
        )
        action, log_prob = dist.sample_with_log_prob()
        entropy = self._log_temp().exp() * log_prob
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action, "min"
        )
        return (entropy - q_t).mean()

    @train_api
    def update_temp(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._temp_optim.zero_grad()

        with torch.no_grad():
            dist = build_squashed_gaussian_distribution(
                self._policy(batch.observations)
            )
            _, log_prob = dist.sample_with_log_prob()
            targ_temp = log_prob - self._action_size

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return {
            "temp_loss": float(loss.cpu().detach().numpy()),
            "temp": float(cur_temp),
        }

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            dist = build_squashed_gaussian_distribution(
                self._policy(batch.next_observations)
            )
            action, log_prob = dist.sample_with_log_prob()
            entropy = self._log_temp().exp() * log_prob
            target = self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
            return target - entropy

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        dist = build_squashed_gaussian_distribution(self._policy(x))
        return dist.sample()


class DiscreteSACImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _policy: CategoricalPolicy
    _q_funcss: nn.ModuleList
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_funcs: nn.ModuleList
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _log_temp: Parameter
    _actor_optim: Optimizer
    _critic_optim: Optimizer
    _temp_optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_funcs: nn.ModuleList,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_funcs: nn.ModuleList,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        policy: CategoricalPolicy,
        log_temp: Parameter,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        temp_optim: Optimizer,
        gamma: float,
        checkpointer: Checkpointer,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            checkpointer=checkpointer,
            device=device,
        )
        self._gamma = gamma
        self._q_funcs = q_funcs
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_funcs = targ_q_funcs
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._policy = policy
        self._log_temp = log_temp
        self._actor_optim = actor_optim
        self._critic_optim = critic_optim
        self._temp_optim = temp_optim
        hard_sync(targ_q_funcs, q_funcs)

    @train_api
    def update_critic(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._critic_optim.step()

        return {"critic_loss": float(loss.cpu().detach().numpy())}

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            dist = self._policy(batch.next_observations)
            log_probs = dist.logits
            probs = dist.probs
            entropy = self._log_temp().exp() * log_probs
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

    @train_api
    def update_actor(self, batch: TorchMiniBatch) -> Dict[str, float]:
        # Q function should be inference mode for stability
        self._q_funcs.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        return {"actor_loss": float(loss.cpu().detach().numpy())}

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            q_t = self._q_func_forwarder.compute_expected_q(
                batch.observations, reduction="min"
            )
        dist = self._policy(batch.observations)
        log_probs = dist.logits
        probs = dist.probs
        entropy = self._log_temp().exp() * log_probs
        return (probs * (entropy - q_t)).sum(dim=1).mean()

    @train_api
    def update_temp(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._temp_optim.zero_grad()

        with torch.no_grad():
            dist = self._policy(batch.observations)
            log_probs = F.log_softmax(dist.logits, dim=1)
            probs = dist.probs
            expct_log_probs = (probs * log_probs).sum(dim=1, keepdim=True)
            entropy_target = 0.98 * (-math.log(1 / self.action_size))
            targ_temp = expct_log_probs + entropy_target

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return {
            "temp_loss": float(loss.cpu().detach().numpy()),
            "temp": float(cur_temp),
        }

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        dist = self._policy(x)
        return dist.probs.argmax(dim=1)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        dist = self._policy(x)
        return dist.sample()

    def update_target(self) -> None:
        hard_sync(self._targ_q_funcs, self._q_funcs)

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def policy_optim(self) -> Optimizer:
        return self._actor_optim

    @property
    def q_function(self) -> nn.ModuleList:
        return self._q_funcs

    @property
    def q_function_optim(self) -> Optimizer:
        return self._critic_optim
