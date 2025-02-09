import dataclasses
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from ....dataclass_utils import asdict_as_float
from ....models.torch import (
    CategoricalPolicy,
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    NormalPolicy,
    Parameter,
    Policy,
    build_squashed_gaussian_distribution,
    get_parameter,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import (
    CudaGraphWrapper,
    Modules,
    TorchMiniBatch,
    hard_sync,
    soft_sync,
)
from ....types import TorchObservation
from ..functional import ActionSampler, Updater
from .ddpg_impl import (
    DDPGBaseActorLoss,
    DDPGBaseActorLossFn,
    DDPGBaseCriticLossFn,
    DDPGBaseModules,
    DDPGBaseUpdater,
)

__all__ = [
    "SACCriticLossFn",
    "SACActorLossFn",
    "SACUpdater",
    "DiscreteSACActorLossFn",
    "DiscreteSACCriticLossFn",
    "CategoricalPolicyExploreActionSampler",
    "CategoricalPolicyExploitActionSampler",
    "DiscreteSACUpdater",
    "SACModules",
    "DiscreteSACModules",
    "SACActorLoss",
]


@dataclasses.dataclass(frozen=True)
class SACModules(DDPGBaseModules):
    policy: NormalPolicy
    log_temp: Parameter
    temp_optim: Optional[OptimizerWrapper]


@dataclasses.dataclass(frozen=True)
class SACActorLoss(DDPGBaseActorLoss):
    temp: torch.Tensor
    temp_loss: torch.Tensor


class SACCriticLossFn(DDPGBaseCriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        policy: Policy,
        log_temp: Parameter,
        gamma: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
        )
        self._policy = policy
        self._log_temp = log_temp

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            dist = build_squashed_gaussian_distribution(
                self._policy(batch.next_observations)
            )
            action, log_prob = dist.sample_with_log_prob()
            entropy = get_parameter(self._log_temp).exp() * log_prob
            target = self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
            return target - entropy


class SACActorLossFn(DDPGBaseActorLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        policy: Policy,
        log_temp: Parameter,
        temp_optim: Optional[OptimizerWrapper],
        action_size: int,
    ):
        self._q_func_forwarder = q_func_forwarder
        self._policy = policy
        self._log_temp = log_temp
        self._temp_optim = temp_optim
        self._action_size = action_size

    def update_temp(self, log_prob: torch.Tensor) -> torch.Tensor:
        assert self._temp_optim
        self._temp_optim.zero_grad()
        with torch.no_grad():
            targ_temp = log_prob - self._action_size
        loss = -(get_parameter(self._log_temp).exp() * targ_temp).mean()
        loss.backward()
        self._temp_optim.step()
        return loss

    def __call__(self, batch: TorchMiniBatch) -> SACActorLoss:
        action = self._policy(batch.observations)
        dist = build_squashed_gaussian_distribution(action)
        sampled_action, log_prob = dist.sample_with_log_prob()

        if self._temp_optim:
            temp_loss = self.update_temp(log_prob)
        else:
            temp_loss = torch.tensor(
                0.0, dtype=torch.float32, device=sampled_action.device
            )

        entropy = get_parameter(self._log_temp).exp() * log_prob
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, sampled_action, "min"
        )
        return SACActorLoss(
            actor_loss=(entropy - q_t).mean(),
            temp_loss=temp_loss,
            temp=get_parameter(self._log_temp).exp()[0][0],
        )


class SACUpdater(DDPGBaseUpdater):
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


@dataclasses.dataclass(frozen=True)
class DiscreteSACModules(Modules):
    policy: CategoricalPolicy
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    log_temp: Optional[Parameter]
    actor_optim: OptimizerWrapper
    critic_optim: OptimizerWrapper
    temp_optim: Optional[OptimizerWrapper]


class DiscreteSACCriticLossFn:
    def __init__(
        self,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        policy: CategoricalPolicy,
        log_temp: Optional[Parameter],
        gamma: float,
    ):
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._policy = policy
        self._log_temp = log_temp
        self._gamma = gamma

    def __call__(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_tpn = self.compute_target(batch)
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            dist = self._policy(batch.next_observations)
            log_probs = dist.logits
            probs = dist.probs
            if self._log_temp is None:
                temp = torch.zeros_like(log_probs)
            else:
                temp = get_parameter(self._log_temp).exp()
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


@dataclasses.dataclass(frozen=True)
class DiscreteSACActorLoss:
    actor_loss: torch.Tensor
    temp: torch.Tensor
    temp_loss: torch.Tensor


class DiscreteSACActorLossFn:
    def __init__(
        self,
        policy: CategoricalPolicy,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        log_temp: Optional[Parameter],
        temp_optim: Optional[OptimizerWrapper],
        action_size: int,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._log_temp = log_temp
        self._temp_optim = temp_optim
        self._action_size = action_size

    def __call__(self, batch: TorchMiniBatch) -> DiscreteSACActorLoss:
        with torch.no_grad():
            q_t = self._q_func_forwarder.compute_expected_q(
                batch.observations, reduction="min"
            )
        dist = self._policy(batch.observations)

        if self._temp_optim:
            temp_loss = self.update_temp(dist)
        else:
            temp_loss = torch.tensor(
                0.0, dtype=torch.float32, device=q_t.device
            )

        log_probs = dist.logits
        probs = dist.probs
        if self._log_temp is None:
            temp = torch.zeros_like(log_probs)
        else:
            temp = get_parameter(self._log_temp).exp()
        entropy = temp * log_probs

        return DiscreteSACActorLoss(
            actor_loss=(probs * (entropy - q_t)).sum(dim=1).mean(),
            temp_loss=temp_loss,
            temp=temp[0][0],
        )

    def update_temp(self, dist: Categorical) -> torch.Tensor:
        assert self._temp_optim
        assert self._log_temp is not None
        self._temp_optim.zero_grad()

        with torch.no_grad():
            log_probs = F.log_softmax(dist.logits, dim=1)
            probs = dist.probs
            expct_log_probs = (probs * log_probs).sum(dim=1, keepdim=True)
            entropy_target = 0.98 * (-math.log(1 / self._action_size))
            targ_temp = expct_log_probs + entropy_target

        loss = -(get_parameter(self._log_temp).exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        return loss


class CategoricalPolicyExploitActionSampler(ActionSampler):
    def __init__(self, policy: CategoricalPolicy):
        self._policy = policy

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        dist = self._policy(x)
        return dist.probs.argmax(dim=1)


class CategoricalPolicyExploreActionSampler(ActionSampler):
    def __init__(self, policy: CategoricalPolicy):
        self._policy = policy

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        dist = self._policy(x)
        return dist.sample()


class DiscreteSACUpdater(Updater):
    def __init__(
        self,
        q_funcs: nn.ModuleList,
        targ_q_funcs: nn.ModuleList,
        critic_optim: OptimizerWrapper,
        actor_optim: OptimizerWrapper,
        critic_loss_fn: DiscreteSACCriticLossFn,
        actor_loss_fn: DiscreteSACActorLossFn,
        target_update_interval: int,
        compiled: bool,
    ):
        self._q_funcs = q_funcs
        self._targ_q_funcs = targ_q_funcs
        self._critic_optim = critic_optim
        self._actor_optim = actor_optim
        self._critic_loss_fn = critic_loss_fn
        self._actor_loss_fn = actor_loss_fn
        self._target_update_interval = target_update_interval
        self._compute_critic_grad = (
            CudaGraphWrapper(self.compute_critic_grad)
            if compiled
            else self.compute_critic_grad
        )
        self._compute_actor_grad = (
            CudaGraphWrapper(self.compute_actor_grad)
            if compiled
            else self.compute_actor_grad
        )

    def compute_actor_grad(self, batch: TorchMiniBatch) -> DiscreteSACActorLoss:
        self._actor_optim.zero_grad()
        loss = self._actor_loss_fn(batch)
        loss.actor_loss.backward()
        return loss

    def compute_critic_grad(self, batch: TorchMiniBatch) -> torch.Tensor:
        self._critic_optim.zero_grad()
        loss = self._critic_loss_fn(batch)
        loss.backward()
        return loss

    def __call__(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        metrics = {}

        critic_loss = self._compute_critic_grad(batch)
        self._critic_optim.step()
        metrics.update({"critic_loss": float(critic_loss.detach().cpu())})

        actor_loss = self._compute_actor_grad(batch)
        self._actor_optim.step()
        metrics.update(asdict_as_float(actor_loss))

        if grad_step % self._target_update_interval == 0:
            hard_sync(self._targ_q_funcs, self._q_funcs)

        return metrics
