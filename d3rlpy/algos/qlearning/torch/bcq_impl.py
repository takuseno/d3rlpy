import dataclasses
import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from ....models.torch import (
    CategoricalPolicy,
    ContinuousEnsembleQFunctionForwarder,
    DeterministicResidualPolicy,
    DiscreteEnsembleQFunctionForwarder,
    Policy,
    VAEDecoder,
    VAEEncoder,
    compute_discrete_imitation_loss,
    compute_max_with_n_actions,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import (
    CudaGraphWrapper,
    TorchMiniBatch,
    expand_and_repeat_recursively,
    flatten_left_recursively,
    get_batch_size,
    get_device,
)
from ....types import TorchObservation
from ..functional import ActionSampler
from ..functional_utils import VAELossFn
from .ddpg_impl import (
    DDPGBaseActorLoss,
    DDPGBaseActorLossFn,
    DDPGBaseCriticLossFn,
    DDPGBaseModules,
    DDPGUpdater,
)
from .dqn_impl import DoubleDQNLossFn, DQNLoss, DQNModules

__all__ = [
    "BCQModules",
    "BCQCriticLossFn",
    "BCQActorLossFn",
    "BCQUpdater",
    "BCQActionSampler",
    "DiscreteBCQModules",
    "DiscreteBCQLoss",
    "DiscreteBCQLossFn",
    "DiscreteBCQActionSampler",
]


@dataclasses.dataclass(frozen=True)
class BCQModules(DDPGBaseModules):
    policy: DeterministicResidualPolicy
    targ_policy: DeterministicResidualPolicy
    vae_encoder: VAEEncoder
    vae_decoder: VAEDecoder
    vae_optim: OptimizerWrapper


def _repeat_observation(x: TorchObservation, n: int) -> TorchObservation:
    # (batch_size, *obs_shape) -> (batch_size, n, *obs_shape)
    return expand_and_repeat_recursively(x, n)


def _sample_repeated_action(
    x: TorchObservation,
    policy: Policy,
    vae_decoder: VAEDecoder,
    n_action_samples: int,
    action_size: int,
) -> torch.Tensor:
    repeated_x = _repeat_observation(x, n_action_samples)
    flattened_x = flatten_left_recursively(repeated_x, dim=1)
    flattened_batch_size = (
        flattened_x.shape[0]
        if isinstance(flattened_x, torch.Tensor)
        else flattened_x[0].shape[0]
    )
    # sample latent variable
    latent = torch.randn(
        flattened_batch_size,
        2 * action_size,
        device=get_device(x),
    )
    clipped_latent = latent.clamp(-0.5, 0.5)
    # sample action
    sampled_action = vae_decoder(flattened_x, clipped_latent)
    # add residual action
    action = policy(flattened_x, sampled_action)
    return action.squashed_mu.view(-1, n_action_samples, action_size)


class BCQCriticLossFn(DDPGBaseCriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_policy: Policy,
        vae_decoder: VAEDecoder,
        gamma: float,
        n_action_samples: int,
        lam: float,
        action_size: int,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
        )
        self._targ_policy = targ_policy
        self._vae_decoder = vae_decoder
        self._n_action_samples = n_action_samples
        self._lam = lam
        self._action_size = action_size

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            actions = _sample_repeated_action(
                x=batch.next_observations,
                policy=self._targ_policy,
                vae_decoder=self._vae_decoder,
                n_action_samples=self._n_action_samples,
                action_size=self._action_size,
            )
            values = compute_max_with_n_actions(
                batch.next_observations,
                actions,
                self._targ_q_func_forwarder,
                self._lam,
            )
            return values


class BCQActorLossFn(DDPGBaseActorLossFn):
    def __init__(
        self,
        policy: Policy,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        vae_decoder: VAEDecoder,
        action_size: int,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._vae_decoder = vae_decoder
        self._action_size = action_size

    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        batch_size = get_batch_size(batch.observations)
        latent = torch.randn(
            batch_size,
            2 * self._action_size,
            device=get_device(batch.observations),
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        sampled_action = self._vae_decoder(
            x=batch.observations,
            latent=clipped_latent,
        )
        action = self._policy(batch.observations, sampled_action)
        value = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )
        return DDPGBaseActorLoss(-value[0].mean())


class BCQActionSampler(ActionSampler):
    def __init__(
        self,
        policy: Policy,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        vae_decoder: VAEDecoder,
        n_action_samples: int,
        action_size: int,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._vae_decoder = vae_decoder
        self._n_action_samples = n_action_samples
        self._action_size = action_size

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        # (batch_size, *obs_shape) -> (batch_size * n, *obs_shape)
        repeated_x = _repeat_observation(x, self._n_action_samples)
        # (batch_size, n, *obs_shape) -> (batch_size * n, *obs_shape)
        flattened_x = flatten_left_recursively(repeated_x, dim=1)
        # (batch_size, n, action_size)
        repeated_actions = _sample_repeated_action(
            x=x,
            policy=self._policy,
            vae_decoder=self._vae_decoder,
            n_action_samples=self._n_action_samples,
            action_size=self._action_size,
        )
        # (batch_size, n, action_size) -> (batch_size * n, action_size)
        flattend_action = repeated_actions.view(-1, self._action_size)

        # estimate values
        values = self._q_func_forwarder.compute_expected_q(
            flattened_x, flattend_action, "none"
        )[0]

        # pick the best (batch_size * n) -> (batch_size,)
        index = values.view(-1, self._n_action_samples).argmax(dim=1)
        return repeated_actions[torch.arange(repeated_actions.shape[0]), index]


class BCQUpdater(DDPGUpdater):
    def __init__(
        self,
        q_funcs: nn.ModuleList,
        targ_q_funcs: nn.ModuleList,
        policy: Policy,
        targ_policy: Policy,
        critic_optim: OptimizerWrapper,
        actor_optim: OptimizerWrapper,
        imitator_optim: OptimizerWrapper,
        critic_loss_fn: DDPGBaseCriticLossFn,
        actor_loss_fn: DDPGBaseActorLossFn,
        imitator_loss_fn: VAELossFn,
        tau: float,
        rl_start_step: int,
        compiled: bool,
    ):
        super().__init__(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            policy=policy,
            targ_policy=targ_policy,
            critic_optim=critic_optim,
            actor_optim=actor_optim,
            critic_loss_fn=critic_loss_fn,
            actor_loss_fn=actor_loss_fn,
            tau=tau,
            compiled=compiled,
        )
        self._imitator_optim = imitator_optim
        self._imitator_loss_fn = imitator_loss_fn
        self._compute_imitator_grad = (
            CudaGraphWrapper(self.compute_imitator_grad)
            if compiled
            else self.compute_imitator_grad
        )
        self._rl_start_step = rl_start_step

    def compute_imitator_grad(self, batch: TorchMiniBatch) -> torch.Tensor:
        self._imitator_optim.zero_grad()
        loss = self._imitator_loss_fn(batch)
        loss.backward()
        return loss

    def __call__(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        metrics = {}

        imitator_loss = self._compute_imitator_grad(batch)
        self._imitator_optim.step()
        metrics.update(
            {"imitator_loss": float(imitator_loss.detach().cpu().numpy())}
        )

        if grad_step < self._rl_start_step:
            return metrics

        metrics.update(super().__call__(batch, grad_step))

        return metrics


@dataclasses.dataclass(frozen=True)
class DiscreteBCQModules(DQNModules):
    imitator: CategoricalPolicy


@dataclasses.dataclass(frozen=True)
class DiscreteBCQLoss(DQNLoss):
    td_loss: torch.Tensor
    imitator_loss: torch.Tensor


class DiscreteBCQLossFn(DoubleDQNLossFn):
    def __init__(
        self,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        imitator: CategoricalPolicy,
        gamma: float,
        beta: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
        )
        self._imitator = imitator
        self._beta = beta

    def __call__(self, batch: TorchMiniBatch) -> DiscreteBCQLoss:
        td_loss = super().__call__(batch).loss
        imitator_loss = compute_discrete_imitation_loss(
            policy=self._imitator,
            x=batch.observations,
            action=batch.actions.long(),
            beta=self._beta,
        )
        loss = td_loss + imitator_loss.loss
        return DiscreteBCQLoss(
            loss=loss, td_loss=td_loss, imitator_loss=imitator_loss.loss
        )


class DiscreteBCQActionSampler(ActionSampler):
    def __init__(
        self,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        imitator: CategoricalPolicy,
        action_flexibility: float,
    ):
        self._q_func_forwarder = q_func_forwarder
        self._imitator = imitator
        self._action_flexibility = action_flexibility

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        dist = self._imitator(x)
        log_probs = F.log_softmax(dist.logits, dim=1)
        ratio = log_probs - log_probs.max(dim=1, keepdim=True).values
        mask = (ratio > math.log(self._action_flexibility)).float()
        value = self._q_func_forwarder.compute_expected_q(x)
        # add a small constant value to deal with the case where the all
        # actions except the min value are masked
        normalized_value = value - value.min(dim=1, keepdim=True).values + 1e-5
        action = (normalized_value * cast(torch.Tensor, mask)).argmax(dim=1)
        return action
