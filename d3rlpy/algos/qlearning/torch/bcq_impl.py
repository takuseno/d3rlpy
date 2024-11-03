import dataclasses
import math
from typing import Callable, Dict, cast

import torch
import torch.nn.functional as F

from ....models.torch import (
    ActionOutput,
    CategoricalPolicy,
    ContinuousEnsembleQFunctionForwarder,
    DeterministicResidualPolicy,
    DiscreteEnsembleQFunctionForwarder,
    VAEDecoder,
    VAEEncoder,
    compute_discrete_imitation_loss,
    compute_max_with_n_actions,
    compute_vae_error,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import (
    CudaGraphWrapper,
    TorchMiniBatch,
    expand_and_repeat_recursively,
    flatten_left_recursively,
    get_batch_size,
    soft_sync,
)
from ....types import Shape, TorchObservation
from .ddpg_impl import DDPGBaseActorLoss, DDPGBaseImpl, DDPGBaseModules
from .dqn_impl import DoubleDQNImpl, DQNLoss, DQNModules

__all__ = [
    "BCQImpl",
    "DiscreteBCQImpl",
    "BCQModules",
    "DiscreteBCQModules",
    "DiscreteBCQLoss",
]


@dataclasses.dataclass(frozen=True)
class BCQModules(DDPGBaseModules):
    policy: DeterministicResidualPolicy
    targ_policy: DeterministicResidualPolicy
    vae_encoder: VAEEncoder
    vae_decoder: VAEDecoder
    vae_optim: OptimizerWrapper


class BCQImpl(DDPGBaseImpl):
    _modules: BCQModules
    _compute_imitator_grad: Callable[[TorchMiniBatch], Dict[str, torch.Tensor]]
    _lam: float
    _n_action_samples: int
    _action_flexibility: float
    _beta: float
    _rl_start_step: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: BCQModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        lam: float,
        n_action_samples: int,
        action_flexibility: float,
        beta: float,
        rl_start_step: int,
        compile_graph: bool,
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
            compile_graph=compile_graph,
            device=device,
        )
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._action_flexibility = action_flexibility
        self._beta = beta
        self._rl_start_step = rl_start_step
        self._compute_imitator_grad = (
            CudaGraphWrapper(self.compute_imitator_grad)
            if compile_graph
            else self.compute_imitator_grad
        )

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> DDPGBaseActorLoss:
        value = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )
        return DDPGBaseActorLoss(-value[0].mean())

    def compute_actor_grad(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        # forward policy
        batch_size = get_batch_size(batch.observations)
        latent = torch.randn(
            batch_size, 2 * self._action_size, device=self._device
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        sampled_action = self._modules.vae_decoder(
            x=batch.observations,
            latent=clipped_latent,
        )
        action = self._modules.policy(batch.observations, sampled_action)

        self._modules.actor_optim.zero_grad()
        loss = self.compute_actor_loss(batch, action)
        loss.actor_loss.backward()
        return loss

    def compute_imitator_grad(
        self, batch: TorchMiniBatch
    ) -> Dict[str, torch.Tensor]:
        self._modules.vae_optim.zero_grad()
        loss = compute_vae_error(
            vae_encoder=self._modules.vae_encoder,
            vae_decoder=self._modules.vae_decoder,
            x=batch.observations,
            action=batch.actions,
            beta=self._beta,
        )
        loss.backward()
        return {"loss": loss}

    def update_imitator(self, batch: TorchMiniBatch) -> Dict[str, float]:
        loss = self._compute_imitator_grad(batch)
        self._modules.vae_optim.step()
        return {"vae_loss": float(loss["loss"].cpu().detach().numpy())}

    def _repeat_observation(self, x: TorchObservation) -> TorchObservation:
        # (batch_size, *obs_shape) -> (batch_size, n, *obs_shape)
        return expand_and_repeat_recursively(x, self._n_action_samples)

    def _sample_repeated_action(
        self, repeated_x: TorchObservation, target: bool = False
    ) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        flattened_x = flatten_left_recursively(repeated_x, dim=1)
        flattened_batch_size = (
            flattened_x.shape[0]
            if isinstance(flattened_x, torch.Tensor)
            else flattened_x[0].shape[0]
        )
        # sample latent variable
        latent = torch.randn(
            flattened_batch_size, 2 * self._action_size, device=self._device
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        # sample action
        sampled_action = self._modules.vae_decoder(flattened_x, clipped_latent)
        # add residual action
        policy = self._modules.targ_policy if target else self._modules.policy
        action = policy(flattened_x, sampled_action)
        return action.squashed_mu.view(
            -1, self._n_action_samples, self._action_size
        )

    def _predict_value(
        self,
        repeated_x: TorchObservation,
        action: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        # (batch_size, n, *obs_shape) -> (batch_size * n, *obs_shape)
        flattened_x = flatten_left_recursively(repeated_x, dim=1)
        # (batch_size, n, action_size) -> (batch_size * n, action_size)
        flattend_action = action.view(-1, self.action_size)
        # estimate values
        return self._q_func_forwarder.compute_expected_q(
            flattened_x, flattend_action, "none"
        )

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        repeated_x = self._repeat_observation(x)
        action = self._sample_repeated_action(repeated_x)
        values = self._predict_value(repeated_x, action)[0]
        # pick the best (batch_size * n) -> (batch_size,)
        index = values.view(-1, self._n_action_samples).argmax(dim=1)
        return action[torch.arange(action.shape[0]), index]

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        with torch.no_grad():
            repeated_x = self._repeat_observation(batch.next_observations)
            actions = self._sample_repeated_action(repeated_x, True)
            values = compute_max_with_n_actions(
                batch.next_observations,
                actions,
                self._targ_q_func_forwarder,
                self._lam,
            )
            return values

    def update_actor_target(self) -> None:
        soft_sync(self._modules.targ_policy, self._modules.policy, self._tau)

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = {}

        metrics.update(self.update_imitator(batch))
        if grad_step < self._rl_start_step:
            return metrics

        # update models
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))
        self.update_critic_target()
        self.update_actor_target()
        return metrics


@dataclasses.dataclass(frozen=True)
class DiscreteBCQModules(DQNModules):
    imitator: CategoricalPolicy


@dataclasses.dataclass(frozen=True)
class DiscreteBCQLoss(DQNLoss):
    td_loss: torch.Tensor
    imitator_loss: torch.Tensor


class DiscreteBCQImpl(DoubleDQNImpl):
    _modules: DiscreteBCQModules
    _action_flexibility: float
    _beta: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteBCQModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        target_update_interval: int,
        gamma: float,
        action_flexibility: float,
        beta: float,
        compile_graph: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=target_update_interval,
            gamma=gamma,
            compile_graph=compile_graph,
            device=device,
        )
        self._action_flexibility = action_flexibility
        self._beta = beta

    def compute_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> DiscreteBCQLoss:
        td_loss = super().compute_loss(batch, q_tpn).loss
        imitator_loss = compute_discrete_imitation_loss(
            policy=self._modules.imitator,
            x=batch.observations,
            action=batch.actions.long(),
            beta=self._beta,
        )
        loss = td_loss + imitator_loss.loss
        return DiscreteBCQLoss(
            loss=loss, td_loss=td_loss, imitator_loss=imitator_loss.loss
        )

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        dist = self._modules.imitator(x)
        log_probs = F.log_softmax(dist.logits, dim=1)
        ratio = log_probs - log_probs.max(dim=1, keepdim=True).values
        mask = (ratio > math.log(self._action_flexibility)).float()
        value = self._q_func_forwarder.compute_expected_q(x)
        # add a small constant value to deal with the case where the all
        # actions except the min value are masked
        normalized_value = value - value.min(dim=1, keepdim=True).values + 1e-5
        action = (normalized_value * cast(torch.Tensor, mask)).argmax(dim=1)
        return action
