import dataclasses
from typing import Dict, Optional

import torch
from torch.optim import Optimizer

from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    Parameter,
    VAEDecoder,
    VAEEncoder,
    build_squashed_gaussian_distribution,
    compute_max_with_n_actions_and_indices,
    compute_vae_error,
    forward_vae_sample_n,
    get_parameter,
)
from ....torch_utility import (
    TorchMiniBatch,
    expand_and_repeat_recursively,
    flatten_left_recursively,
    get_batch_size,
)
from ....types import Shape, TorchObservation
from .sac_impl import SACActorLoss, SACImpl, SACModules

__all__ = ["BEARImpl", "BEARModules"]


def _gaussian_kernel(
    x: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    # x: (batch, n, 1, action), y: (batch, 1, n, action) -> (batch, n, n)
    return (-((x - y) ** 2).sum(dim=3) / (2 * sigma)).exp()


def _laplacian_kernel(
    x: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    # x: (batch, n, 1, action), y: (batch, 1, n, action) -> (batch, n, n)
    return (-(x - y).abs().sum(dim=3) / (2 * sigma)).exp()


@dataclasses.dataclass(frozen=True)
class BEARModules(SACModules):
    vae_encoder: VAEEncoder
    vae_decoder: VAEDecoder
    log_alpha: Parameter
    vae_optim: Optimizer
    alpha_optim: Optional[Optimizer]


@dataclasses.dataclass(frozen=True)
class BEARActorLoss(SACActorLoss):
    mmd_loss: torch.Tensor
    alpha: torch.Tensor


class BEARImpl(SACImpl):
    _modules: BEARModules
    _alpha_threshold: float
    _lam: float
    _n_action_samples: int
    _n_target_samples: int
    _n_mmd_action_samples: int
    _mmd_kernel: str
    _mmd_sigma: float
    _vae_kl_weight: float
    _warmup_steps: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: BEARModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        alpha_threshold: float,
        lam: float,
        n_action_samples: int,
        n_target_samples: int,
        n_mmd_action_samples: int,
        mmd_kernel: str,
        mmd_sigma: float,
        vae_kl_weight: float,
        warmup_steps: int,
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
        self._alpha_threshold = alpha_threshold
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._n_target_samples = n_target_samples
        self._n_mmd_action_samples = n_mmd_action_samples
        self._mmd_kernel = mmd_kernel
        self._mmd_sigma = mmd_sigma
        self._vae_kl_weight = vae_kl_weight
        self._warmup_steps = warmup_steps

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> BEARActorLoss:
        loss = super().compute_actor_loss(batch, action)
        mmd_loss = self._compute_mmd_loss(batch.observations)
        if self._modules.alpha_optim:
            self.update_alpha(mmd_loss)
        return BEARActorLoss(
            actor_loss=loss.actor_loss + mmd_loss,
            temp_loss=loss.temp_loss,
            temp=loss.temp,
            mmd_loss=mmd_loss,
            alpha=get_parameter(self._modules.log_alpha).exp()[0][0],
        )

    def warmup_actor(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.actor_optim.zero_grad()
        loss = self._compute_mmd_loss(batch.observations)
        loss.backward()
        self.clip_gradients()
        self._modules.actor_optim.step()
        return {"actor_loss": float(loss.cpu().detach().numpy())}

    def _compute_mmd_loss(self, obs_t: TorchObservation) -> torch.Tensor:
        mmd = self._compute_mmd(obs_t)
        alpha = get_parameter(self._modules.log_alpha).exp()
        return (alpha * (mmd - self._alpha_threshold)).mean()

    def update_imitator(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.vae_optim.zero_grad()
        loss = self.compute_imitator_loss(batch)
        loss.backward()
        self.clip_gradients()
        self._modules.vae_optim.step()
        return {"imitator_loss": float(loss.cpu().detach().numpy())}

    def compute_imitator_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        return compute_vae_error(
            vae_encoder=self._modules.vae_encoder,
            vae_decoder=self._modules.vae_decoder,
            x=batch.observations,
            action=batch.actions,
            beta=self._vae_kl_weight,
        )

    def update_alpha(self, mmd_loss: torch.Tensor) -> None:
        assert self._modules.alpha_optim
        self._modules.alpha_optim.zero_grad()
        loss = -mmd_loss
        loss.backward(retain_graph=True)
        self._modules.alpha_optim.step()
        # clip for stability
        get_parameter(self._modules.log_alpha).data.clamp_(-5.0, 10.0)

    def _compute_mmd(self, x: TorchObservation) -> torch.Tensor:
        with torch.no_grad():
            behavior_actions = forward_vae_sample_n(
                vae_decoder=self._modules.vae_decoder,
                x=x,
                latent_size=2 * self._action_size,
                n=self._n_mmd_action_samples,
                with_squash=False,
            )
        dist = build_squashed_gaussian_distribution(self._modules.policy(x))
        policy_actions = dist.sample_n_without_squash(
            self._n_mmd_action_samples
        )

        if self._mmd_kernel == "gaussian":
            kernel = _gaussian_kernel
        elif self._mmd_kernel == "laplacian":
            kernel = _laplacian_kernel
        else:
            raise ValueError(f"Invalid kernel type: {self._mmd_kernel}")

        # (batch, n, action) -> (batch, n, 1, action)
        batch_size = (
            x.shape[0] if isinstance(x, torch.Tensor) else x[0].shape[0]
        )
        behavior_actions = behavior_actions.reshape(
            batch_size, -1, 1, self.action_size
        )
        policy_actions = policy_actions.reshape(
            batch_size, -1, 1, self.action_size
        )
        # (batch, n, action) -> (batch, 1, n, action)
        behavior_actions_T = behavior_actions.reshape(
            batch_size, 1, -1, self.action_size
        )
        policy_actions_T = policy_actions.reshape(
            batch_size, 1, -1, self.action_size
        )

        # 1 / N^2 \sum k(a_\pi, a_\pi)
        inter_policy = kernel(policy_actions, policy_actions_T, self._mmd_sigma)
        mmd = inter_policy.mean(dim=[1, 2])

        # 1 / N^2 \sum k(a_\beta, a_\beta)
        inter_data = kernel(
            behavior_actions, behavior_actions_T, self._mmd_sigma
        )
        mmd += inter_data.mean(dim=[1, 2])

        # 2 / N^2 \sum k(a_\pi, a_\beta)
        distance = kernel(policy_actions, behavior_actions_T, self._mmd_sigma)
        mmd -= 2 * distance.mean(dim=[1, 2])

        return (mmd + 1e-6).sqrt().view(-1, 1)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            # BCQ-like target computation
            dist = build_squashed_gaussian_distribution(
                self._modules.policy(batch.next_observations)
            )
            actions, log_probs = dist.sample_n_with_log_prob(
                self._n_target_samples
            )
            values, indices = compute_max_with_n_actions_and_indices(
                batch.next_observations,
                actions,
                self._targ_q_func_forwarder,
                self._lam,
            )

            # (batch, n, 1) -> (batch, 1)
            batch_size = get_batch_size(batch.observations)
            max_log_prob = log_probs[torch.arange(batch_size), indices]

            log_temp = get_parameter(self._modules.log_temp)
            return values - log_temp.exp() * max_log_prob

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        batch_size = (
            x.shape[0] if isinstance(x, torch.Tensor) else x[0].shape[0]
        )
        with torch.no_grad():
            # (batch, n, action)
            dist = build_squashed_gaussian_distribution(self._modules.policy(x))
            actions = dist.onnx_safe_sample_n(self._n_action_samples)
            # (batch, n, action) -> (batch * n, action)
            flat_actions = actions.reshape(-1, self._action_size)

            # (batch, observation) -> (batch, n, observation)
            repeated_x = expand_and_repeat_recursively(
                x, self._n_action_samples
            )
            # (batch, n, observation) -> (batch * n, observation)
            flat_x = flatten_left_recursively(repeated_x, dim=1)

            # (batch * n, 1)
            flat_values = self._q_func_forwarder.compute_expected_q(
                flat_x, flat_actions, "none"
            )[0]

            # (batch, n)
            values = flat_values.view(-1, self._n_action_samples)

            # (batch, n) -> (batch,)
            max_indices = torch.argmax(values, dim=1)

            return actions[torch.arange(batch_size), max_indices]

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = {}
        metrics.update(self.update_imitator(batch))
        metrics.update(self.update_critic(batch))
        if grad_step < self._warmup_steps:
            actor_loss = self.warmup_actor(batch)
        else:
            action = self._modules.policy(batch.observations)
            actor_loss = self.update_actor(batch, action)
        metrics.update(actor_loss)
        self.update_critic_target()
        return metrics
