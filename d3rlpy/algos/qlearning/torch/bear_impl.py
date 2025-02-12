import dataclasses
from typing import Optional

import torch
from torch import nn

from ....dataclass_utils import asdict_as_float
from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    Parameter,
    Policy,
    VAEDecoder,
    VAEEncoder,
    build_squashed_gaussian_distribution,
    compute_max_with_n_actions_and_indices,
    forward_vae_sample_n,
    get_parameter,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import (
    CudaGraphWrapper,
    TorchMiniBatch,
    expand_and_repeat_recursively,
    flatten_left_recursively,
    get_batch_size,
)
from ....types import TorchObservation
from ..functional import ActionSampler
from ..functional_utils import VAELossFn
from .ddpg_impl import DDPGBaseActorLossFn, DDPGBaseCriticLossFn
from .sac_impl import (
    SACActorLoss,
    SACActorLossFn,
    SACCriticLossFn,
    SACModules,
    SACUpdater,
)

__all__ = [
    "BEARModules",
    "BEARActorLoss",
    "BEARActorLossFn",
    "BEARWarmupActorLossFn",
    "BEARSquashedGaussianContinuousActionSampler",
    "BEARCriticLossFn",
    "BEARUpdater",
]


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
    vae_optim: OptimizerWrapper
    alpha_optim: Optional[OptimizerWrapper]


@dataclasses.dataclass(frozen=True)
class BEARActorLoss(SACActorLoss):
    mmd_loss: torch.Tensor
    alpha: torch.Tensor


class BEARCriticLossFn(SACCriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        policy: Policy,
        log_temp: Parameter,
        gamma: float,
        n_target_samples: int,
        lam: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            policy=policy,
            log_temp=log_temp,
            gamma=gamma,
        )
        self._n_target_samples = n_target_samples
        self._lam = lam

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            # BCQ-like target computation
            dist = build_squashed_gaussian_distribution(
                self._policy(batch.next_observations)
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

            log_temp = get_parameter(self._log_temp)
            return values - log_temp.exp() * max_log_prob


def compute_mmd(
    x: TorchObservation,
    policy: Policy,
    vae_decoder: VAEDecoder,
    action_size: int,
    n_mmd_action_samples: int,
    mmd_kernel: str,
    mmd_sigma: float,
) -> torch.Tensor:
    with torch.no_grad():
        behavior_actions = forward_vae_sample_n(
            vae_decoder=vae_decoder,
            x=x,
            latent_size=2 * action_size,
            n=n_mmd_action_samples,
            with_squash=False,
        )
    dist = build_squashed_gaussian_distribution(policy(x))
    policy_actions = dist.sample_n_without_squash(n_mmd_action_samples)

    if mmd_kernel == "gaussian":
        kernel = _gaussian_kernel
    elif mmd_kernel == "laplacian":
        kernel = _laplacian_kernel
    else:
        raise ValueError(f"Invalid kernel type: {mmd_kernel}")

    # (batch, n, action) -> (batch, n, 1, action)
    batch_size = x.shape[0] if isinstance(x, torch.Tensor) else x[0].shape[0]
    behavior_actions = behavior_actions.reshape(batch_size, -1, 1, action_size)
    policy_actions = policy_actions.reshape(batch_size, -1, 1, action_size)
    # (batch, n, action) -> (batch, 1, n, action)
    behavior_actions_T = behavior_actions.reshape(
        batch_size, 1, -1, action_size
    )
    policy_actions_T = policy_actions.reshape(batch_size, 1, -1, action_size)

    # 1 / N^2 \sum k(a_\pi, a_\pi)
    inter_policy = kernel(policy_actions, policy_actions_T, mmd_sigma)
    mmd = inter_policy.mean(dim=[1, 2])

    # 1 / N^2 \sum k(a_\beta, a_\beta)
    inter_data = kernel(behavior_actions, behavior_actions_T, mmd_sigma)
    mmd += inter_data.mean(dim=[1, 2])

    # 2 / N^2 \sum k(a_\pi, a_\beta)
    distance = kernel(policy_actions, behavior_actions_T, mmd_sigma)
    mmd -= 2 * distance.mean(dim=[1, 2])

    return (mmd + 1e-6).sqrt().view(-1, 1)


class BEARActorLossFn(SACActorLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        policy: Policy,
        vae_decoder: VAEDecoder,
        log_temp: Parameter,
        temp_optim: Optional[OptimizerWrapper],
        log_alpha: Parameter,
        alpha_optim: Optional[OptimizerWrapper],
        action_size: int,
        n_mmd_action_samples: int,
        mmd_kernel: str,
        mmd_sigma: float,
        alpha_threshold: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            policy=policy,
            log_temp=log_temp,
            temp_optim=temp_optim,
            action_size=action_size,
        )
        self._vae_decoder = vae_decoder
        self._log_alpha = log_alpha
        self._alpha_optim = alpha_optim
        self._n_mmd_action_samples = n_mmd_action_samples
        self._mmd_kernel = mmd_kernel
        self._mmd_sigma = mmd_sigma
        self._alpha_threshold = alpha_threshold

    def _compute_mmd_loss(self, obs_t: TorchObservation) -> torch.Tensor:
        mmd = compute_mmd(
            x=obs_t,
            policy=self._policy,
            vae_decoder=self._vae_decoder,
            action_size=self._action_size,
            n_mmd_action_samples=self._n_mmd_action_samples,
            mmd_kernel=self._mmd_kernel,
            mmd_sigma=self._mmd_sigma,
        )
        alpha = get_parameter(self._log_alpha).exp()
        return (alpha * (mmd - self._alpha_threshold)).mean()

    def update_alpha(self, mmd_loss: torch.Tensor) -> None:
        assert self._alpha_optim
        self._alpha_optim.zero_grad()
        loss = -mmd_loss
        loss.backward(retain_graph=True)
        self._alpha_optim.step()
        # clip for stability
        get_parameter(self._log_alpha).data.clamp_(-5.0, 10.0)

    def __call__(self, batch: TorchMiniBatch) -> SACActorLoss:
        loss = super().__call__(batch)
        mmd_loss = self._compute_mmd_loss(batch.observations)
        if self._alpha_optim:
            self.update_alpha(mmd_loss)
        return BEARActorLoss(
            actor_loss=loss.actor_loss + mmd_loss,
            temp_loss=loss.temp_loss,
            temp=loss.temp,
            mmd_loss=mmd_loss,
            alpha=get_parameter(self._log_alpha).exp()[0][0],
        )


class BEARWarmupActorLossFn:
    def __init__(
        self,
        policy: Policy,
        vae_decoder: VAEDecoder,
        log_alpha: Parameter,
        action_size: int,
        n_mmd_action_samples: int,
        mmd_kernel: str,
        mmd_sigma: float,
        alpha_threshold: float,
    ):
        self._policy = policy
        self._action_size = action_size
        self._vae_decoder = vae_decoder
        self._log_alpha = log_alpha
        self._n_mmd_action_samples = n_mmd_action_samples
        self._mmd_kernel = mmd_kernel
        self._mmd_sigma = mmd_sigma
        self._alpha_threshold = alpha_threshold

    def _compute_mmd_loss(self, obs_t: TorchObservation) -> torch.Tensor:
        mmd = compute_mmd(
            x=obs_t,
            policy=self._policy,
            vae_decoder=self._vae_decoder,
            action_size=self._action_size,
            n_mmd_action_samples=self._n_mmd_action_samples,
            mmd_kernel=self._mmd_kernel,
            mmd_sigma=self._mmd_sigma,
        )
        alpha = get_parameter(self._log_alpha).exp()
        return (alpha * (mmd - self._alpha_threshold)).mean()

    def __call__(self, batch: TorchMiniBatch) -> torch.Tensor:
        return self._compute_mmd_loss(batch.observations)


class BEARSquashedGaussianContinuousActionSampler(ActionSampler):
    def __init__(
        self,
        policy: Policy,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        n_action_samples: int,
        action_size: int,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._n_action_samples = n_action_samples
        self._action_size = action_size

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        batch_size = (
            x.shape[0] if isinstance(x, torch.Tensor) else x[0].shape[0]
        )
        with torch.no_grad():
            # (batch, n, action)
            dist = build_squashed_gaussian_distribution(self._policy(x))
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


class BEARUpdater(SACUpdater):
    def __init__(
        self,
        q_funcs: nn.ModuleList,
        targ_q_funcs: nn.ModuleList,
        critic_optim: OptimizerWrapper,
        actor_optim: OptimizerWrapper,
        imitator_optim: OptimizerWrapper,
        critic_loss_fn: DDPGBaseCriticLossFn,
        actor_loss_fn: DDPGBaseActorLossFn,
        warmup_actor_loss_fn: BEARWarmupActorLossFn,
        imitator_loss_fn: VAELossFn,
        tau: float,
        warmup_steps: int,
        compiled: bool,
    ):
        super().__init__(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            critic_optim=critic_optim,
            actor_optim=actor_optim,
            critic_loss_fn=critic_loss_fn,
            actor_loss_fn=actor_loss_fn,
            tau=tau,
            compiled=compiled,
        )
        self._warmup_actor_loss_fn = warmup_actor_loss_fn
        self._imitator_optim = imitator_optim
        self._imitator_loss_fn = imitator_loss_fn
        self._warmup_steps = warmup_steps
        self._compute_warmup_actor_grad = (
            CudaGraphWrapper(self.compute_warmup_actor_grad)
            if compiled
            else self.compute_warmup_actor_grad
        )
        self._compute_imitator_grad = (
            CudaGraphWrapper(self.compute_imitator_grad)
            if compiled
            else self.compute_imitator_grad
        )

    def compute_warmup_actor_grad(self, batch: TorchMiniBatch) -> torch.Tensor:
        self._actor_optim.zero_grad()
        return self._warmup_actor_loss_fn(batch)

    def compute_imitator_grad(self, batch: TorchMiniBatch) -> torch.Tensor:
        self._imitator_optim.zero_grad()
        return self._imitator_loss_fn(batch)

    def __call__(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        metrics = {}

        # update imitator
        imitator_loss = self._compute_imitator_grad(batch)
        self._imitator_optim.step()
        metrics.update(
            {"imitator_loss": float(imitator_loss.detach().cpu().numpy())}
        )

        # update critic
        critic_loss = self._compute_critic_grad(batch)
        self._critic_optim.step()
        metrics.update(asdict_as_float(critic_loss))

        # update actor
        if grad_step < self._warmup_steps:
            actor_loss = self._compute_warmup_actor_grad(batch)
            self._actor_optim.step()
            metrics.update(
                {"warmup_actor_loss": float(actor_loss.detach().cpu().numpy())}
            )
        else:
            actor_loss = self._compute_actor_grad(batch)
            self._actor_optim.step()
            metrics.update(asdict_as_float(actor_loss))

        # update target networks
        self.update_target()

        return metrics
