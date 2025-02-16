import dataclasses

import torch
from torch import nn

from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    DeterministicPolicy,
    DeterministicResidualPolicy,
    Policy,
    VAEDecoder,
    VAEEncoder,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import CudaGraphWrapper, TorchMiniBatch, soft_sync
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

__all__ = [
    "PLASCriticLossFn",
    "PLASActorLossFn",
    "PLASUpdater",
    "PLASActionSampler",
    "PLASWithPerturbationActorLossFn",
    "PLASWithPerturbationCriticLossFn",
    "PLASWithPerturbationActionSampler",
    "PLASWithPerturbationUpdater",
    "PLASModules",
    "PLASWithPerturbationModules",
]


@dataclasses.dataclass(frozen=True)
class PLASModules(DDPGBaseModules):
    policy: DeterministicPolicy
    targ_policy: DeterministicPolicy
    vae_encoder: VAEEncoder
    vae_decoder: VAEDecoder
    vae_optim: OptimizerWrapper


class PLASCriticLossFn(DDPGBaseCriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_policy: Policy,
        vae_decoder: VAEDecoder,
        gamma: float,
        lam: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
        )
        self._targ_policy = targ_policy
        self._vae_decoder = vae_decoder
        self._lam = lam

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = (
                2.0 * self._targ_policy(batch.next_observations).squashed_mu
            )
            actions = self._vae_decoder(batch.next_observations, latent_actions)
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                actions,
                "mix",
                self._lam,
            )


class PLASActorLossFn(DDPGBaseActorLossFn):
    def __init__(
        self,
        policy: Policy,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        vae_decoder: VAEDecoder,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._vae_decoder = vae_decoder

    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        action = self._policy(batch.observations)
        latent_actions = 2.0 * action.squashed_mu
        actions = self._vae_decoder(batch.observations, latent_actions)
        loss = -self._q_func_forwarder.compute_expected_q(
            batch.observations, actions, "none"
        )[0].mean()
        return DDPGBaseActorLoss(loss)


class PLASActionSampler(ActionSampler):
    def __init__(self, policy: Policy, vae_decoder: VAEDecoder):
        self._policy = policy
        self._vae_decoder = vae_decoder

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        latent_actions = 2.0 * self._policy(x).squashed_mu
        return self._vae_decoder(x, latent_actions)


class PLASUpdater(DDPGUpdater):
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
        warmup_steps: int,
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
        self._warmup_steps = warmup_steps
        self._compute_imitator_grad = (
            CudaGraphWrapper(self.compute_imitator_grad)
            if compiled
            else self.compute_imitator_grad
        )

    def compute_imitator_grad(self, batch: TorchMiniBatch) -> torch.Tensor:
        self._imitator_optim.zero_grad()
        loss = self._imitator_loss_fn(batch)
        loss.backward()
        return loss

    def __call__(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        metrics = {}
        if grad_step < self._warmup_steps:
            imitator_loss = self._compute_imitator_grad(batch)
            self._imitator_optim.step()
            metrics.update(
                {"imitator_loss": float(imitator_loss.detach().cpu())}
            )
        else:
            metrics.update(super().__call__(batch, grad_step))
        return metrics


@dataclasses.dataclass(frozen=True)
class PLASWithPerturbationModules(PLASModules):
    perturbation: DeterministicResidualPolicy
    targ_perturbation: DeterministicResidualPolicy


class PLASWithPerturbationCriticLossFn(DDPGBaseCriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_policy: Policy,
        targ_perturbation: DeterministicResidualPolicy,
        vae_decoder: VAEDecoder,
        gamma: float,
        lam: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
        )
        self._targ_policy = targ_policy
        self._targ_perturbation = targ_perturbation
        self._vae_decoder = vae_decoder
        self._lam = lam

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = (
                2.0 * self._targ_policy(batch.next_observations).squashed_mu
            )
            actions = self._vae_decoder(batch.next_observations, latent_actions)
            residual_actions = self._targ_perturbation(
                batch.next_observations, actions
            )
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                residual_actions.squashed_mu,
                reduction="mix",
                lam=self._lam,
            )


class PLASWithPerturbationActorLossFn(DDPGBaseActorLossFn):
    def __init__(
        self,
        policy: Policy,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        vae_decoder: VAEDecoder,
        perturbation: DeterministicResidualPolicy,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._vae_decoder = vae_decoder
        self._perturbation = perturbation

    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        action = self._policy(batch.observations)
        latent_actions = 2.0 * action.squashed_mu
        actions = self._vae_decoder(batch.observations, latent_actions)
        residual_actions = self._perturbation(
            batch.observations, actions
        ).squashed_mu
        q_value = self._q_func_forwarder.compute_expected_q(
            batch.observations, residual_actions, "none"
        )
        return DDPGBaseActorLoss(-q_value[0].mean())


class PLASWithPerturbationActionSampler(ActionSampler):
    def __init__(
        self,
        policy: Policy,
        vae_decoder: VAEDecoder,
        perturbation: DeterministicResidualPolicy,
    ):
        self._policy = policy
        self._vae_decoder = vae_decoder
        self._perturbation = perturbation

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        latent_actions = 2.0 * self._policy(x).squashed_mu
        actions = self._vae_decoder(x, latent_actions)
        return self._perturbation(x, actions).squashed_mu


class PLASWithPerturbationUpdater(PLASUpdater):
    def __init__(
        self,
        q_funcs: nn.ModuleList,
        targ_q_funcs: nn.ModuleList,
        policy: Policy,
        targ_policy: Policy,
        perturbation: DeterministicResidualPolicy,
        targ_perturbation: DeterministicResidualPolicy,
        critic_optim: OptimizerWrapper,
        actor_optim: OptimizerWrapper,
        imitator_optim: OptimizerWrapper,
        critic_loss_fn: DDPGBaseCriticLossFn,
        actor_loss_fn: DDPGBaseActorLossFn,
        imitator_loss_fn: VAELossFn,
        tau: float,
        warmup_steps: int,
        compiled: bool,
    ):
        super().__init__(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            policy=policy,
            targ_policy=targ_policy,
            critic_optim=critic_optim,
            actor_optim=actor_optim,
            imitator_optim=imitator_optim,
            critic_loss_fn=critic_loss_fn,
            actor_loss_fn=actor_loss_fn,
            imitator_loss_fn=imitator_loss_fn,
            tau=tau,
            warmup_steps=warmup_steps,
            compiled=compiled,
        )
        self._perturbation = perturbation
        self._targ_perturbation = targ_perturbation

    def update_target(self) -> None:
        super().update_target()
        soft_sync(
            self._targ_perturbation,
            self._perturbation,
            self._tau,
        )
