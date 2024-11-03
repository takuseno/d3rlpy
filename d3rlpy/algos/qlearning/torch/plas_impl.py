import dataclasses
from typing import Callable

import torch

from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    DeterministicPolicy,
    DeterministicResidualPolicy,
    VAEDecoder,
    VAEEncoder,
    compute_vae_error,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import CudaGraphWrapper, TorchMiniBatch, soft_sync
from ....types import Shape, TorchObservation
from .ddpg_impl import DDPGBaseActorLoss, DDPGBaseImpl, DDPGBaseModules

__all__ = [
    "PLASImpl",
    "PLASWithPerturbationImpl",
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


class PLASImpl(DDPGBaseImpl):
    _modules: PLASModules
    _compute_imitator_grad: Callable[[TorchMiniBatch], dict[str, torch.Tensor]]
    _lam: float
    _beta: float
    _warmup_steps: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: PLASModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        lam: float,
        beta: float,
        warmup_steps: int,
        compiled: bool,
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
            compiled=compiled,
            device=device,
        )
        self._lam = lam
        self._beta = beta
        self._warmup_steps = warmup_steps
        self._compute_imitator_grad = (
            CudaGraphWrapper(self.compute_imitator_grad)
            if compiled
            else self.compute_imitator_grad
        )

    def compute_imitator_grad(
        self, batch: TorchMiniBatch
    ) -> dict[str, torch.Tensor]:
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

    def update_imitator(self, batch: TorchMiniBatch) -> dict[str, float]:
        loss = self._compute_imitator_grad(batch)
        self._modules.vae_optim.step()
        return {"vae_loss": float(loss["loss"].cpu().detach().numpy())}

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> DDPGBaseActorLoss:
        latent_actions = 2.0 * action.squashed_mu
        actions = self._modules.vae_decoder(batch.observations, latent_actions)
        loss = -self._q_func_forwarder.compute_expected_q(
            batch.observations, actions, "none"
        )[0].mean()
        return DDPGBaseActorLoss(loss)

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        latent_actions = 2.0 * self._modules.policy(x).squashed_mu
        return self._modules.vae_decoder(x, latent_actions)

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = (
                2.0
                * self._modules.targ_policy(batch.next_observations).squashed_mu
            )
            actions = self._modules.vae_decoder(
                batch.next_observations, latent_actions
            )
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                actions,
                "mix",
                self._lam,
            )

    def update_actor_target(self) -> None:
        soft_sync(self._modules.targ_policy, self._modules.policy, self._tau)

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        metrics = {}

        if grad_step < self._warmup_steps:
            metrics.update(self.update_imitator(batch))
        else:
            metrics.update(self.update_critic(batch))
            metrics.update(self.update_actor(batch))
            self.update_actor_target()
            self.update_critic_target()

        return metrics


@dataclasses.dataclass(frozen=True)
class PLASWithPerturbationModules(PLASModules):
    perturbation: DeterministicResidualPolicy
    targ_perturbation: DeterministicResidualPolicy


class PLASWithPerturbationImpl(PLASImpl):
    _modules: PLASWithPerturbationModules

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: PLASWithPerturbationModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        lam: float,
        beta: float,
        warmup_steps: int,
        compiled: bool,
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
            lam=lam,
            beta=beta,
            warmup_steps=warmup_steps,
            compiled=compiled,
            device=device,
        )

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> DDPGBaseActorLoss:
        latent_actions = 2.0 * action.squashed_mu
        actions = self._modules.vae_decoder(batch.observations, latent_actions)
        residual_actions = self._modules.perturbation(
            batch.observations, actions
        ).squashed_mu
        q_value = self._q_func_forwarder.compute_expected_q(
            batch.observations, residual_actions, "none"
        )
        return DDPGBaseActorLoss(-q_value[0].mean())

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        latent_actions = 2.0 * self._modules.policy(x).squashed_mu
        actions = self._modules.vae_decoder(x, latent_actions)
        return self._modules.perturbation(x, actions).squashed_mu

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = (
                2.0
                * self._modules.targ_policy(batch.next_observations).squashed_mu
            )
            actions = self._modules.vae_decoder(
                batch.next_observations, latent_actions
            )
            residual_actions = self._modules.targ_perturbation(
                batch.next_observations, actions
            )
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                residual_actions.squashed_mu,
                reduction="mix",
                lam=self._lam,
            )

    def update_actor_target(self) -> None:
        super().update_actor_target()
        soft_sync(
            self._modules.targ_perturbation,
            self._modules.perturbation,
            self._tau,
        )
