import dataclasses
from typing import Dict

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    ConditionalVAE,
    ContinuousEnsembleQFunctionForwarder,
    DeterministicPolicy,
    DeterministicResidualPolicy,
    compute_vae_error,
    forward_vae_decode,
)
from ....torch_utility import TorchMiniBatch, soft_sync, train_api
from .ddpg_impl import DDPGBaseImpl, DDPGBaseModules

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
    imitator: ConditionalVAE
    imitator_optim: Optimizer


class PLASImpl(DDPGBaseImpl):
    _modules: PLASModules
    _lam: float
    _beta: float

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
        self._lam = lam
        self._beta = beta

    @train_api
    def update_imitator(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.imitator_optim.zero_grad()

        loss = compute_vae_error(
            vae=self._modules.imitator,
            x=batch.observations,
            action=batch.actions,
            beta=self._beta,
        )

        loss.backward()
        self._modules.imitator_optim.step()

        return {"imitator_loss": float(loss.cpu().detach().numpy())}

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        latent_actions = (
            2.0 * self._modules.policy(batch.observations).squashed_mu
        )
        actions = forward_vae_decode(
            self._modules.imitator, batch.observations, latent_actions
        )
        return -self._q_func_forwarder.compute_expected_q(
            batch.observations, actions, "none"
        )[0].mean()

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        latent_actions = 2.0 * self._modules.policy(x).squashed_mu
        return forward_vae_decode(self._modules.imitator, x, latent_actions)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = (
                2.0
                * self._modules.targ_policy(batch.next_observations).squashed_mu
            )
            actions = forward_vae_decode(
                self._modules.imitator, batch.next_observations, latent_actions
            )
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                actions,
                "mix",
                self._lam,
            )

    def update_actor_target(self) -> None:
        soft_sync(self._modules.targ_policy, self._modules.policy, self._tau)


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
            device=device,
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        latent_actions = (
            2.0 * self._modules.policy(batch.observations).squashed_mu
        )
        actions = forward_vae_decode(
            self._modules.imitator, batch.observations, latent_actions
        )
        residual_actions = self._modules.perturbation(
            batch.observations, actions
        ).squashed_mu
        q_value = self._q_func_forwarder.compute_expected_q(
            batch.observations, residual_actions, "none"
        )
        return -q_value[0].mean()

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        latent_actions = 2.0 * self._modules.policy(x).squashed_mu
        actions = forward_vae_decode(self._modules.imitator, x, latent_actions)
        return self._modules.perturbation(x, actions).squashed_mu

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = (
                2.0
                * self._modules.targ_policy(batch.next_observations).squashed_mu
            )
            actions = forward_vae_decode(
                self._modules.imitator, batch.next_observations, latent_actions
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
