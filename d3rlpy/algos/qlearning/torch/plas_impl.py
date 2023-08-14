import copy
from typing import Dict

import torch
from torch import nn
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
from .ddpg_impl import DDPGBaseImpl

__all__ = ["PLASImpl", "PLASWithPerturbationImpl"]


class PLASImpl(DDPGBaseImpl):
    _lam: float
    _beta: float
    _policy: DeterministicPolicy
    _targ_policy: DeterministicPolicy
    _imitator: ConditionalVAE
    _imitator_optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: DeterministicPolicy,
        q_funcs: nn.ModuleList,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_funcs: nn.ModuleList,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        imitator: ConditionalVAE,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        imitator_optim: Optimizer,
        gamma: float,
        tau: float,
        lam: float,
        beta: float,
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
            device=device,
        )
        self._lam = lam
        self._beta = beta
        self._imitator = imitator
        self._imitator_optim = imitator_optim

    @train_api
    def update_imitator(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._imitator_optim.zero_grad()

        loss = compute_vae_error(
            vae=self._imitator,
            x=batch.observations,
            action=batch.actions,
            beta=self._beta,
        )

        loss.backward()
        self._imitator_optim.step()

        return {"imitator_loss": float(loss.cpu().detach().numpy())}

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        latent_actions = 2.0 * self._policy(batch.observations).squashed_mu
        actions = forward_vae_decode(
            self._imitator, batch.observations, latent_actions
        )
        return -self._q_func_forwarder.compute_expected_q(
            batch.observations, actions, "none"
        )[0].mean()

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        latent_actions = 2.0 * self._policy(x).squashed_mu
        return forward_vae_decode(self._imitator, x, latent_actions)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = (
                2.0 * self._targ_policy(batch.next_observations).squashed_mu
            )
            actions = forward_vae_decode(
                self._imitator, batch.next_observations, latent_actions
            )
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                actions,
                "mix",
                self._lam,
            )


class PLASWithPerturbationImpl(PLASImpl):
    _perturbation: DeterministicResidualPolicy
    _targ_perturbation: DeterministicResidualPolicy

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: DeterministicPolicy,
        q_funcs: nn.ModuleList,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_funcs: nn.ModuleList,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        imitator: ConditionalVAE,
        perturbation: DeterministicResidualPolicy,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        imitator_optim: Optimizer,
        gamma: float,
        tau: float,
        lam: float,
        beta: float,
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
            imitator=imitator,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            imitator_optim=imitator_optim,
            gamma=gamma,
            tau=tau,
            lam=lam,
            beta=beta,
            device=device,
        )
        self._perturbation = perturbation
        self._targ_perturbation = copy.deepcopy(perturbation)

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        latent_actions = 2.0 * self._policy(batch.observations).squashed_mu
        actions = forward_vae_decode(
            self._imitator, batch.observations, latent_actions
        )
        residual_actions = self._perturbation(
            batch.observations, actions
        ).squashed_mu
        q_value = self._q_func_forwarder.compute_expected_q(
            batch.observations, residual_actions, "none"
        )
        return -q_value[0].mean()

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        latent_actions = 2.0 * self._policy(x).squashed_mu
        actions = forward_vae_decode(self._imitator, x, latent_actions)
        return self._perturbation(x, actions).squashed_mu

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = (
                2.0 * self._targ_policy(batch.next_observations).squashed_mu
            )
            actions = forward_vae_decode(
                self._imitator, batch.next_observations, latent_actions
            )
            residual_actions = self._targ_perturbation(
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
        soft_sync(self._targ_perturbation, self._perturbation, self._tau)
