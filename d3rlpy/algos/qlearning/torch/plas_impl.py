import copy

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    ConditionalVAE,
    DeterministicPolicy,
    DeterministicResidualPolicy,
    EnsembleContinuousQFunction,
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
        q_func: EnsembleContinuousQFunction,
        imitator: ConditionalVAE,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        imitator_optim: Optimizer,
        gamma: float,
        tau: float,
        lam: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_func=q_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            gamma=gamma,
            tau=tau,
            device=device,
        )
        self._lam = lam
        self._imitator = imitator
        self._imitator_optim = imitator_optim

    @train_api
    def update_imitator(self, batch: TorchMiniBatch) -> float:
        self._imitator_optim.zero_grad()

        loss = self._imitator.compute_error(batch.observations, batch.actions)

        loss.backward()
        self._imitator_optim.step()

        return float(loss.cpu().detach().numpy())

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        latent_actions = 2.0 * self._policy(batch.observations)
        actions = self._imitator.decode(batch.observations, latent_actions)
        return -self._q_func(batch.observations, actions, "none")[0].mean()

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._imitator.decode(x, 2.0 * self._policy(x))

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = 2.0 * self._targ_policy(batch.next_observations)
            actions = self._imitator.decode(
                batch.next_observations, latent_actions
            )
            return self._targ_q_func.compute_target(
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
        q_func: EnsembleContinuousQFunction,
        imitator: ConditionalVAE,
        perturbation: DeterministicResidualPolicy,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        imitator_optim: Optimizer,
        gamma: float,
        tau: float,
        lam: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_func=q_func,
            imitator=imitator,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            imitator_optim=imitator_optim,
            gamma=gamma,
            tau=tau,
            lam=lam,
            device=device,
        )
        self._perturbation = perturbation
        self._targ_perturbation = copy.deepcopy(perturbation)

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        latent_actions = 2.0 * self._policy(batch.observations)
        actions = self._imitator.decode(batch.observations, latent_actions)
        residual_actions = self._perturbation(batch.observations, actions)
        q_value = self._q_func(batch.observations, residual_actions, "none")
        return -q_value[0].mean()

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        action = self._imitator.decode(x, 2.0 * self._policy(x))
        return self._perturbation(x, action)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            latent_actions = 2.0 * self._targ_policy(batch.next_observations)
            actions = self._imitator.decode(
                batch.next_observations, latent_actions
            )
            residual_actions = self._targ_perturbation(
                batch.next_observations, actions
            )
            return self._targ_q_func.compute_target(
                batch.next_observations,
                residual_actions,
                reduction="mix",
                lam=self._lam,
            )

    def update_actor_target(self) -> None:
        super().update_actor_target()
        soft_sync(self._targ_perturbation, self._perturbation, self._tau)
