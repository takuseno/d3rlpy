import copy
from typing import Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer

from ...models.torch import (
    DeterministicResidualPolicy,
    DeterministicPolicy,
    ConditionalVAE,
)
from ...models.builders import (
    create_deterministic_policy,
    create_deterministic_residual_policy,
    create_conditional_vae,
)
from ...models.optimizers import OptimizerFactory
from ...models.encoders import EncoderFactory
from ...models.q_functions import QFunctionFactory
from ...gpu import Device
from ...preprocessing import Scaler, ActionScaler
from ...augmentation import AugmentationPipeline
from ...torch_utility import torch_api, train_api, soft_sync
from .ddpg_impl import DDPGBaseImpl


class PLASImpl(DDPGBaseImpl):

    _imitator_learning_rate: float
    _imitator_optim_factory: OptimizerFactory
    _imitator_encoder_factory: EncoderFactory
    _n_critics: int
    _lam: float
    _beta: float
    _policy: Optional[DeterministicPolicy]
    _targ_policy: Optional[DeterministicPolicy]
    _imitator: Optional[ConditionalVAE]
    _imitator_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        imitator_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        imitator_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        imitator_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        bootstrap: bool,
        share_encoder: bool,
        target_reduction_type: str,
        lam: float,
        beta: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            bootstrap=bootstrap,
            share_encoder=share_encoder,
            target_reduction_type=target_reduction_type,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            augmentation=augmentation,
        )
        self._imitator_learning_rate = imitator_learning_rate
        self._imitator_optim_factory = imitator_optim_factory
        self._imitator_encoder_factory = imitator_encoder_factory
        self._n_critics = n_critics
        self._lam = lam
        self._beta = beta

        # initialized in build
        self._imitator = None
        self._imitator_optim = None

    def build(self) -> None:
        self._build_imitator()
        super().build()
        # setup optimizer after the parameters move to GPU
        self._build_imitator_optim()

    def _build_actor(self) -> None:
        self._policy = create_deterministic_policy(
            observation_shape=self._observation_shape,
            action_size=2 * self._action_size,
            encoder_factory=self._actor_encoder_factory,
        )

    def _build_imitator(self) -> None:
        self._imitator = create_conditional_vae(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            latent_size=2 * self._action_size,
            beta=self._beta,
            encoder_factory=self._imitator_encoder_factory,
        )

    def _build_imitator_optim(self) -> None:
        assert self._imitator is not None
        self._imitator_optim = self._imitator_optim_factory.create(
            params=self._imitator.parameters(), lr=self._imitator_learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["obs_t"], action_scaler_targets=["act_t"])
    def update_imitator(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> np.ndarray:
        assert self._imitator is not None
        assert self._imitator_optim is not None

        self._imitator_optim.zero_grad()

        loss = self._augmentation.process(
            func=self._imitator.compute_error,
            inputs={"x": obs_t, "action": act_t},
            targets=["x"],
        )

        loss.backward()
        self._imitator_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_actor_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        assert self._q_func is not None
        action = self._imitator.decode(obs_t, 2.0 * self._policy(obs_t))
        return -self._q_func(obs_t, action, "none")[0].mean()

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        return self._imitator.decode(x, 2.0 * self._policy(x))

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._targ_policy is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._imitator.decode(x, 2.0 * self._targ_policy(x))
            return self._targ_q_func.compute_target(
                x, action, self._target_reduction_type, self._lam
            )


class PLASWithPerturbationImpl(PLASImpl):

    _action_flexibility: float
    _perturbation: Optional[DeterministicResidualPolicy]
    _targ_perturbation: Optional[DeterministicResidualPolicy]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        imitator_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        imitator_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        imitator_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        bootstrap: bool,
        share_encoder: bool,
        target_reduction_type: str,
        lam: float,
        beta: float,
        action_flexibility: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            imitator_learning_rate=imitator_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            imitator_optim_factory=imitator_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            imitator_encoder_factory=imitator_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            bootstrap=bootstrap,
            share_encoder=share_encoder,
            target_reduction_type=target_reduction_type,
            lam=lam,
            beta=beta,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            augmentation=augmentation,
        )
        self._action_flexibility = action_flexibility

        # initialized in build
        self._perturbation = None
        self._targ_perturbation = None

    def build(self) -> None:
        super().build()
        self._targ_perturbation = copy.deepcopy(self._perturbation)

    def _build_actor(self) -> None:
        super()._build_actor()
        self._perturbation = create_deterministic_residual_policy(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            scale=self._action_flexibility,
            encoder_factory=self._actor_encoder_factory,
        )

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        assert self._perturbation is not None
        parameters = list(self._policy.parameters())
        parameters += list(self._perturbation.parameters())
        self._actor_optim = self._actor_optim_factory.create(
            params=parameters, lr=self._actor_learning_rate
        )

    def _compute_actor_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        assert self._perturbation is not None
        assert self._q_func is not None
        action = self._imitator.decode(obs_t, 2.0 * self._policy(obs_t))
        residual_action = self._perturbation(obs_t, action)
        return -self._q_func(obs_t, residual_action, "none")[0].mean()

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        assert self._perturbation is not None
        action = self._imitator.decode(x, 2.0 * self._policy(x))
        return self._perturbation(x, action)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._targ_policy is not None
        assert self._targ_perturbation is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._imitator.decode(x, 2.0 * self._targ_policy(x))
            residual_action = self._targ_perturbation(x, action)
            return self._targ_q_func.compute_target(
                x,
                residual_action,
                reduction=self._target_reduction_type,
                lam=self._lam,
            )

    def update_actor_target(self) -> None:
        assert self._perturbation is not None
        assert self._targ_perturbation is not None
        super().update_actor_target()
        soft_sync(self._targ_perturbation, self._perturbation, self._tau)
