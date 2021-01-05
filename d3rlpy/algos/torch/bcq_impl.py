import numpy as np
import torch
import math

from typing import Optional, Sequence, cast
from torch.optim import Optimizer
from ...models.torch import PixelEncoder
from ...models.torch import DeterministicResidualPolicy
from ...models.torch import create_deterministic_residual_policy
from ...models.torch import compute_max_with_n_actions
from ...models.torch import create_conditional_vae, create_discrete_imitator
from ...models.torch import ConditionalVAE, DiscreteImitator
from ...optimizers import OptimizerFactory
from ...encoders import EncoderFactory
from ...q_functions import QFunctionFactory
from ...preprocessing import Scaler
from ...augmentation import AugmentationPipeline
from ...gpu import Device
from ...torch_utility import torch_api, train_api
from .ddpg_impl import DDPGBaseImpl
from .dqn_impl import DoubleDQNImpl


class BCQImpl(DDPGBaseImpl):

    _imitator_learning_rate: float
    _imitator_optim_factory: OptimizerFactory
    _imitator_encoder_factory: EncoderFactory
    _lam: float
    _n_action_samples: int
    _action_flexibility: float
    _latent_size: int
    _beta: float
    _policy: Optional[DeterministicResidualPolicy]
    _targ_policy: Optional[DeterministicResidualPolicy]
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
        lam: float,
        n_action_samples: int,
        action_flexibility: float,
        latent_size: int,
        beta: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
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
            use_gpu=use_gpu,
            scaler=scaler,
            augmentation=augmentation,
        )
        self._imitator_learning_rate = imitator_learning_rate
        self._imitator_optim_factory = imitator_optim_factory
        self._imitator_encoder_factory = imitator_encoder_factory
        self._n_critics = n_critics
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._action_flexibility = action_flexibility
        self._latent_size = latent_size
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
        self._policy = create_deterministic_residual_policy(
            self._observation_shape,
            self._action_size,
            self._action_flexibility,
            self._actor_encoder_factory,
        )

    def _build_imitator(self) -> None:
        self._imitator = create_conditional_vae(
            self._observation_shape,
            self._action_size,
            self._latent_size,
            self._beta,
            self._imitator_encoder_factory,
        )

    def _build_imitator_optim(self) -> None:
        assert self._imitator is not None
        self._imitator_optim = self._imitator_optim_factory.create(
            self._imitator.parameters(), lr=self._imitator_learning_rate
        )

    def _compute_actor_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        assert self._q_func is not None
        latent = torch.randn(
            obs_t.shape[0], self._latent_size, device=self._device
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        sampled_action = self._imitator.decode(obs_t, clipped_latent)
        action = self._policy(obs_t, sampled_action)
        return -self._q_func(obs_t, action, "none")[0].mean()

    @train_api
    @torch_api(scaler_targets=["obs_t"])
    def update_imitator(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> np.ndarray:
        assert self._imitator_optim is not None
        assert self._imitator is not None

        self._imitator_optim.zero_grad()

        loss = self._augmentation.process(
            func=self._imitator.compute_error,
            inputs={"x": obs_t, "action": act_t},
            targets=["x"],
        )

        loss.backward()
        self._imitator_optim.step()

        return loss.cpu().detach().numpy()

    def _repeat_observation(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, *obs_shape) -> (batch_size, n, *obs_shape)
        repeat_shape = (x.shape[0], self._n_action_samples, *x.shape[1:])
        repeated_x = x.view(x.shape[0], 1, *x.shape[1:]).expand(repeat_shape)
        return repeated_x

    def _sample_action(
        self, repeated_x: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        assert self._targ_policy is not None
        # TODO: this seems to be slow with image observation
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # sample latent variable
        latent = torch.randn(
            flattened_x.shape[0], self._latent_size, device=self._device
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        # sample action
        sampled_action = self._imitator.decode(flattened_x, clipped_latent)
        # add residual action
        policy = self._targ_policy if target else self._policy
        action = policy(flattened_x, sampled_action)
        return action.view(-1, self._n_action_samples, self._action_size)

    def _predict_value(
        self,
        repeated_x: torch.Tensor,
        action: torch.Tensor,
        target: bool = False,
    ) -> torch.Tensor:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        # TODO: this seems to be slow with image observation
        # (batch_size, n, *obs_shape) -> (batch_size * n, *obs_shape)
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # (batch_size, n, action_size) -> (batch_size * n, action_size)
        flattend_action = action.view(-1, self.action_size)
        # estimate values
        q_func = self._targ_q_func if target else self._q_func
        return q_func(flattened_x, flattend_action, "none")

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        repeated_x = self._repeat_observation(x)
        action = self._sample_action(repeated_x)
        values = self._predict_value(repeated_x, action)[0]
        # pick the best (batch_size * n) -> (batch_size,)
        index = values.view(-1, self._n_action_samples).argmax(dim=1)
        return action[torch.arange(action.shape[0]), index]

    def sample_action(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("BCQ does not support sampling action")

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._targ_q_func is not None
        # TODO: this seems to be slow with image observation
        with torch.no_grad():
            repeated_x = self._repeat_observation(x)
            actions = self._sample_action(repeated_x, True)

            values = compute_max_with_n_actions(
                x, actions, self._targ_q_func, self._lam
            )

            return values


class DiscreteBCQImpl(DoubleDQNImpl):

    _action_flexibility: float
    _beta: float
    _imitator: Optional[DiscreteImitator]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        bootstrap: bool,
        share_encoder: bool,
        action_flexibility: float,
        beta: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            n_critics=n_critics,
            bootstrap=bootstrap,
            share_encoder=share_encoder,
            use_gpu=use_gpu,
            scaler=scaler,
            augmentation=augmentation,
        )
        self._action_flexibility = action_flexibility
        self._beta = beta

        # initialized in build
        self._imitator = None

    def _build_network(self) -> None:
        assert self._q_func is not None
        super()._build_network()
        # share convolutional layers if observation is pixel
        if isinstance(self._q_func.q_funcs[0].encoder, PixelEncoder):
            self._imitator = DiscreteImitator(
                self._q_func.q_funcs[0].encoder, self._action_size, self._beta
            )
        else:
            self._imitator = create_discrete_imitator(
                self._observation_shape,
                self._action_size,
                self._beta,
                self._encoder_factory,
            )

    def _build_optim(self) -> None:
        assert self._q_func is not None
        assert self._imitator is not None
        q_func_params = list(self._q_func.parameters())
        imitator_params = list(self._imitator.parameters())
        # retrieve unique elements
        unique_params = list(set(q_func_params + imitator_params))
        self._optim = self._optim_factory.create(
            unique_params, lr=self._learning_rate
        )

    def _compute_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        assert self._imitator is not None
        loss = super()._compute_loss(obs_t, act_t, rew_tpn, q_tpn, n_steps)
        imitator_loss = self._imitator.compute_error(obs_t, act_t.long())
        return loss + imitator_loss

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._q_func is not None
        log_probs = self._imitator(x)
        ratio = log_probs - log_probs.max(dim=1, keepdim=True).values
        mask = (ratio > math.log(self._action_flexibility)).float()
        value = self._q_func(x)
        normalized_value = value - value.min(dim=1, keepdim=True).values
        action = (normalized_value * cast(torch.Tensor, mask)).argmax(dim=1)
        return action
