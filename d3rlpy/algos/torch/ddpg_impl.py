import numpy as np
import torch
import copy

from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union, cast
from torch.optim import Optimizer
from ...models.torch import create_continuous_q_function
from ...models.torch import create_deterministic_policy
from ...models.torch import EnsembleContinuousQFunction
from ...models.torch import Policy, DeterministicPolicy
from ...augmentation import AugmentationPipeline
from ...encoders import EncoderFactory
from ...gpu import Device
from ...optimizers import OptimizerFactory
from ...q_functions import QFunctionFactory
from ...preprocessing import Scaler
from ...torch_utility import soft_sync, torch_api, train_api, eval_api
from .utility import ContinuousQFunctionMixin
from .base import TorchImplBase


class DDPGBaseImpl(ContinuousQFunctionMixin, TorchImplBase, metaclass=ABCMeta):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _tau: float
    _n_critics: int
    _bootstrap: bool
    _share_encoder: bool
    _augmentation: AugmentationPipeline
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleContinuousQFunction]
    _policy: Optional[Policy]
    _targ_q_func: Optional[EnsembleContinuousQFunction]
    _targ_policy: Optional[Policy]
    _actor_optim: Optional[Optimizer]
    _critic_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        bootstrap: bool,
        share_encoder: bool,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(observation_shape, action_size, scaler)
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = actor_encoder_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._tau = tau
        self._n_critics = n_critics
        self._bootstrap = bootstrap
        self._share_encoder = share_encoder
        self._augmentation = augmentation
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._policy = None
        self._targ_q_func = None
        self._targ_policy = None
        self._actor_optim = None
        self._critic_optim = None

    def build(self) -> None:
        # setup torch models
        self._build_critic()
        self._build_actor()

        # setup target networks
        self._targ_q_func = copy.deepcopy(self._q_func)
        self._targ_policy = copy.deepcopy(self._policy)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()

    def _build_critic(self) -> None:
        self._q_func = create_continuous_q_function(
            self._observation_shape,
            self._action_size,
            self._critic_encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        self._critic_optim = self._critic_optim_factory.create(
            self._q_func.parameters(), lr=self._critic_learning_rate
        )

    @abstractmethod
    def _build_actor(self) -> None:
        pass

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["obs_t", "obs_tpn"])
    def update_critic(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        obs_tpn: torch.Tensor,
        ter_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        q_tpn = self._augmentation.process(
            func=self.compute_target, inputs={"x": obs_tpn}, targets=["x"]
        )
        q_tpn *= 1.0 - ter_tpn

        loss = self._augmentation.process(
            func=self._compute_critic_loss,
            inputs={
                "obs_t": obs_t,
                "act_t": act_t,
                "rew_tpn": rew_tpn,
                "q_tpn": q_tpn,
                "n_steps": n_steps,
            },
            targets=["obs_t"],
        )

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_critic_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            obs_t, act_t, rew_tpn, q_tpn, self._gamma ** n_steps
        )

    @train_api
    @torch_api(scaler_targets=["obs_t"])
    def update_actor(self, obs_t: torch.Tensor) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self._augmentation.process(
            func=self._compute_actor_loss,
            inputs={"obs_t": obs_t},
            targets=["obs_t"],
        )

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    @abstractmethod
    def _compute_actor_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.best_action(x)

    @eval_api
    @torch_api(scaler_targets=["obs_t"])
    def sample_action(self, x: torch.Tensor) -> np.ndarray:
        assert self._policy is not None
        with torch.no_grad():
            return self._policy.sample(x).cpu().detach().numpy()

    def update_critic_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        soft_sync(self._targ_q_func, self._q_func, self._tau)

    def update_actor_target(self) -> None:
        assert self._policy is not None
        assert self._targ_policy is not None
        soft_sync(self._targ_policy, self._policy, self._tau)


class DDPGImpl(DDPGBaseImpl):

    _policy: Optional[DeterministicPolicy]
    _targ_policy: Optional[DeterministicPolicy]

    def _build_actor(self) -> None:
        self._policy = create_deterministic_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def _compute_actor_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(obs_t)
        q_t = self._q_func(obs_t, action, "min")
        return -q_t.mean()

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._targ_q_func is not None
        assert self._targ_policy is not None
        with torch.no_grad():
            action = self._targ_policy(x)
            return self._targ_q_func.compute_target(x, action.clamp(-1.0, 1.0))

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        return self.predict_best_action(x)
