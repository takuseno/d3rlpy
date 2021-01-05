import numpy as np
import torch
import copy

from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Union
from torch.optim import Optimizer
from ...optimizers import OptimizerFactory
from ...encoders import EncoderFactory
from ...q_functions import QFunctionFactory
from ...preprocessing import Scaler
from ...gpu import Device
from ...models.torch import create_continuous_q_function
from ...models.torch import create_discrete_q_function
from ...models.torch import EnsembleQFunction
from ...models.torch import EnsembleDiscreteQFunction
from ...models.torch import EnsembleContinuousQFunction
from ...torch_utility import torch_api, train_api, eval_api, hard_sync
from ...algos.torch.utility import ContinuousQFunctionMixin
from ...algos.torch.utility import DiscreteQFunctionMixin
from ...algos.torch.base import TorchImplBase


class FQEBaseImpl(TorchImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _n_critics: int
    _bootstrap: bool
    _share_encoder: bool
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleQFunction]
    _targ_q_func: Optional[EnsembleQFunction]
    _optim: Optional[Optimizer]

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
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
    ):
        super().__init__(observation_shape, action_size, scaler)
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._n_critics = n_critics
        self._bootstrap = bootstrap
        self._share_encoder = share_encoder
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._targ_q_func = None
        self._optim = None

    def build(self) -> None:
        self._build_network()

        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        self._build_optim()

    @abstractmethod
    def _build_network(self) -> None:
        pass

    def _build_optim(self) -> None:
        assert self._q_func is not None
        self._optim = self._optim_factory.create(
            self._q_func.parameters(), lr=self._learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["obs_t", "obs_tpn"])
    def update(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        act_tpn: torch.Tensor,
        obs_tpn: torch.Tensor,
        ter_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> np.ndarray:
        assert self._optim is not None

        q_tpn = self.compute_target(obs_tpn, act_tpn)
        q_tpn *= 1.0 - ter_tpn
        loss = self._compute_loss(obs_t, act_t, rew_tpn, q_tpn, n_steps)

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()

    def _compute_loss(
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

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            return self._targ_q_func.compute_target(x, action)

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        raise NotImplementedError

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)

    def save_policy(self, fname: str, as_onnx: bool) -> None:
        raise NotImplementedError


class FQEImpl(ContinuousQFunctionMixin, FQEBaseImpl):

    _q_func: Optional[EnsembleContinuousQFunction]
    _targ_q_func: Optional[EnsembleContinuousQFunction]

    def _build_network(self) -> None:
        self._q_func = create_continuous_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
        )


class DiscreteFQEImpl(DiscreteQFunctionMixin, FQEBaseImpl):

    _q_func: Optional[EnsembleDiscreteQFunction]
    _targ_q_func: Optional[EnsembleDiscreteQFunction]

    def _build_network(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
        )

    def _compute_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        return super()._compute_loss(
            obs_t, act_t.long(), rew_tpn, q_tpn, n_steps
        )

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return super().compute_target(x, action.long())
