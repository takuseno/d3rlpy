from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from ..algos import AlgoBase
from ..argument_utility import check_encoder, EncoderArg
from ..argument_utility import check_use_gpu, UseGPUArg
from ..argument_utility import check_q_func, QFuncArg
from ..argument_utility import ScalerArg, ActionScalerArg
from ..dataset import TransitionMiniBatch
from ..models.encoders import EncoderFactory
from ..models.optimizers import OptimizerFactory, AdamFactory
from ..models.q_functions import QFunctionFactory
from ..gpu import Device
from .torch.fqe_impl import FQEBaseImpl, FQEImpl, DiscreteFQEImpl


class _FQEBase(AlgoBase):

    _algo: Optional[AlgoBase]
    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _n_critics: int
    _bootstrap: bool
    _share_encoder: bool
    _target_update_interval: int
    _use_gpu: Optional[Device]
    _impl: Optional[FQEBaseImpl]

    def __init__(
        self,
        *,
        algo: Optional[AlgoBase] = None,
        learning_rate: float = 1e-4,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_critics: int = 1,
        bootstrap: bool = False,
        share_encoder: bool = False,
        target_update_interval: int = 100,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        impl: Optional[FQEBaseImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            generator=None,
        )
        self._algo = algo
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder(encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._n_critics = n_critics
        self._bootstrap = bootstrap
        self._share_encoder = share_encoder
        self._target_update_interval = target_update_interval
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def save_policy(self, fname: str, as_onnx: bool = False) -> None:
        assert self._algo is not None
        self._algo.save_policy(fname, as_onnx)

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        assert self._algo is not None
        return self._algo.predict(x)

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        assert self._algo is not None
        return self._algo.sample_action(x)

    @abstractmethod
    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        pass

    def update(
        self, epoch: int, total_step: int, batch: TransitionMiniBatch
    ) -> List[Optional[float]]:
        assert self._algo is not None
        assert self._impl is not None
        next_actions = self._algo.predict(batch.observations)
        loss = self._impl.update(
            batch.observations,
            batch.actions,
            batch.next_rewards,
            next_actions,
            batch.next_observations,
            batch.terminals,
            batch.n_steps,
        )
        if total_step % self._target_update_interval == 0:
            self._impl.update_target()
        return [loss]

    def _get_loss_labels(self) -> List[str]:
        return ["value_loss"]


class FQE(_FQEBase):
    r"""Fitted Q Evaluation.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \sim D}
            [(Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory or str):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_update_interval (int): interval to update the target network.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    """

    _impl: Optional[FQEImpl]

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = FQEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
        )
        self._impl.build()


class DiscreteFQE(_FQEBase):
    r"""Fitted Q Evaluation for discrete action-space.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \sim D}
            [(Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory or str):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_update_interval (int): interval to update the target network.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    """

    _impl: Optional[DiscreteFQEImpl]

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DiscreteFQEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=None,
        )
        self._impl.build()
