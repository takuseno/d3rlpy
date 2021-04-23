from typing import Any, Dict, Optional, Sequence

from ..argument_utility import (
    AugmentationArg,
    EncoderArg,
    QFuncArg,
    ScalerArg,
    UseGPUArg,
    check_augmentation,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from ..augmentation import AugmentationPipeline
from ..constants import IMPL_NOT_INITIALIZED_ERROR
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.dqn_impl import DoubleDQNImpl, DQNImpl


class DQN(AlgoBase):
    r"""Deep Q-Network algorithm.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \max_a Q_{\theta'}(s_{t+1}, a) - Q_\theta(s_t, a_t))^2]

    where :math:`\theta'` is the target network parameter. The target network
    parameter is synchronized every `target_update_interval` iterations.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_

    Args:
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
        target_reduction_type (str): ensemble reduction method at target value
            estimation. The available options are
            ``['min', 'max', 'mean', 'mix', 'none']``.
        target_update_interval (int): interval to update the target network.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        impl (d3rlpy.algos.torch.dqn_impl.DQNImpl): algorithm implementation.

    """

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _n_critics: int
    _target_reduction_type: str
    _target_update_interval: int
    _augmentation: AugmentationPipeline
    _use_gpu: Optional[Device]
    _impl: Optional[DQNImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 6.25e-5,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 32,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_critics: int = 1,
        target_reduction_type: str = "min",
        target_update_interval: int = 8000,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        augmentation: AugmentationArg = None,
        impl: Optional[DQNImpl] = None,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=None,
            kwargs=kwargs,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder(encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._n_critics = n_critics
        self._target_reduction_type = target_reduction_type
        self._target_update_interval = target_update_interval
        self._augmentation = check_augmentation(augmentation)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            target_reduction_type=self._target_reduction_type,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()

    def update(
        self, epoch: int, total_step: int, batch: TransitionMiniBatch
    ) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(
            batch.observations,
            batch.actions,
            batch.next_rewards,
            batch.next_observations,
            batch.terminals,
            batch.n_steps,
            batch.masks,
        )
        if total_step % self._target_update_interval == 0:
            self._impl.update_target()
        return {"loss": loss}


class DoubleDQN(DQN):
    r"""Double Deep Q-Network algorithm.

    The difference from DQN is that the action is taken from the current Q
    function instead of the target Q function.
    This modification significantly decreases overestimation bias of TD
    learning.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma Q_{\theta'}(s_{t+1}, \text{argmax}_a
            Q_\theta(s_{t+1}, a)) - Q_\theta(s_t, a_t))^2]

    where :math:`\theta'` is the target network parameter. The target network
    parameter is synchronized every `target_update_interval` iterations.

    References:
        * `Hasselt et al., Deep reinforcement learning with double Q-learning.
          <https://arxiv.org/abs/1509.06461>`_

    Args:
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions.
        target_reduction_type (str): ensemble reduction method at target value
            estimation. The available options are
            ``['min', 'max', 'mean', 'mix', 'none']``.
        target_update_interval (int): interval to synchronize the target
            network.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        impl (d3rlpy.algos.torch.dqn_impl.DoubleDQNImpl):
            algorithm implementation.

    """

    _impl: Optional[DoubleDQNImpl]

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DoubleDQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            target_reduction_type=self._target_reduction_type,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()
