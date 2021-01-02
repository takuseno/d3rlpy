import numpy as np

from typing import Any, List, Optional, Sequence, Union
from .base import AlgoBase
from .torch.bc_impl import BCImpl, DiscreteBCImpl
from ..augmentation import AugmentationPipeline
from ..dataset import TransitionMiniBatch
from ..dynamics.base import DynamicsBase
from ..encoders import EncoderFactory
from ..gpu import Device
from ..optimizers import OptimizerFactory, AdamFactory
from ..argument_utils import check_encoder, check_use_gpu, check_augmentation
from ..argument_utils import EncoderArg, UseGPUArg, AugmentationArg, ScalerArg


class BC(AlgoBase):
    r"""Behavior Cloning algorithm.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [(a_t - \pi_\theta(s_t))^2]

    Args:
        learning_rate (float): learing rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.bc_impl.BCImpl):
            implemenation of the algorithm.

    Attributes:
        learning_rate (float): learing rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory): encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.bc_impl.BCImpl):
            implemenation of the algorithm.
        eval_results_ (dict): evaluation results.

    """

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _augmentation: AugmentationPipeline
    _use_gpu: Optional[Device]
    _impl: Optional[BCImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        batch_size: int = 100,
        n_frames: int = 1,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        augmentation: AugmentationArg = None,
        dynamics: Optional[DynamicsBase] = None,
        impl: Optional[BCImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=1,
            gamma=1.0,
            scaler=scaler,
            dynamics=dynamics,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = check_encoder(encoder_factory)
        self._augmentation = check_augmentation(augmentation)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = BCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()

    def update(
        self, epoch: int, itr: int, batch: TransitionMiniBatch
    ) -> List[float]:
        assert self._impl is not None
        loss = self._impl.update_imitator(batch.observations, batch.actions)
        return [loss]

    def predict_value(
        self,
        x: Union[np.ndarray, list],
        action: Union[np.ndarray, list],
        with_std: bool = False,
    ) -> np.ndarray:
        """value prediction is not supported by BC algorithms."""
        raise NotImplementedError("BC does not support value estimation.")

    def sample_action(self, x: Union[np.ndarray, list]) -> None:
        """sampling action is not supported by BC algorithm."""
        raise NotImplementedError("BC does not support sampling action.")

    def _get_loss_labels(self) -> List[str]:
        return ["loss"]


class DiscreteBC(BC):
    r"""Behavior Cloning algorithm for discrete control.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [-\sum_a p(a|s_t) \log \pi_\theta(a|s_t)]

    where :math:`p(a|s_t)` is implemented as a one-hot vector.

    Args:
        learning_rate (float): learing rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        beta (float): reguralization factor.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.bc_impl.DiscreteBCImpl):
            implemenation of the algorithm.

    Attributes:
        learning_rate (float): learing rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory): encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        beta (float): reguralization factor.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.bc_impl.DiscreteBCImpl):
            implemenation of the algorithm.
        eval_results_ (dict): evaluation results.

    """

    _beta: float
    _impl: Optional[DiscreteBCImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        batch_size: int = 100,
        n_frames: int = 1,
        beta: float = 0.5,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        augmentation: AugmentationArg = None,
        dynamics: Optional[DynamicsBase] = None,
        impl: Optional[DiscreteBCImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            use_gpu=use_gpu,
            scaler=scaler,
            augmentation=augmentation,
            dynamics=dynamics,
            impl=impl,
            **kwargs
        )
        self._beta = beta

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DiscreteBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            beta=self._beta,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()
