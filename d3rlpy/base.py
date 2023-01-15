import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Union, cast

import gym
from gym.spaces import Discrete

from .constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from .dataset import DatasetInfo, ReplayBuffer, Shape, TransitionMiniBatch
from .gpu import Device
from .logger import LOG, D3RLPyLogger
from .preprocessing import (
    ActionScaler,
    ObservationScaler,
    RewardScaler,
    make_action_scaler_field,
    make_observation_scaler_field,
    make_reward_scaler_field,
)
from .serializable_config import DynamicConfig, generate_config_registration

__all__ = [
    "UseGPUArg",
    "ImplBase",
    "save_config",
    "LearnableBase",
    "LearnableConfig",
    "LearnableConfigWithShape",
    "register_learnable",
]


UseGPUArg = Optional[Union[bool, int, Device]]


class ImplBase(metaclass=ABCMeta):
    @abstractmethod
    def save_model(self, fname: str) -> None:
        pass

    @abstractmethod
    def load_model(self, fname: str) -> None:
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> Shape:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        pass


@dataclasses.dataclass(frozen=True)
class LearnableConfig(DynamicConfig):
    batch_size: int = 256
    gamma: float = 0.99
    observation_scaler: Optional[
        ObservationScaler
    ] = make_observation_scaler_field()
    action_scaler: Optional[ActionScaler] = make_action_scaler_field()
    reward_scaler: Optional[RewardScaler] = make_reward_scaler_field()

    def create(self, use_gpu: UseGPUArg = False) -> "LearnableBase":
        raise NotImplementedError


register_learnable, make_learnable_field = generate_config_registration(
    LearnableConfig
)


@dataclasses.dataclass(frozen=True)
class LearnableConfigWithShape(DynamicConfig):
    observation_shape: Shape
    action_size: int
    config: LearnableConfig = make_learnable_field()

    def create(self, use_gpu: UseGPUArg = False) -> "LearnableBase":
        algo = self.config.create(use_gpu)
        algo.create_impl(self.observation_shape, self.action_size)
        return algo


def save_config(alg: "LearnableBase", logger: D3RLPyLogger) -> None:
    assert alg.impl
    config = LearnableConfigWithShape(
        observation_shape=alg.impl.observation_shape,
        action_size=alg.impl.action_size,
        config=alg.config,
    )
    logger.add_params(config.serialize_to_dict())


def _process_use_gpu(value: UseGPUArg) -> Optional[Device]:
    """Checks value and returns Device object.

    Returns:
        d3rlpy.gpu.Device: device object.

    """
    # isinstance cannot tell difference between bool and int
    if isinstance(value, bool):
        if value:
            return Device(0)
        return None
    if isinstance(value, int):
        return Device(value)
    if isinstance(value, Device):
        return value
    if value is None:
        return None
    raise ValueError("This argument must be bool, int or Device.")


class LearnableBase:
    _config: LearnableConfig
    _use_gpu: Optional[Device]
    _impl: Optional[ImplBase]
    _grad_step: int

    def __init__(
        self,
        config: LearnableConfig,
        use_gpu: UseGPUArg,
        impl: Optional[ImplBase] = None,
    ):
        self._config = config
        self._use_gpu = _process_use_gpu(use_gpu)
        self._impl = impl
        self._grad_step = 0

    def save_model(self, fname: str) -> None:
        """Saves neural network parameters.

        .. code-block:: python

            algo.save_model('model.pt')

        Args:
            fname: destination file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        self._impl.save_model(fname)

    def load_model(self, fname: str) -> None:
        """Load neural network parameters.

        .. code-block:: python

            algo.load_model('model.pt')

        Args:
            fname: source file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        self._impl.load_model(fname)

    @classmethod
    def from_json(
        cls, fname: str, use_gpu: UseGPUArg = False
    ) -> "LearnableBase":
        config = LearnableConfigWithShape.deserialize_from_file(fname)
        return config.create(use_gpu)

    def create_impl(self, observation_shape: Shape, action_size: int) -> None:
        """Instantiate implementation objects with the dataset shapes.

        This method will be used internally when `fit` method is called.

        Args:
            observation_shape: observation shape.
            action_size: dimension of action-space.

        """
        if self._impl:
            LOG.warn("Parameters will be reinitialized.")
        self._create_impl(observation_shape, action_size)

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        raise NotImplementedError

    def build_with_dataset(self, dataset: ReplayBuffer) -> None:
        """Instantiate implementation object with MDPDataset object.

        Args:
            dataset: dataset.

        """
        dataset_info = DatasetInfo.from_episodes(dataset.episodes)
        observation_shape = dataset.sample_transition().observation_shape
        self.create_impl(observation_shape, dataset_info.action_size)

    def build_with_env(self, env: gym.Env) -> None:
        """Instantiate implementation object with OpenAI Gym object.

        Args:
            env: gym-like environment.

        """
        observation_shape = env.observation_space.shape
        if isinstance(env.action_space, Discrete):
            action_size = cast(int, env.action_space.n)
        else:
            action_size = cast(int, env.action_space.shape[0])
        self.create_impl(observation_shape, action_size)

    def update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: mini-batch data.

        Returns:
            dictionary of metrics.

        """
        loss = self._update(batch)
        self._grad_step += 1
        return loss

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        raise NotImplementedError

    def get_action_type(self) -> ActionSpace:
        """Returns action type (continuous or discrete).

        Returns:
            action type.

        """
        raise NotImplementedError

    @property
    def config(self) -> LearnableConfig:
        return self._config

    @property
    def batch_size(self) -> int:
        """Batch size to train.

        Returns:
            int: batch size.

        """
        return self._config.batch_size

    @property
    def gamma(self) -> float:
        """Discount factor.

        Returns:
            float: discount factor.

        """
        return self._config.gamma

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]:
        """Preprocessing observation scaler.

        Returns:
            Optional[ObservationScaler]: preprocessing observation scaler.

        """
        return self._config.observation_scaler

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        """Preprocessing action scaler.

        Returns:
            Optional[ActionScaler]: preprocessing action scaler.

        """
        return self._config.action_scaler

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        """Preprocessing reward scaler.

        Returns:
            Optional[RewardScaler]: preprocessing reward scaler.

        """
        return self._config.reward_scaler

    @property
    def impl(self) -> Optional[ImplBase]:
        """Implementation object.

        Returns:
            Optional[ImplBase]: implementation object.

        """
        return self._impl

    @property
    def observation_shape(self) -> Optional[Shape]:
        """Observation shape.

        Returns:
            Optional[Sequence[int]]: observation shape.

        """
        if self._impl:
            return self._impl.observation_shape
        return None

    @property
    def action_size(self) -> Optional[int]:
        """Action size.

        Returns:
            Optional[int]: action size.

        """
        if self._impl:
            return self._impl.action_size
        return None

    @property
    def grad_step(self) -> int:
        """Total gradient step counter.

        This value will keep counting after ``fit`` and ``fit_online``
        methods finish.

        Returns:
            total gradient step counter.

        """
        return self._grad_step

    def set_grad_step(self, grad_step: int) -> None:
        """Set total gradient step counter.

        This method can be used to restart the middle of training with an
        arbitrary gradient step counter, which has effects on periodic
        functions such as the target update.

        Args:
            grad_step: total gradient step counter.

        """
        self._grad_step = grad_step
