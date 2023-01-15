import dataclasses
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..base import DeviceArg, LearnableConfig, register_learnable
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import Shape, TransitionMiniBatch
from ..models.encoders import EncoderFactory, make_encoder_field
from ..models.optimizers import OptimizerFactory, make_optimizer_field
from .base import AlgoBase
from .torch.bc_impl import BCBaseImpl, BCImpl, DiscreteBCImpl

__all__ = ["BCConfig", "BC", "DiscreteBCConfig", "DiscreteBC"]


class _BCBase(AlgoBase):
    _impl: Optional[BCBaseImpl]

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update_imitator(batch.observations, batch.actions)
        return {"loss": loss}

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> np.ndarray:
        """value prediction is not supported by BC algorithms."""
        raise NotImplementedError("BC does not support value estimation.")

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> None:
        """sampling action is not supported by BC algorithm."""
        raise NotImplementedError("BC does not support sampling action.")


@dataclasses.dataclass(frozen=True)
class BCConfig(LearnableConfig):
    r"""Config of Behavior Cloning algorithm.

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
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory.
        batch_size (int): mini-batch size.
        policy_type (str): the policy type. The available options are
            ``['deterministic', 'stochastic']``.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): action scaler.
    """
    batch_size: int = 100
    learning_rate: float = 1e-3
    policy_type: str = "deterministic"
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()

    def create(self, device: DeviceArg = False) -> "BC":
        return BC(self, device)

    @staticmethod
    def get_type() -> str:
        return "bc"


class BC(_BCBase):

    _config: BCConfig
    _impl: Optional[BCImpl]

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        self._impl = BCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._config.learning_rate,
            optim_factory=self._config.optim_factory,
            encoder_factory=self._config.encoder_factory,
            policy_type=self._config.policy_type,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            device=self._device,
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass(frozen=True)
class DiscreteBCConfig(LearnableConfig):
    r"""Config of Behavior Cloning algorithm for discrete control.

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
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory.
        batch_size (int): mini-batch size.
        beta (float): reguralization factor.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
    """
    batch_size: int = 100
    learning_rate: float = 1e-3
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    beta: float = 0.5

    def create(self, device: DeviceArg = False) -> "DiscreteBC":
        return DiscreteBC(self, device)

    @staticmethod
    def get_type() -> str:
        return "discrete_bc"


class DiscreteBC(_BCBase):
    _config: DiscreteBCConfig
    _impl: Optional[DiscreteBCImpl]

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        self._impl = DiscreteBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._config.learning_rate,
            optim_factory=self._config.optim_factory,
            encoder_factory=self._config.encoder_factory,
            beta=self._config.beta,
            observation_scaler=self._config.observation_scaler,
            device=self._device,
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(BCConfig)
register_learnable(DiscreteBCConfig)
