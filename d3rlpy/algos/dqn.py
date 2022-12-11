import dataclasses
from typing import Dict, Optional

from ..argument_utility import UseGPUArg
from ..base import ImplBase, LearnableConfig, register_learnable
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import Shape, TransitionMiniBatch
from ..models.encoders import EncoderFactory, make_encoder_field
from ..models.optimizers import OptimizerFactory, make_optimizer_field
from ..models.q_functions import QFunctionFactory, make_q_func_field
from .base import AlgoBase
from .torch.dqn_impl import DoubleDQNImpl, DQNImpl

__all__ = ["DQNConfig", "DQN", "DoubleDQNConfig", "DoubleDQN"]


@dataclasses.dataclass(frozen=True)
class DQNConfig(LearnableConfig):
    r"""Config of Deep Q-Network algorithm.

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
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        target_update_interval (int): interval to update the target network.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): reward preprocessor.
    """
    batch_size: int = 32
    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    gamma: float = 0.99
    n_critics: int = 1
    target_update_interval: int = 8000

    def create(
        self, use_gpu: UseGPUArg = False, impl: Optional[ImplBase] = None
    ) -> "DQN":
        return DQN(self, use_gpu, impl)

    @staticmethod
    def get_type() -> str:
        return "dqn"


class DQN(AlgoBase):
    _config: DQNConfig
    _impl: Optional[DQNImpl]

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        self._impl = DQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._config.learning_rate,
            optim_factory=self._config.optim_factory,
            encoder_factory=self._config.encoder_factory,
            q_func_factory=self._config.q_func_factory,
            gamma=self._config.gamma,
            n_critics=self._config.n_critics,
            use_gpu=self._use_gpu,
            observation_scaler=self._config.observation_scaler,
            reward_scaler=self._config.reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        if self._grad_step % self._config.target_update_interval == 0:
            self._impl.update_target()
        return {"loss": loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


@dataclasses.dataclass(frozen=True)
class DoubleDQNConfig(DQNConfig):
    r"""Config of Double Deep Q-Network algorithm.

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
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions.
        target_update_interval (int): interval to synchronize the target
            network.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
    """
    batch_size: int = 32
    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    gamma: float = 0.99
    n_critics: int = 1
    target_update_interval: int = 8000

    def create(
        self, use_gpu: UseGPUArg = False, impl: Optional[ImplBase] = None
    ) -> "DoubleDQN":
        return DoubleDQN(self, use_gpu, impl)

    @staticmethod
    def get_type() -> str:
        return "double_dqn"


class DoubleDQN(DQN):
    _config: DoubleDQNConfig
    _impl: Optional[DoubleDQNImpl]

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        self._impl = DoubleDQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._config.learning_rate,
            optim_factory=self._config.optim_factory,
            encoder_factory=self._config.encoder_factory,
            q_func_factory=self._config.q_func_factory,
            gamma=self._config.gamma,
            n_critics=self._config.n_critics,
            observation_scaler=self._config.observation_scaler,
            reward_scaler=self._config.reward_scaler,
            use_gpu=self._use_gpu,
        )
        self._impl.build()


register_learnable(DQNConfig)
register_learnable(DoubleDQNConfig)
