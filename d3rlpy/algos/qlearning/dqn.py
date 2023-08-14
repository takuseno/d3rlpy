import dataclasses
from typing import Dict

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import Shape
from ...models.builders import create_discrete_q_function
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...torch_utility import Checkpointer, TorchMiniBatch
from .base import QLearningAlgoBase
from .torch.dqn_impl import DoubleDQNImpl, DQNImpl

__all__ = ["DQNConfig", "DQN", "DoubleDQNConfig", "DoubleDQN"]


@dataclasses.dataclass()
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
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        learning_rate (float): Learning rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        n_critics (int): Number of Q functions for ensemble.
        target_update_interval (int): Interval to update the target network.
    """
    batch_size: int = 32
    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    gamma: float = 0.99
    n_critics: int = 1
    target_update_interval: int = 8000

    def create(self, device: DeviceArg = False) -> "DQN":
        return DQN(self, device)

    @staticmethod
    def get_type() -> str:
        return "dqn"


class DQN(QLearningAlgoBase[DQNImpl, DQNConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        q_funcs, forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )

        optim = self._config.optim_factory.create(
            q_funcs.parameters(), lr=self._config.learning_rate
        )

        checkpointer = Checkpointer(
            modules={
                "q_func": q_funcs,
                "targ_q_func": targ_q_funcs,
                "optim": optim,
            },
            device=self._device,
        )

        self._impl = DQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            q_func_forwarder=forwarder,
            targ_q_func_forwarder=targ_forwarder,
            optim=optim,
            gamma=self._config.gamma,
            checkpointer=checkpointer,
            device=self._device,
        )

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        if self._grad_step % self._config.target_update_interval == 0:
            self._impl.update_target()
        return loss

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


@dataclasses.dataclass()
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
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        learning_rate (float): Learning rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        n_critics (int): Number of Q functions.
        target_update_interval (int): Interval to synchronize the target
            network.
    """
    batch_size: int = 32
    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    gamma: float = 0.99
    n_critics: int = 1
    target_update_interval: int = 8000

    def create(self, device: DeviceArg = False) -> "DoubleDQN":
        return DoubleDQN(self, device)

    @staticmethod
    def get_type() -> str:
        return "double_dqn"


class DoubleDQN(DQN):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        q_funcs, forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )

        optim = self._config.optim_factory.create(
            q_funcs.parameters(), lr=self._config.learning_rate
        )

        checkpointer = Checkpointer(
            modules={
                "q_func": q_funcs,
                "targ_q_func": targ_q_funcs,
                "optim": optim,
            },
            device=self._device,
        )

        self._impl = DoubleDQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            q_func_forwarder=forwarder,
            targ_q_func_forwarder=targ_forwarder,
            optim=optim,
            gamma=self._config.gamma,
            checkpointer=checkpointer,
            device=self._device,
        )


register_learnable(DQNConfig)
register_learnable(DoubleDQNConfig)
