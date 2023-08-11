import dataclasses
from typing import Dict

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import Shape
from ...models.builders import create_discrete_q_function
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...torch_utility import TorchMiniBatch
from .base import QLearningAlgoBase
from .torch.dqn_impl import DQNImpl

__all__ = ["NFQConfig", "NFQ"]


@dataclasses.dataclass()
class NFQConfig(LearnableConfig):
    r"""Config of Neural Fitted Q Iteration algorithm.

    This NFQ implementation in d3rlpy is practically same as DQN, but excluding
    the target network mechanism.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \max_a Q_{\theta'}(s_{t+1}, a) - Q_\theta(s_t, a_t))^2]

    where :math:`\theta'` is the target network parameter. The target network
    parameter is synchronized every `target_update_interval` iterations.

    References:
        * `Riedmiller., Neural Fitted Q Iteration - first experiences with a
          data efficient neural reinforcement learning method.
          <https://link.springer.com/chapter/10.1007/11564096_32>`_

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
    """
    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 32
    gamma: float = 0.99
    n_critics: int = 1

    def create(self, device: DeviceArg = False) -> "NFQ":
        return NFQ(self, device)

    @staticmethod
    def get_type() -> str:
        return "nfq"


class NFQ(QLearningAlgoBase[DQNImpl, NFQConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        q_func = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )

        optim = self._config.optim_factory.create(
            q_func.parameters(), lr=self._config.learning_rate
        )

        self._impl = DQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            q_func=q_func,
            optim=optim,
            gamma=self._config.gamma,
            device=self._device,
        )

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        self._impl.update_target()
        return loss

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(NFQConfig)
