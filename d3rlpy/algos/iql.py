import dataclasses
from typing import Dict, Optional

from ..base import LearnableConfig, UseGPUArg, register_learnable
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import Shape, TransitionMiniBatch
from ..models.encoders import EncoderFactory, make_encoder_field
from ..models.optimizers import OptimizerFactory, make_optimizer_field
from .base import AlgoBase
from .torch.iql_impl import IQLImpl

__all__ = ["IQLConfig", "IQL"]


@dataclasses.dataclass(frozen=True)
class IQLConfig(LearnableConfig):
    r"""Implicit Q-Learning algorithm.

    IQL is the offline RL algorithm that avoids ever querying values of unseen
    actions while still being able to perform multi-step dynamic programming
    updates.

    There are three functions to train in IQL. First the state-value function
    is trained via expectile regression.

    .. math::

        L_V(\psi) = \mathbb{E}_{(s, a) \sim D}
            [L_2^\tau (Q_\theta (s, a) - V_\psi (s))]

    where :math:`L_2^\tau (u) = |\tau - \mathbb{1}(u < 0)|u^2`.

    The Q-function is trained with the state-value function to avoid query the
    actions.

    .. math::

        L_Q(\theta) = \mathbb{E}_{(s, a, r, a') \sim D}
            [(r + \gamma V_\psi(s') - Q_\theta(s, a))^2]

    Finally, the policy function is trained by using advantage weighted
    regression.

    .. math::

        L_\pi (\phi) = \mathbb{E}_{(s, a) \sim D}
            [\exp(\beta (Q_\theta - V_\psi(s))) \log \pi_\phi(a|s)]

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the critic.
        value_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the value function.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        expectile (float): the expectile value for value function training.
        weight_temp (float): inverse temperature value represented as
            :math:`\beta`.
        max_weight (float): the maximum advantage weight value to clip.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): reward preprocessor.
    """
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    value_encoder_factory: EncoderFactory = make_encoder_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    expectile: float = 0.7
    weight_temp: float = 3.0
    max_weight: float = 100.0

    def create(self, use_gpu: UseGPUArg = False) -> "IQL":
        return IQL(self, use_gpu)

    @staticmethod
    def get_type() -> str:
        return "iql"


class IQL(AlgoBase):
    _config: IQLConfig
    _impl: Optional[IQLImpl]

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        self._impl = IQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._config.actor_learning_rate,
            critic_learning_rate=self._config.critic_learning_rate,
            actor_optim_factory=self._config.actor_optim_factory,
            critic_optim_factory=self._config.critic_optim_factory,
            actor_encoder_factory=self._config.actor_encoder_factory,
            critic_encoder_factory=self._config.critic_encoder_factory,
            value_encoder_factory=self._config.value_encoder_factory,
            gamma=self._config.gamma,
            tau=self._config.tau,
            n_critics=self._config.n_critics,
            expectile=self._config.expectile,
            weight_temp=self._config.weight_temp,
            max_weight=self._config.max_weight,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
            use_gpu=self._use_gpu,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        critic_loss, value_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss, "value_loss": value_loss})

        actor_loss = self._impl.update_actor(batch)
        metrics.update({"actor_loss": actor_loss})

        self._impl.update_critic_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(IQLConfig)
