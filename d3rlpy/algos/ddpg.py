import dataclasses
from typing import Dict, Optional

from ..argument_utility import UseGPUArg
from ..base import LearnableConfig, register_learnable
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import Shape, TransitionMiniBatch
from ..models.encoders import EncoderFactory, make_encoder_field
from ..models.optimizers import OptimizerFactory, make_optimizer_field
from ..models.q_functions import QFunctionFactory, make_q_func_field
from .base import AlgoBase
from .torch.ddpg_impl import DDPGImpl

__all__ = ["DDPGConfig", "DDPG"]


@dataclasses.dataclass(frozen=True)
class DDPGConfig(LearnableConfig):
    r"""Config of Deep Deterministic Policy Gradients algorithm.

    DDPG is an actor-critic algorithm that trains a Q function parametrized
    with :math:`\theta` and a policy function parametrized with :math:`\phi`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t,\, a_t,\, r_{t+1},\, s_{t+1} \sim D} \Big[(r_{t+1}
            + \gamma Q_{\theta'}\big(s_{t+1}, \pi_{\phi'}(s_{t+1}))
            - Q_\theta(s_t, a_t)\big)^2\Big]

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D} \Big[Q_\theta\big(s_t, \pi_\phi(s_t)\big)\Big]

    where :math:`\theta'` and :math:`\phi` are the target network parameters.
    There target network parameters are updated every iteration.

    .. math::

        \theta' \gets \tau \theta + (1 - \tau) \theta'

        \phi' \gets \tau \phi + (1 - \tau) \phi'

    References:
        * `Silver et al., Deterministic policy gradient algorithms.
          <http://proceedings.mlr.press/v32/silver14.html>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q function.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): reward preprocessor.
    """
    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    tau: float = 0.005
    n_critics: int = 1

    def create(self, use_gpu: UseGPUArg = False) -> "DDPG":
        return DDPG(self, use_gpu)

    @staticmethod
    def get_type() -> str:
        return "ddpg"


class DDPG(AlgoBase):
    _config: DDPGConfig
    _impl: Optional[DDPGImpl]

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        self._impl = DDPGImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._config.actor_learning_rate,
            critic_learning_rate=self._config.critic_learning_rate,
            actor_optim_factory=self._config.actor_optim_factory,
            critic_optim_factory=self._config.critic_optim_factory,
            actor_encoder_factory=self._config.actor_encoder_factory,
            critic_encoder_factory=self._config.critic_encoder_factory,
            q_func_factory=self._config.q_func_factory,
            gamma=self._config.gamma,
            tau=self._config.tau,
            n_critics=self._config.n_critics,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
            use_gpu=self._use_gpu,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        critic_loss = self._impl.update_critic(batch)
        actor_loss = self._impl.update_actor(batch)
        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return {"critic_loss": critic_loss, "actor_loss": actor_loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(DDPGConfig)
