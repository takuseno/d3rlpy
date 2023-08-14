import dataclasses
import math
from typing import Dict

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import Shape
from ...models.builders import (
    create_categorical_policy,
    create_continuous_q_function,
    create_discrete_q_function,
    create_normal_policy,
    create_parameter,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...torch_utility import Checkpointer, TorchMiniBatch
from .base import QLearningAlgoBase
from .torch.sac_impl import DiscreteSACImpl, SACImpl

__all__ = ["SACConfig", "SAC", "DiscreteSACConfig", "DiscreteSAC"]


@dataclasses.dataclass()
class SACConfig(LearnableConfig):
    r"""Config Soft Actor-Critic algorithm.

    SAC is a DDPG-based maximum entropy RL algorithm, which produces
    state-of-the-art performance in online RL settings.
    SAC leverages twin Q functions proposed in TD3. Additionally,
    `delayed policy update` in TD3 is also implemented, which is not done in
    the paper.

    .. math::

        L(\theta_i) = \mathbb{E}_{s_t,\, a_t,\, r_{t+1},\, s_{t+1} \sim D,\,
                                   a_{t+1} \sim \pi_\phi(\cdot|s_{t+1})} \Big[
            \big(y - Q_{\theta_i}(s_t, a_t)\big)^2\Big]

    .. math::

        y = r_{t+1} + \gamma \Big(\min_j Q_{\theta_j}(s_{t+1}, a_{t+1})
            - \alpha \log \big(\pi_\phi(a_{t+1}|s_{t+1})\big)\Big)

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
            \Big[\alpha \log (\pi_\phi (a_t|s_t))
              - \min_i Q_{\theta_i}\big(s_t, \pi_\phi(a_t|s_t)\big)\Big]

    The temperature parameter :math:`\alpha` is also automatically adjustable.

    .. math::

        J(\alpha) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
            \bigg[-\alpha \Big(\log \big(\pi_\phi(a_t|s_t)\big) + H\Big)\bigg]

    where :math:`H` is a target
    entropy, which is defined as :math:`\dim a`.

    References:
        * `Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep
          Reinforcement Learning with a Stochastic Actor.
          <https://arxiv.org/abs/1801.01290>`_
        * `Haarnoja et al., Soft Actor-Critic Algorithms and Applications.
          <https://arxiv.org/abs/1812.05905>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        temp_learning_rate (float): Learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        initial_temperature (float): Initial temperature value.
    """
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    initial_temperature: float = 1.0

    def create(self, device: DeviceArg = False) -> "SAC":
        return SAC(self, device)

    @staticmethod
    def get_type() -> str:
        return "sac"


class SAC(QLearningAlgoBase[SACImpl, SACConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        log_temp = create_parameter(
            (1, 1),
            math.log(self._config.initial_temperature),
            device=self._device,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.parameters(), lr=self._config.actor_learning_rate
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.parameters(), lr=self._config.critic_learning_rate
        )
        temp_optim = self._config.temp_optim_factory.create(
            log_temp.parameters(), lr=self._config.temp_learning_rate
        )

        checkpointer = Checkpointer(
            modules={
                "policy": policy,
                "q_func": q_funcs,
                "targ_q_func": targ_q_funcs,
                "log_temp": log_temp,
                "actor_optim": actor_optim,
                "critic_optim": critic_optim,
                "temp_optim": temp_optim,
            },
            device=self._device,
        )

        self._impl = SACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_funcs=q_funcs,
            q_func_forwarder=q_func_forwarder,
            targ_q_funcs=targ_q_funcs,
            targ_q_func_forwarder=targ_q_func_forwarder,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
            gamma=self._config.gamma,
            tau=self._config.tau,
            checkpointer=checkpointer,
            device=self._device,
        )

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        # lagrangian parameter update for SAC temperature
        if self._config.temp_learning_rate > 0:
            metrics.update(self._impl.update_temp(batch))

        metrics.update(self._impl.update_critic(batch))
        metrics.update(self._impl.update_actor(batch))
        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass()
class DiscreteSACConfig(LearnableConfig):
    r"""Config of Soft Actor-Critic algorithm for discrete action-space.

    This discrete version of SAC is built based on continuous version of SAC
    with additional modifications.

    The target state-value is calculated as expectation of all action-values.

    .. math::

        V(s_t) = \pi_\phi (s_t)^T [Q_\theta(s_t) - \alpha \log (\pi_\phi (s_t))]

    Similarly, the objective function for the temperature parameter is as
    follows.

    .. math::

        J(\alpha) = \pi_\phi (s_t)^T [-\alpha (\log(\pi_\phi (s_t)) + H)]

    Finally, the objective function for the policy function is as follows.

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D}
            [\pi_\phi(s_t)^T [\alpha \log(\pi_\phi(s_t)) - Q_\theta(s_t)]]

    References:
        * `Christodoulou, Soft Actor-Critic for Discrete Action Settings.
          <https://arxiv.org/abs/1910.07207>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        temp_learning_rate (float): Learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        n_critics (int): Number of Q functions for ensemble.
        initial_temperature (float): Initial temperature value.
    """
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 64
    gamma: float = 0.99
    n_critics: int = 2
    initial_temperature: float = 1.0
    target_update_interval: int = 8000

    def create(self, device: DeviceArg = False) -> "DiscreteSAC":
        return DiscreteSAC(self, device)

    @staticmethod
    def get_type() -> str:
        return "discrete_sac"


class DiscreteSAC(QLearningAlgoBase[DiscreteSACImpl, DiscreteSACConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        policy = create_categorical_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
        )
        log_temp = create_parameter(
            (1, 1),
            math.log(self._config.initial_temperature),
            device=self._device,
        )

        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.parameters(), lr=self._config.critic_learning_rate
        )
        actor_optim = self._config.actor_optim_factory.create(
            policy.parameters(), lr=self._config.actor_learning_rate
        )
        temp_optim = self._config.temp_optim_factory.create(
            log_temp.parameters(), lr=self._config.temp_learning_rate
        )

        checkpointer = Checkpointer(
            modules={
                "policy": policy,
                "q_func": q_funcs,
                "targ_q_func": targ_q_funcs,
                "log_temp": log_temp,
                "critic_optim": critic_optim,
                "actor_optim": actor_optim,
                "temp_optim": temp_optim,
            },
            device=self._device,
        )

        self._impl = DiscreteSACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            q_funcs=q_funcs,
            q_func_forwarder=q_func_forwarder,
            targ_q_funcs=targ_q_funcs,
            targ_q_func_forwarder=targ_q_func_forwarder,
            policy=policy,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
            gamma=self._config.gamma,
            checkpointer=checkpointer,
            device=self._device,
        )

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        # lagrangian parameter update for SAC temeprature
        if self._config.temp_learning_rate > 0:
            metrics.update(self._impl.update_temp(batch))

        metrics.update(self._impl.update_critic(batch))
        metrics.update(self._impl.update_actor(batch))

        if self._grad_step % self._config.target_update_interval == 0:
            self._impl.update_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(SACConfig)
register_learnable(DiscreteSACConfig)
