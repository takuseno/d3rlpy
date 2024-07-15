import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_discrete_q_function,
    create_categorical_policy,
    create_normal_policy,
    create_value_function,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import MeanQFunctionFactory
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.iql_impl import IQLImpl, IQLModules, DiscreteIQLImpl, DiscreteIQLModules

__all__ = ["IQLConfig", "IQL", "DiscreteIQLConfig", "DiscreteIQL"]


@dataclasses.dataclass()
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

        L_Q(\theta) = \mathbb{E}_{(s, a, r, s') \sim D}
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
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        value_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the value function.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        expectile (float): Expectile value for value function training.
        weight_temp (float): Inverse temperature value represented as
            :math:`\beta`.
        max_weight (float): Maximum advantage weight value to clip.
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

    def create(self, device: DeviceArg = False) -> "IQL":
        return IQL(self, device)

    @staticmethod
    def get_type() -> str:
        return "iql"


class IQL(QLearningAlgoBase[IQLImpl, IQLConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
            device=self._device,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        value_func = create_value_function(
            observation_shape,
            self._config.value_encoder_factory,
            device=self._device,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(), lr=self._config.actor_learning_rate
        )
        q_func_params = list(q_funcs.named_modules())
        v_func_params = list(value_func.named_modules())
        critic_optim = self._config.critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._config.critic_learning_rate
        )

        modules = IQLModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            value_func=value_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
        )

        self._impl = IQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            expectile=self._config.expectile,
            weight_temp=self._config.weight_temp,
            max_weight=self._config.max_weight,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

@dataclasses.dataclass()
class DiscreteIQLConfig(LearnableConfig):
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4

    q_func_factory: QFunctionFactory = make_q_func_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    value_encoder_factory: EncoderFactory = make_encoder_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    expectile: float = 0.7
    weight_temp: float = 3.0
    max_weight: float = 100.0

    def create(self, device: DeviceArg = False) -> "DiscreteIQL":
        return DiscreteIQL(self, device)

    @staticmethod
    def get_type() -> str:
        return "discrete_iql"

class DiscreteIQL(QLearningAlgoBase[DiscreteIQLImpl, DiscreteIQLConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_categorical_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            self._device,
        )
        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        value_func = create_value_function(
            observation_shape,
            self._config.value_encoder_factory,
            device=self._device,
        )

        q_func_params = list(q_funcs.named_modules())
        v_func_params = list(value_func.named_modules())
        critic_optim = self._config.critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._config.critic_learning_rate
        )
        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(), lr=self._config.actor_learning_rate
        )

        modules = DiscreteIQLModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            value_func=value_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
        )

        self._impl = DiscreteIQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            expectile=self._config.expectile,
            weight_temp=self._config.weight_temp,
            max_weight=self._config.max_weight,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE

register_learnable(IQLConfig)
register_learnable(DiscreteIQLConfig)
