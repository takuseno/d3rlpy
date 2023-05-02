import dataclasses
import math
from typing import Dict

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import Shape
from ...models.builders import (
    create_continuous_q_function,
    create_discrete_q_function,
    create_parameter,
    create_squashed_normal_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...torch_utility import TorchMiniBatch
from .base import QLearningAlgoBase
from .torch.cql_impl import CQLImpl, DiscreteCQLImpl

__all__ = ["CQLConfig", "CQL", "DiscreteCQLConfig", "DiscreteCQL"]


@dataclasses.dataclass()
class CQLConfig(LearnableConfig):
    r"""Config of Conservative Q-Learning algorithm.

    CQL is a SAC-based data-driven deep reinforcement learning algorithm, which
    achieves state-of-the-art performance in offline RL problems.

    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.

    .. math::

        L(\theta_i) = \alpha\, \mathbb{E}_{s_t \sim D}
            \left[\log{\sum_a \exp{Q_{\theta_i}(s_t, a)}}
             - \mathbb{E}_{a \sim D} \big[Q_{\theta_i}(s_t, a)\big] - \tau\right]
            + L_\mathrm{SAC}(\theta_i)

    where :math:`\alpha` is an automatically adjustable value via Lagrangian
    dual gradient descent and :math:`\tau` is a threshold value.
    If the action-value difference is smaller than :math:`\tau`, the
    :math:`\alpha` will become smaller.
    Otherwise, the :math:`\alpha` will become larger to aggressively penalize
    action-values.

    In continuous control, :math:`\log{\sum_a \exp{Q(s, a)}}` is computed as
    follows.

    .. math::

        \log{\sum_a \exp{Q(s, a)}} \approx \log{\left(
            \frac{1}{2N} \sum_{a_i \sim \text{Unif}(a)}^N
                \left[\frac{\exp{Q(s, a_i)}}{\text{Unif}(a)}\right]
            + \frac{1}{2N} \sum_{a_i \sim \pi_\phi(a|s)}^N
                \left[\frac{\exp{Q(s, a_i)}}{\pi_\phi(a_i|s)}\right]\right)}

    where :math:`N` is the number of sampled actions.

    The rest of optimization is exactly same as :class:`d3rlpy.algos.SAC`.

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float):
            learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        alpha_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for :math:`\alpha`.
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
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\alpha` value.
        alpha_threshold (float): threshold value described as :math:`\tau`.
        conservative_weight (float): constant weight to scale conservative loss.
        n_action_samples (int): the number of sampled actions to compute
            :math:`\log{\sum_a \exp{Q(s, a)}}`.
        soft_q_backup (bool): flag to use SAC-style backup.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): reward preprocessor.
    """
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 1e-4
    alpha_learning_rate: float = 1e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    alpha_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    initial_temperature: float = 1.0
    initial_alpha: float = 1.0
    alpha_threshold: float = 10.0
    conservative_weight: float = 5.0
    n_action_samples: int = 10
    soft_q_backup: bool = False

    def create(self, device: DeviceArg = False) -> "CQL":
        return CQL(self, device)

    @staticmethod
    def get_type() -> str:
        return "cql"


class CQL(QLearningAlgoBase[CQLImpl, CQLConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_squashed_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
        )
        q_func = create_continuous_q_function(
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
        log_alpha = create_parameter(
            (1, 1), math.log(self._config.initial_alpha), device=self._device
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.parameters(), lr=self._config.actor_learning_rate
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_func.parameters(), lr=self._config.critic_learning_rate
        )
        temp_optim = self._config.temp_optim_factory.create(
            log_temp.parameters(), lr=self._config.temp_learning_rate
        )
        alpha_optim = self._config.alpha_optim_factory.create(
            log_alpha.parameters(), lr=self._config.alpha_learning_rate
        )

        self._impl = CQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_func=q_func,
            log_temp=log_temp,
            log_alpha=log_alpha,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
            alpha_optim=alpha_optim,
            gamma=self._config.gamma,
            tau=self._config.tau,
            alpha_threshold=self._config.alpha_threshold,
            conservative_weight=self._config.conservative_weight,
            n_action_samples=self._config.n_action_samples,
            soft_q_backup=self._config.soft_q_backup,
            device=self._device,
        )

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        # lagrangian parameter update for SAC temperature
        if self._config.temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(batch)
            metrics.update({"temp_loss": temp_loss, "temp": temp})

        # lagrangian parameter update for conservative loss weight
        if self._config.alpha_learning_rate > 0:
            alpha_loss, alpha = self._impl.update_alpha(batch)
            metrics.update({"alpha_loss": alpha_loss, "alpha": alpha})

        critic_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss})

        actor_loss = self._impl.update_actor(batch)
        metrics.update({"actor_loss": actor_loss})

        self._impl.update_critic_target()
        self._impl.update_actor_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass()
class DiscreteCQLConfig(LearnableConfig):
    r"""Config of Discrete version of Conservative Q-Learning algorithm.

    Discrete version of CQL is a DoubleDQN-based data-driven deep reinforcement
    learning algorithm (the original paper uses DQN), which achieves
    state-of-the-art performance in offline RL problems.

    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.

    .. math::

        L(\theta) = \alpha \mathbb{E}_{s_t \sim D}
            [\log{\sum_a \exp{Q_{\theta}(s_t, a)}}
             - \mathbb{E}_{a \sim D} [Q_{\theta}(s, a)]]
            + L_{DoubleDQN}(\theta)

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

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
        target_update_interval (int): interval to synchronize the target
            network.
        alpha (float): the :math:`\alpha` value above.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): reward preprocessor.
    """
    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 32
    gamma: float = 0.99
    n_critics: int = 1
    target_update_interval: int = 8000
    alpha: float = 1.0

    def create(self, device: DeviceArg = False) -> "DiscreteCQL":
        return DiscreteCQL(self, device)

    @staticmethod
    def get_type() -> str:
        return "discrete_cql"


class DiscreteCQL(QLearningAlgoBase[DiscreteCQLImpl, DiscreteCQLConfig]):
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

        self._impl = DiscreteCQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            q_func=q_func,
            optim=optim,
            gamma=self._config.gamma,
            alpha=self._config.alpha,
            device=self._device,
        )

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch)
        if self._grad_step % self._config.target_update_interval == 0:
            self._impl.update_target()
        return {"loss": loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(CQLConfig)
register_learnable(DiscreteCQLConfig)
