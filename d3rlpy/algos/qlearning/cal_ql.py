import dataclasses
import math

from ...base import DeviceArg, register_learnable
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
    create_parameter,
)
from ...types import Shape
from .cql import CQL, CQLConfig
from .torch.cal_ql_impl import CalQLImpl
from .torch.cql_impl import CQLModules

__all__ = ["CalQLConfig", "CalQL"]


@dataclasses.dataclass()
class CalQLConfig(CQLConfig):
    r"""Config of Calibrated Q-Learning algorithm.

    Cal-QL is an extension to CQL to mitigate issues in offline-to-online
    fine-tuning.

    The CQL regularizer is modified as follows:

    .. math::

        \mathbb{E}_{s \sim D, a \sim \pi} [\max{(Q(s, a), V(s))}]
          - \mathbb{E}_{s, a \sim D} [Q(s, a)]

    References:
        * `Mitsuhiko et al., Cal-QL: Calibrated Offline RL Pre-Training for
          Efficient Online Fine-Tuning. <https://arxiv.org/abs/2303.05479>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        temp_learning_rate (float):
            Learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): Learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        alpha_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for :math:`\alpha`.
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
        initial_alpha (float): Initial :math:`\alpha` value.
        alpha_threshold (float): Threshold value described as :math:`\tau`.
        conservative_weight (float): Constant weight to scale conservative loss.
        n_action_samples (int): Number of sampled actions to compute
            :math:`\log{\sum_a \exp{Q(s, a)}}`.
        soft_q_backup (bool): Flag to use SAC-style backup.
    """

    def create(self, device: DeviceArg = False) -> "CalQL":
        return CalQL(self, device)

    @staticmethod
    def get_type() -> str:
        return "cal_ql"


class CalQL(CQL):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
        )
        q_funcs, q_func_fowarder = create_continuous_q_function(
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
        log_alpha = create_parameter(
            (1, 1), math.log(self._config.initial_alpha), device=self._device
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(), lr=self._config.actor_learning_rate
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(), lr=self._config.critic_learning_rate
        )
        if self._config.temp_learning_rate > 0:
            temp_optim = self._config.temp_optim_factory.create(
                log_temp.named_modules(), lr=self._config.temp_learning_rate
            )
        else:
            temp_optim = None
        if self._config.alpha_learning_rate > 0:
            alpha_optim = self._config.alpha_optim_factory.create(
                log_alpha.named_modules(), lr=self._config.alpha_learning_rate
            )
        else:
            alpha_optim = None

        modules = CQLModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            log_alpha=log_alpha,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
            alpha_optim=alpha_optim,
        )

        self._impl = CalQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_fowarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            alpha_threshold=self._config.alpha_threshold,
            conservative_weight=self._config.conservative_weight,
            n_action_samples=self._config.n_action_samples,
            soft_q_backup=self._config.soft_q_backup,
            device=self._device,
        )

    @property
    def need_returns_to_go(self) -> bool:
        return True


register_learnable(CalQLConfig)
