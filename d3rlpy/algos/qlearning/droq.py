import dataclasses
import math

from .torch import SACModules
from .torch.droq_impl import DroQImpl
from ...base import DeviceArg, register_learnable, LearnableConfig
from ...constants import ActionSpace
from ...dataset import Shape
from ...models import QFunctionFactory, make_q_func_field, make_optimizer_field, OptimizerFactory
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
    create_parameter,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from .base import QLearningAlgoBase


__all__ = ["DroQConfig", "DroQ"]


@dataclasses.dataclass()
class DroQConfig(LearnableConfig):
    r"""TODO
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

    def create(self, device: DeviceArg = False) -> "DroQ":
        return DroQ(self, device)

    @staticmethod
    def get_type() -> str:
        return "droq"


class DroQ(QLearningAlgoBase[DroQImpl, DroQConfig]):
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
        if self._config.temp_learning_rate > 0:
            temp_optim = self._config.temp_optim_factory.create(
                log_temp.parameters(), lr=self._config.temp_learning_rate
            )
        else:
            temp_optim = None

        modules = SACModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
        )

        self._impl = DroQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


# (TODO IF VALID) class DiscreteDroQConfig(LearnableConfig):

register_learnable(DroQConfig)
