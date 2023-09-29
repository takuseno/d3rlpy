import dataclasses
import math

from .torch import SACModules
from .torch.droq_impl import DroQImpl
from .. import SACConfig
from ...base import DeviceArg, register_learnable
from ...constants import ActionSpace
from ...dataset import Shape
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
    create_parameter,
)
from ...models.encoders import EncoderFactory, VectorEncoderFactory
from .base import QLearningAlgoBase


__all__ = ["DroQConfig", "DroQ"]


@dataclasses.dataclass()
class DroQConfig(SACConfig):
    r"""TODO
    """
    critic_encoder_factory: EncoderFactory = VectorEncoderFactory(
        dropout_rate=0.9,
        use_layer_norm=True,
    )
    dropout_rate: int = 0.9

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
