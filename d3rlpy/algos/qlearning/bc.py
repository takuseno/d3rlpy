import dataclasses
from typing import Optional

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_categorical_policy,
    create_deterministic_policy,
    create_normal_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.bc_impl import (
    BCBaseImpl,
    BCImpl,
    BCModules,
    DiscreteBCImpl,
    DiscreteBCModules,
)

__all__ = ["BCConfig", "BC", "DiscreteBCConfig", "DiscreteBC"]


@dataclasses.dataclass()
class BCConfig(LearnableConfig):
    r"""Config of Behavior Cloning algorithm.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [(a_t - \pi_\theta(s_t))^2]

    Args:
        learning_rate (float): Learing rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        batch_size (int): Mini-batch size.
        policy_type (str): the policy type. Available options are
            ``['deterministic', 'stochastic']``.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    batch_size: int = 100
    learning_rate: float = 1e-3
    policy_type: str = "deterministic"
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "BC":
        return BC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "bc"


class BC(QLearningAlgoBase[BCBaseImpl, BCConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        if self._config.policy_type == "deterministic":
            imitator = create_deterministic_policy(
                observation_shape,
                action_size,
                self._config.encoder_factory,
                device=self._device,
                enable_ddp=self._enable_ddp,
            )
        elif self._config.policy_type == "stochastic":
            imitator = create_normal_policy(
                observation_shape,
                action_size,
                self._config.encoder_factory,
                min_logstd=-4.0,
                max_logstd=15.0,
                device=self._device,
                enable_ddp=self._enable_ddp,
            )
        else:
            raise ValueError(f"invalid policy_type: {self._config.policy_type}")

        optim = self._config.optim_factory.create(
            imitator.named_modules(),
            lr=self._config.learning_rate,
            compiled=self.compiled,
        )

        modules = BCModules(optim=optim, imitator=imitator)

        self._impl = BCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            policy_type=self._config.policy_type,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass()
class DiscreteBCConfig(LearnableConfig):
    r"""Config of Behavior Cloning algorithm for discrete control.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [-\sum_a p(a|s_t) \log \pi_\theta(a|s_t)]

    where :math:`p(a|s_t)` is implemented as a one-hot vector.

    Args:
        learning_rate (float): Learing rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        batch_size (int): Mini-batch size.
        beta (float): Reguralization factor.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    batch_size: int = 100
    learning_rate: float = 1e-3
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    beta: float = 0.5
    entropy_beta: float = 0.0
    embedding_size: Optional[int] = None
    automatic_mixed_precision: bool = False
    scheduler_on_train_step: bool = True

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DiscreteBC":
        return DiscreteBC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "discrete_bc"


class DiscreteBC(QLearningAlgoBase[BCBaseImpl, DiscreteBCConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        imitator = create_categorical_policy(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
            embedding_size=self._config.embedding_size,
        )

        optim = self._config.optim_factory.create(
            imitator.named_modules(),
            lr=self._config.learning_rate,
            compiled=self.compiled,
        )

        modules = DiscreteBCModules(optim=optim, imitator=imitator)

        self._impl = DiscreteBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            beta=self._config.beta,
            entropy_beta=self._config.entropy_beta,
            compiled=self.compiled,
            device=self._device,
            automatic_mixed_precision=self._config.automatic_mixed_precision,
            scheduler_on_train_step=self._config.scheduler_on_train_step,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(BCConfig)
register_learnable(DiscreteBCConfig)
