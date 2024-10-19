import dataclasses
from typing import Optional

from ..algos.qlearning import QLearningAlgoBase, QLearningAlgoImplBase
from ..base import DeviceArg, LearnableConfig, register_learnable
from ..constants import ALGO_NOT_GIVEN_ERROR, ActionSpace
from ..models.builders import (
    create_continuous_q_function,
    create_discrete_q_function,
)
from ..models.encoders import EncoderFactory, make_encoder_field
from ..models.q_functions import QFunctionFactory, make_q_func_field
from ..optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ..types import NDArray, Observation, Shape
from .torch.fqe_impl import (
    DiscreteFQEImpl,
    FQEBaseImpl,
    FQEBaseModules,
    FQEImpl,
)

__all__ = ["FQEConfig", "FQE", "DiscreteFQE"]


@dataclasses.dataclass()
class FQEConfig(LearnableConfig):
    r"""Config of Fitted Q Evaluation.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \sim D}
            [(Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.qlearning.base.QLearningAlgoBase):
            Algorithm to evaluate.
        learning_rate (float): Learning rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        n_critics (int): Number of Q functions for ensemble.
        target_update_interval (int): Interval to update the target network.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
    """

    learning_rate: float = 1e-4
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 100
    gamma: float = 0.99
    n_critics: int = 1
    target_update_interval: int = 100

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "_FQEBase":
        raise NotImplementedError(
            "Config object must be directly given to constructor"
        )

    @staticmethod
    def get_type() -> str:
        return "fqe"


class _FQEBase(QLearningAlgoBase[FQEBaseImpl, FQEConfig]):
    _algo: QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]
    _config: FQEConfig
    _impl: Optional[FQEBaseImpl]

    def __init__(
        self,
        algo: QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig],
        config: FQEConfig,
        device: DeviceArg = False,
        enable_ddp: bool = False,
        impl: Optional[FQEBaseImpl] = None,
    ):
        super().__init__(config, device, enable_ddp, impl)
        self._algo = algo

    def save_policy(self, fname: str) -> None:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        self._algo.save_policy(fname)

    def predict(self, x: Observation) -> NDArray:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        return self._algo.predict(x)

    def sample_action(self, x: Observation) -> NDArray:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        return self._algo.sample_action(x)

    @property
    def algo(self) -> QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]:
        return self._algo


class FQE(_FQEBase):
    r"""Fitted Q Evaluation.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \sim D}
            [(Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): Algorithm to evaluate.
        config (d3rlpy.ope.FQEConfig): FQE config.
        device (bool, int or str):
            Flag to use GPU, device ID or PyTorch device identifier.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): Algorithm implementation.
    """

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        assert self._algo.impl, "The target algorithm is not initialized."

        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        optim = self._config.optim_factory.create(
            q_funcs.named_modules(), lr=self._config.learning_rate
        )

        modules = FQEBaseModules(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            optim=optim,
        )

        self._impl = FQEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            algo=self._algo.impl,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            target_update_interval=self._config.target_update_interval,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


class DiscreteFQE(_FQEBase):
    r"""Fitted Q Evaluation for discrete action-space.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \sim D}
            [(Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.qlearning.base.QLearningAlgoBase):
            Algorithm to evaluate.
        config (d3rlpy.ope.FQEConfig): FQE config.
        device (bool, int or str):
            Flag to use GPU, device ID or PyTorch device identifier.
        impl (d3rlpy.metrics.ope.torch.DiscreteFQEImpl):
            Algorithm implementation.
    """

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        assert self._algo.impl, "The target algorithm is not initialized."

        q_funcs, q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        optim = self._config.optim_factory.create(
            q_funcs.named_modules(), lr=self._config.learning_rate
        )
        modules = FQEBaseModules(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            optim=optim,
        )
        self._impl = DiscreteFQEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            algo=self._algo.impl,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            target_update_interval=self._config.target_update_interval,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(FQEConfig)
