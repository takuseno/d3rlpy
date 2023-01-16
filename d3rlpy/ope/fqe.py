import dataclasses
from typing import Dict, Optional

import numpy as np

from ..algos import AlgoBase
from ..base import DeviceArg, LearnableConfig, register_learnable
from ..constants import (
    ALGO_NOT_GIVEN_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from ..dataset import Observation, Shape
from ..models.encoders import EncoderFactory, make_encoder_field
from ..models.optimizers import OptimizerFactory, make_optimizer_field
from ..models.q_functions import QFunctionFactory, make_q_func_field
from ..torch_utility import TorchMiniBatch, convert_to_torch
from .torch.fqe_impl import DiscreteFQEImpl, FQEBaseImpl, FQEImpl

__all__ = ["FQEConfig", "FQE", "DiscreteFQE"]


@dataclasses.dataclass(frozen=True)
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
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
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
        target_update_interval (int): interval to update the target network.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler):
            reward preprocessor.

    """
    learning_rate: float = 1e-4
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 100
    gamma: float = 0.99
    n_critics: int = 1
    target_update_interval: int = 100

    def create(self, device: DeviceArg = False) -> "_FQEBase":
        raise NotImplementedError(
            "Config object must be directly given to constructor"
        )

    @staticmethod
    def get_type() -> str:
        return "fqe"


class _FQEBase(AlgoBase):

    _algo: Optional[AlgoBase]
    _config: FQEConfig
    _impl: Optional[FQEBaseImpl]

    def __init__(
        self,
        algo: AlgoBase,
        config: FQEConfig,
        device: DeviceArg = False,
        impl: Optional[FQEBaseImpl] = None,
    ):
        super().__init__(config, device, impl)
        self._algo = algo

    def save_policy(self, fname: str) -> None:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        self._algo.save_policy(fname)

    def predict(self, x: Observation) -> np.ndarray:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        return self._algo.predict(x)

    def sample_action(self, x: Observation) -> np.ndarray:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        return self._algo.sample_action(x)

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._algo is not None, ALGO_NOT_GIVEN_ERROR
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert batch.numpy_batch
        next_actions = self._algo.predict(batch.numpy_batch.next_observations)
        loss = self._impl.update(
            batch, convert_to_torch(next_actions, self._device)
        )
        if self._grad_step % self._config.target_update_interval == 0:
            self._impl.update_target()
        return {"loss": loss}


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
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        config (d3rlpy.ope.FQEConfig): FQE config.
        device (bool, int or str):
            flag to use GPU, device ID or PyTorch device identifier.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    """

    _impl: Optional[FQEImpl]

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        self._impl = FQEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._config.learning_rate,
            optim_factory=self._config.optim_factory,
            encoder_factory=self._config.encoder_factory,
            q_func_factory=self._config.q_func_factory,
            gamma=self._config.gamma,
            n_critics=self._config.n_critics,
            device=self._device,
        )
        self._impl.build()

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
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        config (d3rlpy.ope.FQEConfig): FQE config.
        device (bool, int or str):
            flag to use GPU, device ID or PyTorch device identifier.
        impl (d3rlpy.metrics.ope.torch.DiscreteFQEImpl):
            algorithm implementation.

    """

    _impl: Optional[DiscreteFQEImpl]

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        self._impl = DiscreteFQEImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._config.learning_rate,
            optim_factory=self._config.optim_factory,
            encoder_factory=self._config.encoder_factory,
            q_func_factory=self._config.q_func_factory,
            gamma=self._config.gamma,
            n_critics=self._config.n_critics,
            device=self._device,
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(FQEConfig)
