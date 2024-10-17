from typing import Optional

import pytest

from d3rlpy.algos.qlearning.iql import DiscreteIQLConfig, IQLConfig
from d3rlpy.types import Shape
from d3rlpy.models import (
    QFunctionFactory,
)

from ...models.torch.model_test import DummyEncoderFactory
from ...testing_utils import create_scaler_tuple
from .algo_test import algo_tester


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 84, 84), ((100,), (200,))]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_iql(observation_shape: Shape, scalers: Optional[str]) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = IQLConfig(
        actor_encoder_factory=DummyEncoderFactory(),
        critic_encoder_factory=DummyEncoderFactory(),
        value_encoder_factory=DummyEncoderFactory(),
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    iql = config.create()
    algo_tester(iql, observation_shape)  # type: ignore


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 84, 84), ((100,), (200,))]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_discrete_iql(
    observation_shape: Shape, 
    q_func_factory: QFunctionFactory,
    scalers: Optional[str]) -> None:
    observation_scaler, _, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = DiscreteIQLConfig(
        actor_encoder_factory=DummyEncoderFactory(),
        encoder_factory=DummyEncoderFactory(),
        value_encoder_factory=DummyEncoderFactory(),
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    iql = config.create()
    algo_tester(iql, observation_shape)  # type: ignore
