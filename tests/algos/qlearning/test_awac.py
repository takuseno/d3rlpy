from typing import Optional

import pytest

from d3rlpy.algos.qlearning.awac import AWACConfig
from d3rlpy.models import (
    MeanQFunctionFactory,
    QFunctionFactory,
    QRQFunctionFactory,
)
from d3rlpy.types import Shape

from ...models.torch.model_test import DummyEncoderFactory
from ...testing_utils import create_scaler_tuple
from .algo_test import algo_tester


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 84, 84), ((100,), (200,))]
)
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_awac(
    observation_shape: Shape,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = AWACConfig(
        actor_encoder_factory=DummyEncoderFactory(),
        critic_encoder_factory=DummyEncoderFactory(),
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    awac = config.create()
    algo_tester(awac, observation_shape)  # type: ignore
