from typing import Optional

import pytest

from d3rlpy.algos.qlearning.bear import BEARConfig
from d3rlpy.types import Shape

from ...models.torch.model_test import DummyEncoderFactory
from ...testing_utils import create_scaler_tuple
from .algo_test import algo_tester


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 32, 32), ((100,), (200,))]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_bear(
    observation_shape: Shape,
    scalers: Optional[str],
) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = BEARConfig(
        actor_encoder_factory=DummyEncoderFactory(),
        critic_encoder_factory=DummyEncoderFactory(),
        imitator_encoder_factory=DummyEncoderFactory(),
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        warmup_steps=0,
    )
    bear = config.create()
    algo_tester(
        bear,  # type: ignore
        observation_shape,
        deterministic_best_action=False,
        test_policy_copy=False,
    )
