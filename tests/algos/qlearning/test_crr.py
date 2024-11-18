from typing import Optional

import pytest

from d3rlpy.algos.qlearning.crr import CRRConfig
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
    "observation_shape", [(100,), (4, 32, 32), ((100,), (200,))]
)
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
@pytest.mark.parametrize("advantage_type", ["mean", "max"])
@pytest.mark.parametrize("weight_type", ["exp", "binary"])
@pytest.mark.parametrize("target_update_type", ["hard", "soft"])
def test_crr(
    observation_shape: Shape,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
    advantage_type: str,
    weight_type: str,
    target_update_type: str,
) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = CRRConfig(
        actor_encoder_factory=DummyEncoderFactory(),
        critic_encoder_factory=DummyEncoderFactory(),
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        advantage_type=advantage_type,
        weight_type=weight_type,
        target_update_type=target_update_type,
    )
    crr = config.create()
    algo_tester(
        crr,  # type: ignore
        observation_shape,
        deterministic_best_action=False,
        test_policy_copy=False,
    )
