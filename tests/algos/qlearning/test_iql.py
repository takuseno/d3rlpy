from typing import Optional, Sequence

import pytest

from d3rlpy.algos.qlearning.iql import IQLConfig

from ...testing_utils import create_scaler_tuple
from .algo_test import algo_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_iql(observation_shape: Sequence[int], scalers: Optional[str]) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers
    )
    config = IQLConfig(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    iql = config.create()
    algo_tester(iql, observation_shape)  # type: ignore
