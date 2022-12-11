import pytest

from d3rlpy.algos.iql import IQLConfig

from ..testing_utils import create_scaler_tuple
from .algo_test import algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_iql(
    observation_shape,
    action_size,
    scalers,
):
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers
    )
    config = IQLConfig(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    iql = config.create()
    algo_tester(
        iql, observation_shape, test_policy_copy=True, test_q_function_copy=True
    )
    algo_update_tester(
        iql,
        observation_shape,
        action_size,
        test_q_function_optim_copy=True,
        test_policy_optim_copy=True,
    )
