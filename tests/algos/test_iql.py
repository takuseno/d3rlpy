import pytest

from d3rlpy.algos.iql import IQL

from .algo_test import algo_pendulum_tester, algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize(
    "scalers", [(None, None, None), ("min_max", "min_max", "min_max")]
)
def test_iql(
    observation_shape,
    action_size,
    scalers,
):
    scaler, action_scaler, reward_scaler = scalers
    iql = IQL(
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
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


@pytest.mark.skip(reason="IQL is computationally expensive.")
def test_iql_performance():
    iql = IQL()
    algo_pendulum_tester(iql, n_trials=3)
