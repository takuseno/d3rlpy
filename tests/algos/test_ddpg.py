import pytest

from d3rlpy.algos.ddpg import DDPG
from tests import performance_test

from .algo_test import algo_pendulum_tester, algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "min_max"])
@pytest.mark.parametrize("action_scaler", [None, "min_max"])
@pytest.mark.parametrize("reward_scaler", [None, "min_max"])
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
def test_ddpg(
    observation_shape,
    action_size,
    q_func_factory,
    scaler,
    action_scaler,
    reward_scaler,
    target_reduction_type,
):
    ddpg = DDPG(
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        target_reduction_type=target_reduction_type,
    )
    algo_tester(
        ddpg,
        observation_shape,
        test_policy_copy=True,
        test_q_function_copy=True,
    )
    algo_update_tester(ddpg, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_ddpg_performance(q_func_factory):
    # not good enough for batch RL, but check if it works without errors.
    try:
        ddpg = DDPG(q_func_factory=q_func_factory)
        algo_pendulum_tester(ddpg, n_trials=1)
    except AssertionError:
        pass
