import pytest

from d3rlpy.algos.bear import BEAR
from tests import performance_test

from .algo_test import algo_pendulum_tester, algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "min_max"])
@pytest.mark.parametrize("action_scaler", [None, "min_max"])
@pytest.mark.parametrize("reward_scaler", [None, "min_max"])
def test_bear(
    observation_shape,
    action_size,
    q_func_factory,
    scaler,
    action_scaler,
    reward_scaler,
):
    bear = BEAR(
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    algo_tester(
        bear,
        observation_shape,
        test_q_function_copy=True,
    )
    algo_update_tester(bear, observation_shape, action_size)


@pytest.mark.skip(reason="BEAR is computationally expensive.")
def test_bear_performance():
    bear = BEAR()
    algo_pendulum_tester(bear, n_trials=3)
