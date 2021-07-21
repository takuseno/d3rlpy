import pytest

from d3rlpy.algos.awac import AWAC
from tests import performance_test

from .algo_test import algo_pendulum_tester, algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize(
    "scalers", [(None, None, None), ("min_max", "min_max", "min_max")]
)
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
def test_awac(
    observation_shape,
    action_size,
    q_func_factory,
    scalers,
    target_reduction_type,
):
    scaler, action_scaler, reward_scaler = scalers
    awac = AWAC(
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        target_reduction_type=target_reduction_type,
    )
    algo_tester(
        awac,
        observation_shape,
        test_policy_copy=True,
        test_q_function_copy=True,
    )
    algo_update_tester(awac, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_awac_performance(q_func_factory):
    awac = AWAC(q_func_factory=q_func_factory)
    algo_pendulum_tester(awac, n_trials=1)
