import pytest

from d3rlpy.algos.dqn import DQN, DoubleDQN
from tests import performance_test

from .algo_test import algo_cartpole_tester, algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scalers", [(None, None), ("min_max", "min_max")])
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
def test_dqn(
    observation_shape,
    action_size,
    n_critics,
    q_func_factory,
    scalers,
    target_reduction_type,
):
    scaler, reward_scaler = scalers
    dqn = DQN(
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        scaler=scaler,
        reward_scaler=reward_scaler,
        target_reduction_type=target_reduction_type,
    )
    algo_tester(dqn, observation_shape, test_q_function_copy=True)
    algo_update_tester(dqn, observation_shape, action_size, discrete=True)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_dqn_performance(q_func_factory):
    dqn = DQN(q_func_factory=q_func_factory)
    algo_cartpole_tester(dqn)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scalers", [(None, None), ("min_max", "min_max")])
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
def test_double_dqn(
    observation_shape,
    action_size,
    n_critics,
    q_func_factory,
    scalers,
    target_reduction_type,
):
    scaler, reward_scaler = scalers
    double_dqn = DoubleDQN(
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        scaler=scaler,
        reward_scaler=reward_scaler,
        target_reduction_type=target_reduction_type,
    )
    algo_tester(double_dqn, observation_shape, test_q_function_copy=True)
    algo_update_tester(double_dqn, observation_shape, action_size, True)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_double_dqn_performance(q_func_factory):
    double_dqn = DoubleDQN(q_func_factory=q_func_factory)
    algo_cartpole_tester(double_dqn)
