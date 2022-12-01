import pytest

from d3rlpy.algos.cql import CQL, DiscreteCQL
from tests import performance_test

from .algo_test import (
    algo_cartpole_tester,
    algo_pendulum_tester,
    algo_tester,
    algo_update_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize(
    "scalers", [(None, None, None), ("min_max", "min_max", "min_max")]
)
def test_cql(
    observation_shape,
    action_size,
    q_func_factory,
    scalers,
):
    scaler, action_scaler, reward_scaler = scalers
    cql = CQL(
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    algo_tester(
        cql, observation_shape, test_policy_copy=True, test_q_function_copy=True
    )
    algo_update_tester(
        cql,
        observation_shape,
        action_size,
        test_q_function_optim_copy=True,
        test_policy_optim_copy=True,
    )


@pytest.mark.skip(reason="CQL is computationally expensive.")
def test_cql_performance():
    cql = CQL()
    algo_pendulum_tester(cql, n_trials=3)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scalers", [(None, None), ("min_max", "min_max")])
def test_discrete_cql(
    observation_shape,
    action_size,
    n_critics,
    q_func_factory,
    scalers,
):
    scaler, reward_scaler = scalers
    cql = DiscreteCQL(
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        scaler=scaler,
        reward_scaler=reward_scaler,
    )
    algo_tester(cql, observation_shape, test_q_function_copy=True)
    algo_update_tester(
        cql,
        observation_shape,
        action_size,
        discrete=True,
        test_q_function_optim_copy=True,
    )


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_discrete_cql_performance(q_func_factory):
    cql = DiscreteCQL(q_func_factory=q_func_factory)
    algo_cartpole_tester(cql)
