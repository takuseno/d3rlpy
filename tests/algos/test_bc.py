import pytest

from d3rlpy.algos.bc import BC, DiscreteBC
from tests import performance_test

from .algo_test import (
    algo_cartpole_tester,
    algo_pendulum_tester,
    algo_tester,
    algo_update_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("policy_type", ["deterministic", "stochastic"])
@pytest.mark.parametrize("scalers", [(None, None), ("min_max", "min_max")])
def test_bc(observation_shape, action_size, policy_type, scalers):
    observation_scaler, action_scaler = scalers
    bc = BC(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        policy_type=policy_type,
    )
    algo_tester(bc, observation_shape, imitator=True)
    algo_update_tester(bc, observation_shape, action_size)


@performance_test
def test_bc_performance():
    bc = BC()
    algo_pendulum_tester(bc)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("observation_scaler", [None, "min_max"])
def test_discrete_bc(observation_shape, action_size, observation_scaler):
    bc = DiscreteBC(observation_scaler=observation_scaler)
    algo_tester(bc, observation_shape, imitator=True)
    algo_update_tester(bc, observation_shape, action_size, discrete=True)


@performance_test
def test_discrete_bc_performance():
    bc = DiscreteBC()
    algo_cartpole_tester(bc)
