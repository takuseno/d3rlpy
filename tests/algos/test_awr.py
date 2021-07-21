import numpy as np
import pytest

from d3rlpy.algos.awr import AWR, DiscreteAWR
from tests import performance_test

from .algo_test import (
    algo_cartpole_tester,
    algo_pendulum_tester,
    algo_tester,
    algo_update_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize(
    "scalers", [(None, None, None), ("min_max", "min_max", "min_max")]
)
def test_awr(observation_shape, action_size, scalers):
    scaler, action_scaler, reward_scaler = scalers
    awr = AWR(
        batch_size=100,
        batch_size_per_update=30,
        n_actor_updates=1,
        n_critic_updates=1,
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    algo_tester(awr, observation_shape, state_value=True)
    algo_update_tester(awr, observation_shape, action_size)


@performance_test
def test_awr_performance():
    awr = AWR()
    algo_pendulum_tester(awr, n_trials=3)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("scalers", [(None, None), ("min_max", "min_max")])
def test_discrete_awr(observation_shape, action_size, scalers):
    scaler, reward_scaler = scalers
    awr = DiscreteAWR(
        batch_size=100,
        batch_size_per_update=30,
        n_actor_updates=1,
        n_critic_updates=1,
        scaler=scaler,
        reward_scaler=reward_scaler,
    )
    algo_tester(awr, observation_shape, state_value=True)
    algo_update_tester(awr, observation_shape, action_size, True)


@performance_test
def test_discrete_awr_performance():
    awr = DiscreteAWR()
    algo_cartpole_tester(awr, n_trials=3)
