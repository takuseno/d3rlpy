import pytest

from d3rlpy.algos.mopo import MOPO
from d3rlpy.dynamics import ProbabilisticEnsembleDynamics
from tests import performance_test

from .algo_test import (
    algo_cartpole_tester,
    algo_pendulum_tester,
    algo_tester,
    algo_update_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "min_max"])
@pytest.mark.parametrize("action_scaler", [None, "min_max"])
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
@pytest.mark.parametrize("rollout_interval", [1])
@pytest.mark.parametrize("horizon", [2])
@pytest.mark.parametrize("n_initial_transitions", [4])
def test_mopo(
    observation_shape,
    action_size,
    q_func_factory,
    scaler,
    action_scaler,
    target_reduction_type,
    rollout_interval,
    horizon,
    n_initial_transitions,
):
    dynamics = ProbabilisticEnsembleDynamics()
    dynamics.create_impl(observation_shape, action_size)

    mopo = MOPO(
        rollout_interval=rollout_interval,
        horizon=horizon,
        n_initial_transitions=n_initial_transitions,
        dynamics=dynamics,
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        target_reduction_type=target_reduction_type,
    )
    algo_tester(mopo, observation_shape)
    algo_update_tester(mopo, observation_shape, action_size)
