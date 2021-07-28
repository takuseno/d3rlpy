import pytest

from d3rlpy.dynamics.probabilistic_ensemble_dynamics import (
    ProbabilisticEnsembleDynamics,
)

from .dynamics_test import dynamics_tester, dynamics_update_tester


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize(
    "scalers", [(None, None, None), ("min_max", "min_max", "min_max")]
)
@pytest.mark.parametrize("discrete_action", [False, True])
def test_probabilistic_ensemble_dynamics(
    observation_shape,
    action_size,
    scalers,
    discrete_action,
):
    scaler, action_scaler, reward_scaler = scalers
    if discrete_action:
        action_scaler = None
    dynamics = ProbabilisticEnsembleDynamics(
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        discrete_action=discrete_action,
    )
    dynamics_tester(dynamics, observation_shape, action_size, discrete_action)
    dynamics_update_tester(
        dynamics, observation_shape, action_size, discrete_action
    )
