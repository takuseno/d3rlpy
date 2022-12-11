import pytest

from d3rlpy.algos.bc import BCConfig, DiscreteBCConfig

from ..testing_utils import create_scaler_tuple
from .algo_test import algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("policy_type", ["deterministic", "stochastic"])
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_bc(observation_shape, action_size, policy_type, scalers):
    observation_scaler, action_scaler, _ = create_scaler_tuple(scalers)
    config = BCConfig(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        policy_type=policy_type,
    )
    bc = config.create()
    algo_tester(bc, observation_shape, imitator=True)
    algo_update_tester(bc, observation_shape, action_size)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("scaler", [None, "min_max"])
def test_discrete_bc(observation_shape, action_size, scaler):
    observation_scaler, _, _ = create_scaler_tuple(scaler)
    config = DiscreteBCConfig(observation_scaler=observation_scaler)
    bc = config.create()
    algo_tester(bc, observation_shape, imitator=True)
    algo_update_tester(bc, observation_shape, action_size, discrete=True)
