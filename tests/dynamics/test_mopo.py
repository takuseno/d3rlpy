import pytest

from d3rlpy.dynamics.mopo import MOPO
from .dynamics_test import dynamics_tester, dynamics_update_tester


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('scaler', [None, 'pixel', 'min_max', 'standard'])
@pytest.mark.parametrize('discrete_action', [False, True])
def test_mopo(observation_shape, action_size, scaler, discrete_action):
    mopo = MOPO(scaler=scaler, discrete_action=discrete_action)
    dynamics_tester(mopo, observation_shape, action_size)
    dynamics_update_tester(mopo, observation_shape, action_size,
                           discrete_action)
