import pytest

from d3rlpy.algos.awr import AWR
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('scaler', [None, 'standard'])
def test_awr(observation_shape, action_size, scaler):
    awr = AWR(scaler=scaler, batch_size=100, batch_size_per_update=20)
    algo_tester(awr, observation_shape)
    algo_update_tester(awr, observation_shape, action_size)


@performance_test
def test_awr_performance():
    awr = AWR(n_epochs=5)
    algo_pendulum_tester(awr, n_trials=3)
