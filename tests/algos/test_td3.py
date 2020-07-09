import pytest

from d3rlpy.algos.td3 import TD3
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_td3(observation_shape, action_size, q_func_type):
    td3 = TD3(q_func_type=q_func_type)
    algo_tester(td3)
    algo_update_tester(td3, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_td3_performance(q_func_type):
    if q_func_type == 'iqn':
        pytest.skip('IQN is computationally expensive')

    td3 = TD3(n_epochs=5, q_func_type=q_func_type)
    algo_pendulum_tester(td3, n_trials=3)
