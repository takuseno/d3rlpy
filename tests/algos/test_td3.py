import pytest

from d3rlpy.algos.td3 import TD3
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_factory', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
def test_td3(observation_shape, action_size, q_func_factory, scaler):
    td3 = TD3(q_func_factory=q_func_factory, scaler=scaler)
    algo_tester(td3, observation_shape)
    algo_update_tester(td3, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize('q_func_factory', ['mean', 'qr', 'iqn', 'fqf'])
def test_td3_performance(q_func_factory):
    if q_func_factory == 'iqn' or q_func_factory == 'fqf':
        pytest.skip('IQN is computationally expensive')

    td3 = TD3(q_func_factory=q_func_factory)
    algo_pendulum_tester(td3, n_trials=3)
