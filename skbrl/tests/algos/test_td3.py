import pytest

from skbrl.algos.td3 import TD3
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('use_quantile_regression', [None, 'qr', 'iqn'])
def test_td3(observation_shape, action_size, use_quantile_regression):
    td3 = TD3()
    algo_tester(td3)
    algo_update_tester(td3, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize('use_quantile_regression', [None, 'qr', 'iqn'])
def test_td3_performance(use_quantile_regression):
    if use_quantile_regression == 'iqn':
        pytest.skip('IQN is computationally expensive')

    td3 = TD3(n_epochs=5, use_quantile_regression=use_quantile_regression)
    algo_pendulum_tester(td3, n_trials=3)
