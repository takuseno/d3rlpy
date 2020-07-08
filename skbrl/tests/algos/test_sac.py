import pytest

from skbrl.algos.sac import SAC
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('distribution_type', [None, 'qr', 'iqn'])
def test_sac(observation_shape, action_size, distribution_type):
    sac = SAC(distribution_type=distribution_type)
    algo_tester(sac)
    algo_update_tester(sac, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize('distribution_type', [None, 'qr', 'iqn'])
def test_sac_performance(distribution_type):
    if distribution_type == 'iqn':
        pytest.skip('IQN is computationally expensive')

    sac = SAC(n_epochs=5, distribution_type=distribution_type)
    algo_pendulum_tester(sac, n_trials=3)
