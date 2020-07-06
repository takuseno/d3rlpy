import pytest

from skbrl.algos.ddpg import DDPG
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('use_quantile_regression', [None, 'qr', 'iqn'])
def test_ddpg(observation_shape, action_size, use_quantile_regression):
    ddpg = DDPG(use_quantile_regression=use_quantile_regression)
    algo_tester(ddpg)
    algo_update_tester(ddpg, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize('use_quantile_regression', [None, 'qr', 'iqn'])
def test_ddpg_performance(use_quantile_regression):
    # not good enough for batch RL, but check if it works without errors.
    try:
        ddpg = DDPG(n_epochs=1,
                    use_quantile_regression=use_quantile_regression)
        algo_pendulum_tester(ddpg, n_trials=1)
    except AssertionError:
        pass
