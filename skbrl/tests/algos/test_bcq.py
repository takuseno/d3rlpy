import pytest

from skbrl.algos.bcq import BCQ, DiscreteBCQ
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_update_tester
from .algo_test import algo_pendulum_tester, algo_cartpole_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('use_quantile_regression', [None, 'qr', 'iqn'])
def test_bcq(observation_shape, action_size, use_quantile_regression):
    bcq = BCQ(use_quantile_regression=use_quantile_regression)
    algo_tester(bcq)
    algo_update_tester(bcq, observation_shape, action_size)


@pytest.mark.skip(reason='BCQ is computationally expensive.')
def test_bcq_performance():
    bcq = BCQ(n_epochs=5, use_batch_norm=False)
    algo_pendulum_tester(bcq, n_trials=5)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('use_quantile_regression', [None, 'qr', 'iqn'])
def test_discrete_bcq(observation_shape, action_size, use_quantile_regression):
    bcq = DiscreteBCQ(use_quantile_regression=use_quantile_regression)
    algo_tester(bcq)
    algo_update_tester(bcq, observation_shape, action_size, discrete=True)


@performance_test
@pytest.mark.parametrize('use_quantile_regression', [None, 'qr', 'iqn'])
def test_discrete_bcq_performance(use_quantile_regression):
    bcq = DiscreteBCQ(n_epochs=1,
                      use_quantile_regression=use_quantile_regression)
    algo_cartpole_tester(bcq)
