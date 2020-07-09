import pytest

from skbrl.algos.bcq import BCQ, DiscreteBCQ
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester
from .algo_test import algo_pendulum_tester, algo_cartpole_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_bcq(observation_shape, action_size, q_func_type):
    bcq = BCQ(q_func_type=q_func_type)
    algo_tester(bcq)
    algo_update_tester(bcq, observation_shape, action_size)


@pytest.mark.skip(reason='BCQ is computationally expensive.')
def test_bcq_performance():
    bcq = BCQ(n_epochs=5, use_batch_norm=False)
    algo_pendulum_tester(bcq, n_trials=5)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_discrete_bcq(observation_shape, action_size, q_func_type):
    bcq = DiscreteBCQ(q_func_type=q_func_type)
    algo_tester(bcq)
    algo_update_tester(bcq, observation_shape, action_size, discrete=True)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_discrete_bcq_performance(q_func_type):
    bcq = DiscreteBCQ(n_epochs=1, q_func_type=q_func_type)
    algo_cartpole_tester(bcq)
