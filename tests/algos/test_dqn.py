import pytest

from skbrl.algos.dqn import DQN, DoubleDQN
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_cartpole_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_dqn(observation_shape, action_size, q_func_type):
    dqn = DQN(q_func_type=q_func_type)
    algo_tester(dqn)
    algo_update_tester(dqn, observation_shape, action_size, discrete=True)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_dqn_performance(q_func_type):
    dqn = DQN(n_epochs=1, q_func_type=q_func_type)
    algo_cartpole_tester(dqn)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_double_dqn(observation_shape, action_size, q_func_type):
    double_dqn = DoubleDQN(q_func_type=q_func_type)
    algo_tester(double_dqn)
    algo_update_tester(double_dqn, observation_shape, action_size, True)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_double_dqn_performance(q_func_type):
    double_dqn = DoubleDQN(n_epochs=1, q_func_type=q_func_type)
    algo_cartpole_tester(double_dqn)
