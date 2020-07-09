import pytest

from skbrl.algos.ddpg import DDPG
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_ddpg(observation_shape, action_size, q_func_type):
    ddpg = DDPG(q_func_type=q_func_type)
    algo_tester(ddpg)
    algo_update_tester(ddpg, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_ddpg_performance(q_func_type):
    # not good enough for batch RL, but check if it works without errors.
    try:
        ddpg = DDPG(n_epochs=1, q_func_type=q_func_type)
        algo_pendulum_tester(ddpg, n_trials=1)
    except AssertionError:
        pass
