import pytest

from skbrl.algos.dqn import DQN, DoubleDQN
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_cartpole_tester


def test_dqn():
    dqn = DQN()
    algo_tester(dqn)


@performance_test
@pytest.mark.parametrize('use_quantile_regression', [True, False])
def test_dqn_performance(use_quantile_regression):
    dqn = DQN(n_epochs=1, use_quantile_regression=use_quantile_regression)
    algo_cartpole_tester(dqn)


def test_double_dqn():
    double_dqn = DoubleDQN()
    algo_tester(double_dqn)


@performance_test
@pytest.mark.parametrize('use_quantile_regression', [True, False])
def test_double_dqn_performance(use_quantile_regression):
    double_dqn = DoubleDQN(n_epochs=1,
                           use_quantile_regression=use_quantile_regression)
    algo_cartpole_tester(double_dqn)
