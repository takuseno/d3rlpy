from skbrl.algos.dqn import DQN, DoubleDQN
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_cartpole_tester


def test_dqn():
    dqn = DQN()
    algo_tester(dqn)


@performance_test
def test_dqn_performance():
    dqn = DQN(n_epochs=1)
    algo_cartpole_tester(dqn)


def test_double_dqn():
    double_dqn = DoubleDQN()
    algo_tester(double_dqn)


@performance_test
def test_double_dqn_performance():
    double_dqn = DoubleDQN(n_epochs=1)
    algo_cartpole_tester(double_dqn)
