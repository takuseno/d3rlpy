from skbrl.algos.dqn import DQN, DoubleDQN
from .algo_test import algo_tester


def test_dqn():
    dqn = DQN()
    algo_tester(dqn)


def test_double_dqn():
    double_dqn = DoubleDQN()
    algo_tester(double_dqn)
