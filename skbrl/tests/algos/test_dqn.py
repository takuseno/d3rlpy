from skbrl.algos.dqn import DQN, DoubleDQN
from .algo_test import algo_tester, algo_cartpole_tester


def test_dqn():
    dqn = DQN()
    algo_tester(dqn)


def test_dqn_performance():
    dqn = DQN(n_epochs=1, learning_rate=1e-4, gamma=0.99)
    algo_cartpole_tester(dqn)


def test_double_dqn():
    double_dqn = DoubleDQN()
    algo_tester(double_dqn)


def test_double_dqn_performance():
    double_dqn = DoubleDQN(n_epochs=1, learning_rate=1e-4, gamma=0.99)
    algo_cartpole_tester(double_dqn)
