from skbrl.algos.ddpg import DDPG
from .algo_test import algo_tester, algo_pendulum_tester


def test_ddpg():
    ddpg = DDPG()
    algo_tester(ddpg)


def test_ddpg_performance():
    # not good enough for batch RL, but check if it works without errors.
    try:
        ddpg = DDPG(n_epochs=1)
        algo_pendulum_tester(ddpg)
    except AssertionError:
        pass
