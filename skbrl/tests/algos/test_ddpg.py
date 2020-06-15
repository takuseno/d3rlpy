from skbrl.algos.ddpg import DDPG
from .algo_test import algo_tester, algo_cartpole_tester


def test_ddpg():
    ddpg = DDPG()
    algo_tester(ddpg)
