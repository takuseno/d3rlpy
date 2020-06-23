from skbrl.algos.sac import SAC
from .algo_test import algo_tester, algo_pendulum_tester


def test_sac():
    sac = SAC()
    algo_tester(sac)


def test_sac_performance():
    sac = SAC(n_epochs=5, use_batch_norm=False)
    algo_pendulum_tester(sac)
