from skbrl.algos.sac import SAC
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_pendulum_tester


def test_sac():
    sac = SAC()
    algo_tester(sac)


@performance_test
def test_sac_performance():
    sac = SAC(n_epochs=5, use_batch_norm=False)
    algo_pendulum_tester(sac, n_trials=5)
