from skbrl.algos.td3 import TD3
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_pendulum_tester


def test_td3():
    td3 = TD3()
    algo_tester(td3)


@performance_test
def test_td3_performance():
    td3 = TD3(n_epochs=5, use_batch_norm=False)
    # TD3 works well, but it's still unstable.
    algo_pendulum_tester(td3, n_trials=5)
