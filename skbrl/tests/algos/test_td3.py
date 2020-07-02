import pytest

from skbrl.algos.td3 import TD3
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_pendulum_tester


def test_td3():
    td3 = TD3()
    algo_tester(td3)


@performance_test
@pytest.mark.parametrize('use_quantile_regression', [True, False])
def test_td3_performance(use_quantile_regression):
    td3 = TD3(n_epochs=5, use_quantile_regression=use_quantile_regression)
    # TD3 works well, but it's still unstable.
    algo_pendulum_tester(td3, n_trials=5)
