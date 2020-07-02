import pytest

from skbrl.algos.sac import SAC
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_pendulum_tester


def test_sac():
    sac = SAC()
    algo_tester(sac)


@performance_test
@pytest.mark.parametrize('use_quantile_regression', [True, False])
def test_sac_performance(use_quantile_regression):
    sac = SAC(n_epochs=5, use_quantile_regression=use_quantile_regression)
    algo_pendulum_tester(sac, n_trials=5)
