import pytest

from skbrl.algos.bcq import BCQ, DiscreteBCQ
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_pendulum_tester, algo_cartpole_tester


def test_bcq():
    bcq = BCQ()
    algo_tester(bcq)


@pytest.mark.skip(reason='BCQ is computationally expensive.')
def test_bcq_performance():
    bcq = BCQ(n_epochs=5, use_batch_norm=False)
    algo_pendulum_tester(bcq, n_trials=5)


def test_discrete_bcq():
    bcq = DiscreteBCQ()
    algo_tester(bcq)


@performance_test
def test_discrete_bcq_performance():
    bcq = DiscreteBCQ(n_epochs=1)
    algo_cartpole_tester(bcq)
