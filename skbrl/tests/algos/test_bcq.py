import pytest

from skbrl.algos.bcq import BCQ
from .algo_test import algo_tester, algo_pendulum_tester


def test_bcq():
    bcq = BCQ()
    algo_tester(bcq)


@pytest.mark.skip(reason='BCQ is computationally expensive.')
def test_bcq_performance():
    bcq = BCQ(n_epochs=5, use_batch_norm=False)
    algo_pendulum_tester(bcq, n_trials=5)
