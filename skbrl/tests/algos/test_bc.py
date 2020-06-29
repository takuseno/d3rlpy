from skbrl.algos.bc import BC
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_pendulum_tester


def test_bc():
    bc = BC()
    algo_tester(bc)


@performance_test
def test_bc_performance():
    bc = BC(n_epochs=1)
    algo_pendulum_tester(bc)
