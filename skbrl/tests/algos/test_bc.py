from skbrl.algos.bc import BC, DiscreteBC
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_pendulum_tester, algo_cartpole_tester


def test_bc():
    bc = BC()
    algo_tester(bc)


@performance_test
def test_bc_performance():
    bc = BC(n_epochs=1)
    algo_pendulum_tester(bc)


def test_discrete_bc():
    bc = DiscreteBC()
    algo_tester(bc)


@performance_test
def test_discrete_bc_performance():
    bc = DiscreteBC(n_epochs=1)
    algo_cartpole_tester(bc)
