import pytest

from skbrl.algos.bc import BC, DiscreteBC
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_update_tester
from .algo_test import algo_pendulum_tester, algo_cartpole_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
def test_bc(observation_shape, action_size):
    bc = BC()
    algo_tester(bc)
    algo_update_tester(bc, observation_shape, action_size)


@performance_test
def test_bc_performance():
    bc = BC(n_epochs=1)
    algo_pendulum_tester(bc)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
def test_discrete_bc(observation_shape, action_size):
    bc = DiscreteBC()
    algo_tester(bc)
    algo_update_tester(bc, observation_shape, action_size, discrete=True)


@performance_test
def test_discrete_bc_performance():
    bc = DiscreteBC(n_epochs=1)
    algo_cartpole_tester(bc)
