import pytest

from d3rlpy.algos.cql import CQL, DiscreteCQL
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester
from .algo_test import algo_cartpole_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
def test_cql(observation_shape, action_size, q_func_type, scaler):
    cql = CQL(q_func_type=q_func_type, scaler=scaler)
    algo_tester(cql, observation_shape)
    algo_update_tester(cql, observation_shape, action_size)


@pytest.mark.skip(reason='CQL is computationally expensive.')
def test_cql_performance():
    cql = CQL(n_epochs=5)
    algo_pendulum_tester(cql, n_trials=3)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
def test_discrete_cql(observation_shape, action_size, q_func_type, scaler):
    cql = DiscreteCQL(q_func_type=q_func_type, scaler=scaler)
    algo_tester(cql, observation_shape)
    algo_update_tester(cql, observation_shape, action_size, True)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_discrete_cql_performance(q_func_type):
    cql = DiscreteCQL(n_epochs=1, q_func_type=q_func_type)
    algo_cartpole_tester(cql)
