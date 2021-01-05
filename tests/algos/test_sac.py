import pytest

from d3rlpy.algos.sac import SAC, DiscreteSAC
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester
from .algo_test import algo_pendulum_tester, algo_cartpole_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "standard"])
def test_sac(observation_shape, action_size, q_func_factory, scaler):
    sac = SAC(q_func_factory=q_func_factory, scaler=scaler)
    algo_tester(sac, observation_shape)
    algo_update_tester(sac, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_sac_performance(q_func_factory):
    if q_func_factory == "iqn" or q_func_factory == "fqf":
        pytest.skip("IQN is computationally expensive")

    sac = SAC(q_func_factory=q_func_factory)
    algo_pendulum_tester(sac, n_trials=3)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "standard"])
def test_discrete_sac(observation_shape, action_size, q_func_factory, scaler):
    sac = DiscreteSAC(q_func_factory=q_func_factory, scaler=scaler)
    algo_tester(sac, observation_shape)
    algo_update_tester(sac, observation_shape, action_size, discrete=True)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_discrete_sac_performance(q_func_factory):
    if q_func_factory == "iqn" or q_func_factory == "fqf":
        pytest.skip("IQN is computationally expensive")

    sac = DiscreteSAC(q_func_factory=q_func_factory)
    algo_cartpole_tester(sac, n_trials=3)
