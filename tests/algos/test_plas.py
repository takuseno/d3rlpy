import pytest

from d3rlpy.algos.plas import PLAS, PLASWithPerturbation
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "min_max"])
@pytest.mark.parametrize("action_scaler", [None, "min_max"])
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
def test_plas(
    observation_shape,
    action_size,
    q_func_factory,
    scaler,
    action_scaler,
    target_reduction_type,
):
    plas = PLAS(
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        target_reduction_type=target_reduction_type,
    )
    algo_tester(plas, observation_shape)
    algo_update_tester(plas, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_plas_performance(q_func_factory):
    plas = PLAS(q_func_factory=q_func_factory)
    algo_pendulum_tester(plas, n_trials=1)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "min_max"])
@pytest.mark.parametrize("action_scaler", [None, "min_max"])
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
def test_plas_with_perturbation(
    observation_shape,
    action_size,
    q_func_factory,
    scaler,
    action_scaler,
    target_reduction_type,
):
    plas = PLASWithPerturbation(
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        target_reduction_type=target_reduction_type,
    )
    algo_tester(plas, observation_shape)
    algo_update_tester(plas, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_plas_with_perturbation_performance(q_func_factory):
    plas = PLASWithPerturbation(q_func_factory=q_func_factory)
    algo_pendulum_tester(plas, n_trials=1)
