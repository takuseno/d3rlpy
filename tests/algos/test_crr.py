import pytest

from d3rlpy.algos.crr import CRR
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "min_max"])
@pytest.mark.parametrize("action_scaler", [None, "min_max"])
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
@pytest.mark.parametrize("advantage_type", ["mean", "max"])
@pytest.mark.parametrize("weight_type", ["exp", "binary"])
def test_crr(
    observation_shape,
    action_size,
    q_func_factory,
    scaler,
    action_scaler,
    target_reduction_type,
    advantage_type,
    weight_type,
):
    crr = CRR(
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        target_reduction_type=target_reduction_type,
        advantage_type=advantage_type,
        weight_type=weight_type,
    )
    algo_tester(crr, observation_shape)
    algo_update_tester(crr, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_crr_performance(q_func_factory):
    crr = CRR(q_func_factory=q_func_factory)
    algo_pendulum_tester(crr, n_trials=1)
