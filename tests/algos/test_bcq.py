import pytest

from d3rlpy.algos.bcq import BCQ, DiscreteBCQ
from tests import performance_test

from .algo_test import (
    algo_cartpole_tester,
    algo_pendulum_tester,
    algo_tester,
    algo_update_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "min_max"])
@pytest.mark.parametrize("action_scaler", [None, "min_max"])
def test_bcq(
    observation_shape, action_size, q_func_factory, scaler, action_scaler
):
    bcq = BCQ(
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        rl_start_epoch=0,
    )
    algo_tester(bcq, observation_shape)
    algo_update_tester(bcq, observation_shape, action_size)


@pytest.mark.skip(reason="BCQ is computationally expensive.")
def test_bcq_performance():
    bcq = BCQ(use_batch_norm=False)
    algo_pendulum_tester(bcq, n_trials=5)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_critics", [1, 2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "min_max"])
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
def test_discrete_bcq(
    observation_shape,
    action_size,
    n_critics,
    q_func_factory,
    scaler,
    target_reduction_type,
):
    bcq = DiscreteBCQ(
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        scaler=scaler,
        target_reduction_type=target_reduction_type,
    )
    algo_tester(bcq, observation_shape)
    algo_update_tester(bcq, observation_shape, action_size, discrete=True)


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_discrete_bcq_performance(q_func_factory):
    bcq = DiscreteBCQ(q_func_factory=q_func_factory)
    algo_cartpole_tester(bcq)
