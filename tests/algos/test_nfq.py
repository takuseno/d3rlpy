import pytest

from d3rlpy.algos.nfq import NFQ
from tests import performance_test

from .algo_test import algo_cartpole_tester, algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scalers", [(None, None), ("min_max", "min_max")])
def test_nfq(
    observation_shape,
    action_size,
    n_critics,
    q_func_factory,
    scalers,
):
    observation_scaler, reward_scaler = scalers
    nfq = NFQ(
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    algo_tester(nfq, observation_shape, test_q_function_copy=True)
    algo_update_tester(
        nfq,
        observation_shape,
        action_size,
        discrete=True,
        test_q_function_optim_copy=True,
    )


@performance_test
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
def test_nfq_performance(q_func_factory):
    nfq = NFQ(q_func_factory=q_func_factory)
    algo_cartpole_tester(nfq)
