import pytest

from d3rlpy.algos.bcq import BCQConfig, DiscreteBCQConfig
from d3rlpy.models import MeanQFunctionFactory, QRQFunctionFactory

from ..testing_utils import create_scaler_tuple
from .algo_test import algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_bcq(
    observation_shape,
    action_size,
    q_func_factory,
    scalers,
):
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers
    )
    config = BCQConfig(
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        rl_start_step=0,
    )
    bcq = config.create()
    algo_tester(bcq, observation_shape, test_q_function_copy=True)
    algo_update_tester(
        bcq,
        observation_shape,
        action_size,
        test_q_function_optim_copy=True,
        test_policy_optim_copy=True,
    )


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_discrete_bcq(
    observation_shape,
    action_size,
    n_critics,
    q_func_factory,
    scalers,
):
    observation_scaler, _, reward_scaler = create_scaler_tuple(scalers)
    config = DiscreteBCQConfig(
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    bcq = config.create()
    algo_tester(bcq, observation_shape, test_q_function_copy=True)
    algo_update_tester(
        bcq,
        observation_shape,
        action_size,
        discrete=True,
        test_q_function_optim_copy=True,
    )
