import pytest

from d3rlpy.algos.dqn import DoubleDQNConfig, DQNConfig
from d3rlpy.models import MeanQFunctionFactory, QRQFunctionFactory

from ..testing_utils import create_scaler_tuple
from .algo_test import algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_dqn(
    observation_shape,
    action_size,
    n_critics,
    q_func_factory,
    scalers,
):
    observation_scaler, _, reward_scaler = create_scaler_tuple(scalers)
    config = DQNConfig(
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    dqn = config.create()
    algo_tester(dqn, observation_shape, test_q_function_copy=True)
    algo_update_tester(
        dqn,
        observation_shape,
        action_size,
        discrete=True,
        test_q_function_optim_copy=True,
    )


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_double_dqn(
    observation_shape,
    action_size,
    n_critics,
    q_func_factory,
    scalers,
):
    observation_scaler, _, reward_scaler = create_scaler_tuple(scalers)
    config = DoubleDQNConfig(
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    double_dqn = config.create()
    algo_tester(double_dqn, observation_shape, test_q_function_copy=True)
    algo_update_tester(
        double_dqn,
        observation_shape,
        action_size,
        discrete=True,
        test_q_function_optim_copy=True,
    )
