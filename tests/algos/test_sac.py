import pytest

from d3rlpy.algos.sac import DiscreteSACConfig, SACConfig
from d3rlpy.models import MeanQFunctionFactory, QRQFunctionFactory

from ..testing_utils import create_scaler_tuple
from .algo_test import algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_sac(
    observation_shape,
    action_size,
    q_func_factory,
    scalers,
):
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers
    )
    config = SACConfig(
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    sac = config.create()
    algo_tester(
        sac, observation_shape, test_policy_copy=True, test_q_function_copy=True
    )
    algo_update_tester(
        sac,
        observation_shape,
        action_size,
        test_policy_optim_copy=True,
        test_q_function_optim_copy=True,
    )


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_discrete_sac(observation_shape, action_size, q_func_factory, scalers):
    observation_scaler, _, reward_scaler = create_scaler_tuple(scalers)
    config = DiscreteSACConfig(
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    sac = config.create()
    algo_tester(
        sac, observation_shape, test_policy_copy=True, test_q_function_copy=True
    )
    algo_update_tester(
        sac,
        observation_shape,
        action_size,
        discrete=True,
        test_q_function_optim_copy=True,
        test_policy_optim_copy=True,
    )
