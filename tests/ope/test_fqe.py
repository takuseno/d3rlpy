import pytest

from d3rlpy.algos import DDPGConfig, DQNConfig
from d3rlpy.models import MeanQFunctionFactory, QRQFunctionFactory
from d3rlpy.ope.fqe import FQE, DiscreteFQE, FQEConfig
from tests.algos.algo_test import algo_tester

from ..testing_utils import create_scaler_tuple


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_fqe(
    observation_shape,
    q_func_factory,
    scalers,
):
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers
    )
    algo = DDPGConfig().create()
    algo.create_impl(observation_shape, 2)
    config = FQEConfig(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        q_func_factory=q_func_factory,
    )
    fqe = FQE(algo=algo, config=config)
    algo_tester(
        fqe,
        observation_shape,
        test_policy_copy=False,
        test_policy_optim_copy=False,
        test_from_json=False,
        test_q_function_copy=False,
        test_q_function_optim_copy=False,
    )


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_discrete_fqe(observation_shape, q_func_factory, scalers):
    observation_scaler, _, reward_scaler = create_scaler_tuple(scalers)
    algo = DQNConfig().create()
    algo.create_impl(observation_shape, 2)
    config = FQEConfig(
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
        q_func_factory=q_func_factory,
    )
    fqe = DiscreteFQE(algo=algo, config=config)
    algo_tester(
        fqe,
        observation_shape,
        test_policy_copy=False,
        test_policy_optim_copy=False,
        test_from_json=False,
        test_q_function_copy=False,
        test_q_function_optim_copy=False,
    )
