from unittest.mock import Mock

import numpy as np
import pytest

from d3rlpy.algos import DDPGConfig, DQNConfig
from d3rlpy.models import MeanQFunctionFactory, QRQFunctionFactory
from d3rlpy.ope.fqe import FQE, DiscreteFQE, FQEConfig
from tests.algos.algo_test import DummyImpl, algo_update_tester
from tests.base_test import base_tester

from ..testing_utils import create_scaler_tuple


def ope_tester(ope, observation_shape, action_size=2):
    # dummy impl object
    impl = DummyImpl(observation_shape, action_size)

    base_tester(ope, impl, observation_shape, action_size, skip_from_json=True)

    ope._algo._impl = impl
    ope._impl = impl

    # check save policy
    impl.save_policy = Mock()
    ope.save_policy("policy.pt")
    impl.save_policy.assert_called_with("policy.pt")

    # check predict
    x = np.random.random((2, 3)).tolist()
    ref_y = np.random.random((2, action_size)).tolist()
    impl.predict_best_action = Mock(return_value=ref_y)
    y = ope.predict(x)
    assert y == ref_y
    impl.predict_best_action.assert_called_with(x)

    # check predict_value
    action = np.random.random((2, action_size)).tolist()
    ref_value = np.random.random((2, 3)).tolist()
    impl.predict_value = Mock(return_value=ref_value)
    value = ope.predict_value(x, action)
    assert value == ref_value
    impl.predict_value.assert_called_with(x, action, False)

    # check sample_action
    impl.sample_action = Mock(return_value=ref_y)
    try:
        y = ope.sample_action(x)
        assert y == ref_y
        impl.sample_action.assert_called_with(x)
    except NotImplementedError:
        pass

    ope._impl = None
    ope._algo._impl = None


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_fqe(
    observation_shape,
    action_size,
    q_func_factory,
    scalers,
):
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers
    )
    algo = DDPGConfig().create()
    config = FQEConfig(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        q_func_factory=q_func_factory,
    )
    fqe = FQE(algo=algo, config=config)
    ope_tester(fqe, observation_shape)
    algo.create_impl(observation_shape, action_size)
    algo_update_tester(fqe, observation_shape, action_size, discrete=False)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_discrete_fqe(observation_shape, action_size, q_func_factory, scalers):
    observation_scaler, _, reward_scaler = create_scaler_tuple(scalers)
    algo = DQNConfig().create()
    config = FQEConfig(
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
        q_func_factory=q_func_factory,
    )
    fqe = DiscreteFQE(algo=algo, config=config)
    ope_tester(fqe, observation_shape)
    algo.create_impl(observation_shape, action_size)
    algo_update_tester(fqe, observation_shape, action_size, discrete=True)
