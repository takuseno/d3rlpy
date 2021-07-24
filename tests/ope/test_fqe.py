from unittest.mock import Mock

import numpy as np
import pytest

from d3rlpy.algos import DDPG, DQN
from d3rlpy.ope.fqe import FQE, DiscreteFQE
from tests.algos.algo_test import DummyImpl, algo_update_tester
from tests.base_test import base_tester


def ope_tester(ope, observation_shape, action_size=2):
    # dummy impl object
    impl = DummyImpl(observation_shape, action_size)

    base_tester(ope, impl, observation_shape, action_size)

    ope._algo.impl = impl
    ope.impl = impl

    # check save policy
    impl.save_policy = Mock()
    ope.save_policy("policy.pt", False)
    impl.save_policy.assert_called_with("policy.pt", False)

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

    ope.impl = None
    ope._algo.impl = None


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize(
    "scalers", [(None, None, None), ("min_max", "min_max", "min_max")]
)
def test_fqe(
    observation_shape,
    action_size,
    q_func_factory,
    scalers,
):
    scaler, action_scaler, reward_scaler = scalers
    algo = DDPG()
    fqe = FQE(
        algo=algo,
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        q_func_factory=q_func_factory,
    )
    ope_tester(fqe, observation_shape)
    algo.create_impl(observation_shape, action_size)
    algo_update_tester(fqe, observation_shape, action_size, discrete=False)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scalers", [(None, None), ("min_max", "min_max")])
def test_discrete_fqe(observation_shape, action_size, q_func_factory, scalers):
    scaler, reward_scaler = scalers
    algo = DQN()
    fqe = DiscreteFQE(
        algo=algo,
        scaler=scaler,
        reward_scaler=reward_scaler,
        q_func_factory=q_func_factory,
    )
    ope_tester(fqe, observation_shape)
    algo.create_impl(observation_shape, action_size)
    algo_update_tester(fqe, observation_shape, action_size, discrete=True)
