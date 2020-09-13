import pytest
import numpy as np

from unittest.mock import Mock
from d3rlpy.metrics.ope.fqe import FQE
from d3rlpy.algos import DDPG, DQN
from tests.base_test import base_tester
from tests.algos.algo_test import algo_update_tester
from tests.algos.algo_test import DummyImpl


def ope_tester(ope, observation_shape, action_size=2):
    # dummy impl object
    impl = DummyImpl(observation_shape, action_size)

    base_tester(ope, impl, observation_shape, action_size)

    ope.algo.impl = impl
    ope.impl = impl

    # check save policy
    impl.save_policy = Mock()
    ope.save_policy('policy.pt', False)
    impl.save_policy.assert_called_with('policy.pt', False)

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
    ope.algo.impl = None


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('discrete_action', [False, True])
def test_fqe(observation_shape, action_size, q_func_type, scaler,
             discrete_action):
    if discrete_action:
        algo = DQN()
    else:
        algo = DDPG()
    fqe = FQE(algo=algo,
              discrete_action=discrete_action,
              scaler=scaler,
              q_func_type=q_func_type)
    ope_tester(fqe, observation_shape)
    algo.create_impl(observation_shape, action_size)
    algo_update_tester(fqe,
                       observation_shape,
                       action_size,
                       discrete=discrete_action)
