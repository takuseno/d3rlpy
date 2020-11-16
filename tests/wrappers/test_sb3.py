import numpy as np
import pytest

from d3rlpy.algos import SAC
from d3rlpy.wrappers.sb3 import SB3Wrapper


@pytest.mark.parametrize('observation_shape', [(10, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [22])
def test_sb3_wrapper(observation_shape, action_size, batch_size):
    algo = SAC()
    algo.create_impl(observation_shape, action_size)

    sb3 = SB3Wrapper(algo)

    observations = np.random.random((batch_size, ) + observation_shape)

    # check greedy action
    actions, state = sb3.predict(observations, deterministic=True)
    assert actions.shape == (batch_size, action_size)
    assert state is None

    # check sampling
    stochastic_actions, state = sb3.predict(observations, deterministic=False)
    assert stochastic_actions.shape == (batch_size, action_size)
    assert state is None
    assert not np.allclose(actions, stochastic_actions)
