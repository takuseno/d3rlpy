import numpy as np
import pytest

from d3rlpy.algos.random_policy import DiscreteRandomPolicy, RandomPolicy


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("action_scaler", [None, "min_max"])
@pytest.mark.parametrize("action_size", [4])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
def test_random_policy(
    distribution, action_scaler, action_size, batch_size, observation_shape
):
    algo = RandomPolicy(distribution=distribution, action_scler=action_scaler)
    algo.create_impl(observation_shape, action_size)

    x = np.random.random((batch_size,) + observation_shape)

    # check predict
    action = algo.predict(x)
    assert action.shape == (batch_size, action_size)

    # check sample_action
    action = algo.sample_action(x)
    assert action.shape == (batch_size, action_size)


@pytest.mark.parametrize("action_size", [4])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
def test_discrete_random_policy(action_size, batch_size, observation_shape):
    algo = DiscreteRandomPolicy()
    algo.create_impl(observation_shape, action_size)

    x = np.random.random((batch_size,) + observation_shape)

    # check predict
    action = algo.predict(x)
    assert action.shape == (batch_size,)
    assert np.all(action < action_size)

    # check sample_action
    action = algo.sample_action(x)
    assert action.shape == (batch_size,)
    assert np.all(action < action_size)
