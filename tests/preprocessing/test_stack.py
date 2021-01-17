import pytest
import numpy as np

from d3rlpy.preprocessing.stack import StackedObservation
from d3rlpy.preprocessing.stack import BatchStackedObservation


@pytest.mark.parametrize("observation_shape", [(4, 84, 84)])
@pytest.mark.parametrize("n_frames", [4])
@pytest.mark.parametrize("data_size", [100])
def test_stacked_observation(observation_shape, n_frames, data_size):
    shape = (data_size, *observation_shape)
    images = np.random.randint(255, size=shape, dtype=np.uint8)
    padding = np.zeros((n_frames - 1, *observation_shape), dtype=np.uint8)
    padded_images = np.vstack([padding, images])

    stacked_observation = StackedObservation(observation_shape, n_frames)

    for i in range(data_size):
        image = images[i]
        stacked_observation.append(image)
        ref_observation = np.vstack(padded_images[i : i + n_frames])
        assert stacked_observation.eval().shape == ref_observation.shape
        assert np.all(stacked_observation.eval() == ref_observation)

    stacked_observation.clear()
    assert np.all(stacked_observation.eval() == 0)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84)])
@pytest.mark.parametrize("n_frames", [4])
@pytest.mark.parametrize("n_envs", [5])
@pytest.mark.parametrize("data_size", [100])
def test_batch_stacked_observation(
    observation_shape, n_frames, n_envs, data_size
):
    shape = (data_size, n_envs, *observation_shape)
    images = np.random.randint(255, size=shape, dtype=np.uint8)
    padding = np.zeros(
        (n_frames - 1, n_envs, *observation_shape), dtype=np.uint8
    )
    padded_images = np.vstack([padding, images])

    stacked_observation = BatchStackedObservation(
        observation_shape, n_frames, n_envs
    )

    for i in range(data_size):
        image = images[i]
        stacked_observation.append(image)
        obs = np.transpose(padded_images[i : i + n_frames], [0, 2, 1, 3, 4])
        ref_observation = np.transpose(np.vstack(obs), [1, 0, 2, 3])
        assert stacked_observation.eval().shape == ref_observation.shape
        assert np.all(stacked_observation.eval() == ref_observation)

    stacked_observation.clear()
    assert np.all(stacked_observation.eval() == 0)
