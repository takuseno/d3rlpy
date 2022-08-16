import numpy as np
import pytest

from d3rlpy.dataset import (
    BasicWriterPreprocess,
    ExperienceWriter,
    InfiniteBuffer,
    LastFrameWriterPreprocess,
)

from ..testing_utils import create_observation


@pytest.mark.parametrize("observation_shape", [(4,)])
@pytest.mark.parametrize("action_size", [2])
def test_basic_writer_preprocess(observation_shape, action_size):
    observation = create_observation(observation_shape)
    action = np.random.random(action_size)
    reward = np.random.random()

    preprocessor = BasicWriterPreprocess()

    processed_observation = preprocessor.process_observation(observation)
    processed_action = preprocessor.process_action(action)
    processed_reward = preprocessor.process_reward(reward)

    assert np.all(processed_observation == observation)
    assert np.all(processed_action == action)
    assert processed_reward == reward


@pytest.mark.parametrize(
    "observation_shape",
    [
        (
            2,
            4,
        ),
        (4, 84, 84),
        ((2, 4), (4, 84, 84)),
    ],
)
def test_last_frame_writer_process(observation_shape):
    observation = create_observation(observation_shape)

    preprocessor = LastFrameWriterPreprocess()

    processed_observation = preprocessor.process_observation(observation)

    if isinstance(observation, (list, tuple)):
        for obs, processed_obs in zip(observation, processed_observation):
            assert np.all(obs[-1] == processed_obs[0])
    else:
        assert np.all(observation[-1] == processed_observation[0])


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("terminated", [True, False])
def test_episode_writer(observation_shape, action_size, length, terminated):
    buffer = InfiniteBuffer()
    writer = ExperienceWriter(buffer, BasicWriterPreprocess())

    for _ in range(length):
        writer.write(
            observation=create_observation(observation_shape),
            action=np.random.random(action_size),
            reward=np.random.random(),
        )
    writer.clip_episode(terminated)

    if terminated:
        assert buffer.transition_count == length
    else:
        assert buffer.transition_count == length - 1
    episode = buffer.episodes[0]
    assert tuple(episode.observation_shape) == observation_shape
