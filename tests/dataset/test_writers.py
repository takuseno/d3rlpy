from typing import Sequence

import numpy as np
import pytest

from d3rlpy.dataset import (
    BasicWriterPreprocess,
    EpisodeBase,
    ExperienceWriter,
    InfiniteBuffer,
    LastFrameWriterPreprocess,
)
from d3rlpy.types import Shape

from ..testing_utils import create_episode, create_observation


@pytest.mark.parametrize("observation_shape", [(4,)])
@pytest.mark.parametrize("action_size", [2])
def test_basic_writer_preprocess(
    observation_shape: Sequence[int], action_size: int
) -> None:
    observation = create_observation(observation_shape)
    action = np.random.random(action_size)
    reward = np.random.random(1)

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
def test_last_frame_writer_process(observation_shape: Shape) -> None:
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
def test_episode_writer(
    observation_shape: Shape, action_size: int, length: int, terminated: bool
) -> None:
    episode: EpisodeBase = create_episode(
        observation_shape, action_size, length
    )
    buffer = InfiniteBuffer()
    writer = ExperienceWriter(
        buffer,
        BasicWriterPreprocess(),
        observation_signature=episode.observation_signature,
        action_signature=episode.action_signature,
        reward_signature=episode.reward_signature,
    )

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
    if isinstance(observation_shape[0], tuple):
        assert tuple(episode.observation_signature.shape) == observation_shape
    else:
        assert (
            tuple(episode.observation_signature.shape[0]) == observation_shape
        )
