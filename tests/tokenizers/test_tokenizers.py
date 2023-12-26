import numpy as np

from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer
from d3rlpy.tokenizers import (
    FloatTokenizer,
    tokenize_action,
    tokenize_observation,
    tokenize_replay_buffer,
)
from d3rlpy.types import NDArray

from ..testing_utils import create_episode


def test_float_tokenizer() -> None:
    tokenizer = FloatTokenizer(num_bins=100, use_mu_law_encode=False)
    v: NDArray = (2.0 * np.arange(100) / 100 - 1).astype(np.float32)
    tokenized_v = tokenizer(v)
    assert np.all(tokenized_v == np.arange(100))

    # check mu_law_encode
    tokenizer = FloatTokenizer(num_bins=100)
    v = np.arange(100) - 50
    tokenized_v = tokenizer(v)
    assert np.all(tokenized_v >= 0)
    assert np.all(tokenized_v < 100)

    # check token_offset
    tokenizer = FloatTokenizer(
        num_bins=100, use_mu_law_encode=False, token_offset=1
    )
    v = np.array([-1, 1])
    tokenized_v = tokenizer(v)
    assert tokenized_v[0] == 1
    assert tokenized_v[1] == 100


def test_tokenize_observation() -> None:
    tokenizer = FloatTokenizer(num_bins=100)
    episode = create_episode(
        observation_shape=(100,), action_size=2, length=100
    )
    assert isinstance(episode.observations, np.ndarray)
    ref_observations = tokenizer(episode.observations)
    tokenized_episode = tokenize_observation(episode, tokenizer)
    assert np.all(tokenized_episode.observations == ref_observations)
    assert np.all(tokenized_episode.actions == episode.actions)
    assert np.all(tokenized_episode.rewards == episode.rewards)
    assert tokenized_episode.terminated == episode.terminated

    # check tuple observation
    episode = create_episode(
        observation_shape=((100,), (200,)), action_size=2, length=100
    )
    ref_tuple_observations = []
    for i in range(2):
        ref_tuple_observations.append(tokenizer(episode.observations[i]))
    tokenized_episode = tokenize_observation(episode, [tokenizer, tokenizer])
    for i in range(2):
        assert np.all(
            ref_tuple_observations[i] == tokenized_episode.observations[i]
        )


def test_tokenize_action() -> None:
    tokenizer = FloatTokenizer(num_bins=100)
    episode = create_episode(
        observation_shape=(100,), action_size=2, length=100
    )
    ref_actions = tokenizer(episode.actions)
    tokenized_episode = tokenize_action(episode, tokenizer)
    assert np.all(tokenized_episode.observations == episode.observations)
    assert np.all(tokenized_episode.actions == ref_actions)
    assert np.all(tokenized_episode.rewards == episode.rewards)
    assert tokenized_episode.terminated == episode.terminated


def test_tokenize_replay_buffer() -> None:
    tokenizer = FloatTokenizer(num_bins=100)
    episode1 = create_episode(
        observation_shape=(100,), action_size=2, length=100
    )
    episode2 = create_episode(
        observation_shape=(100,), action_size=2, length=100
    )
    replay_buffer = ReplayBuffer(
        InfiniteBuffer(), episodes=[episode1, episode2]
    )

    tokenized_replay_buffer = tokenize_replay_buffer(
        replay_buffer,
        observation_tokenizer=tokenizer,
        action_tokenizer=tokenizer,
    )

    for ep, tokenized_ep in zip(
        replay_buffer.episodes, tokenized_replay_buffer.episodes
    ):
        assert isinstance(ep.observations, np.ndarray)
        assert np.all(tokenizer(ep.observations) == tokenized_ep.observations)
        assert np.all(tokenizer(ep.actions) == tokenized_ep.actions)
        assert np.all(ep.rewards == tokenized_ep.rewards)
        assert np.all(ep.terminated == tokenized_ep.terminated)
