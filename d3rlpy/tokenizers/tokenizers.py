from typing import Optional, Sequence, Union

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from ..dataset import Episode, EpisodeBase, InfiniteBuffer, ReplayBuffer
from ..types import Float32NDArray, Int32NDArray, NDArray, Observation
from .utils import mu_law_decode, mu_law_encode

__all__ = [
    "Tokenizer",
    "FloatTokenizer",
    "tokenize_observation",
    "tokenize_action",
    "tokenize_replay_buffer",
]


@runtime_checkable
class Tokenizer(Protocol):
    def __call__(self, x: NDArray) -> NDArray:
        ...

    def decode(self, y: Int32NDArray) -> NDArray:
        ...


class FloatTokenizer(Tokenizer):
    _bins: Float32NDArray
    _use_mu_law_encode: bool
    _mu: float
    _basis: float
    _token_offset: int

    def __init__(
        self,
        num_bins: int,
        minimum: float = -1.0,
        maximum: float = 1.0,
        use_mu_law_encode: bool = True,
        mu: float = 100.0,
        basis: float = 256.0,
        token_offset: int = 0,
    ):
        self._bins = np.array(
            (maximum - minimum) * np.arange(num_bins) / num_bins + minimum,
            dtype=np.float32,
        )
        self._use_mu_law_encode = use_mu_law_encode
        self._mu = mu
        self._basis = basis
        self._token_offset = token_offset

    def __call__(self, x: NDArray) -> Int32NDArray:
        if self._use_mu_law_encode:
            x = mu_law_encode(x, self._mu, self._basis)
        return np.digitize(x, self._bins) - 1 + self._token_offset

    def decode(self, y: Int32NDArray) -> NDArray:
        x = self._bins[y - self._token_offset]
        if self._use_mu_law_encode:
            x = mu_law_decode(x, mu=self._mu, basis=self._basis)
        return x  # type: ignore


def tokenize_observation(
    episode: EpisodeBase, tokenizer: Union[Tokenizer, Sequence[Tokenizer]]
) -> Episode:
    assert isinstance(episode, Episode)
    tokenized_observations: Observation
    if len(episode.observation_signature.shape) > 1:
        assert isinstance(episode.observations, (list, tuple))
        assert isinstance(tokenizer, (list, tuple))
        tokenized_observations = [
            tok(np.reshape(obs, [episode.size(), -1]))
            for obs, tok in zip(episode.observations, tokenizer)
        ]
    else:
        assert isinstance(episode.observations, np.ndarray)
        assert isinstance(tokenizer, Tokenizer)
        tokenized_observations = tokenizer(
            np.reshape(episode.observations, [episode.size(), -1])
        )
    return Episode(
        observations=tokenized_observations,
        actions=episode.actions,
        rewards=episode.rewards,
        terminated=episode.terminated,
    )


def tokenize_action(episode: EpisodeBase, tokenizer: Tokenizer) -> Episode:
    assert isinstance(episode, Episode)
    return Episode(
        observations=episode.observations,
        actions=tokenizer(episode.actions),
        rewards=episode.rewards,
        terminated=episode.terminated,
    )


def tokenize_replay_buffer(
    replay_buffer: ReplayBuffer,
    observation_tokenizer: Optional[
        Union[Tokenizer, Sequence[Tokenizer]]
    ] = None,
    action_tokenizer: Optional[Tokenizer] = None,
) -> ReplayBuffer:
    episodes = replay_buffer.episodes
    if observation_tokenizer:
        episodes = [
            tokenize_observation(episode, observation_tokenizer)
            for episode in episodes
        ]
    if action_tokenizer:
        episodes = [
            tokenize_action(episode, action_tokenizer) for episode in episodes
        ]
    return ReplayBuffer(InfiniteBuffer(), episodes=episodes)
