import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, List, Sequence, Union

import numpy as np
import torch

from ...dataset import (
    Episode,
    EpisodeBase,
    ReplayBufferBase,
    slice_observations,
)
from ...models.torch import TokenEmbedding
from ...torch_utility import torch_batch_pad_array
from ...types import NDArray, Observation

__all__ = [
    "GatoTokenEpisode",
    "GatoInputEmbedding",
    "GatoTrainingInputEmbedding",
    "GatoTokenSlicer",
    "GatoEmbeddingMiniBatch",
    "ReplayBufferWithEmbeddingKeys",
    "GatoReplayBuffer",
]


@dataclasses.dataclass(frozen=True)
class GatoTokenEpisode(Episode):
    observation_to_embedding_keys: Sequence[str]
    action_embedding_key: str
    one_step_block_size: int
    task_id: str

    @classmethod
    def from_episode(
        cls,
        episode: EpisodeBase,
        observation_to_embedding_keys: Union[str, Sequence[str]],
        action_embedding_key: str,
        token_embeddings: Dict[str, TokenEmbedding],
        task_id: str,
    ) -> "GatoTokenEpisode":
        block_size = 0

        observations: List[NDArray]
        if isinstance(observation_to_embedding_keys, str):
            assert isinstance(episode.observations, np.ndarray)
            observation_to_embedding_keys = [observation_to_embedding_keys]
            observations = [episode.observations]
        else:
            assert isinstance(episode, (tuple, list))
            observations = list(episode.observations)

        for key, observation in zip(
            observation_to_embedding_keys, observations
        ):
            token_embedding = token_embeddings[key]
            observation = observation[0][None, :]
            with torch.no_grad():
                embedding = token_embedding(observation)
            # (1, num_tokens, embed_size)
            block_size += embedding.shape[1]

        action = episode.actions[0][None, :]
        action_token_embedding = token_embeddings[action_embedding_key]
        with torch.no_grad():
            embedding = action_token_embedding(action)
        # (1, num_tokens, embed_size)
        block_size += embedding.shape[1]

        return GatoTokenEpisode(
            observations=episode.observations,
            actions=episode.actions,
            rewards=episode.rewards,
            terminated=episode.terminated,
            observation_to_embedding_keys=observation_to_embedding_keys,
            action_embedding_key=action_embedding_key,
            one_step_block_size=block_size,
            task_id=task_id,
        )


def convert_observations_to_embeddings(
    observations: Observation,
    observation_to_embedding_keys: Union[str, Sequence[str]],
    token_embeddings: Dict[str, TokenEmbedding],
) -> torch.Tensor:
    if isinstance(observations, np.ndarray):
        token_embedding = token_embeddings[observation_to_embedding_keys[0]]
        # (T, obs_num_tokens, N)
        observation_embeddings = [token_embedding(observations)]
    else:
        observation_embeddings = []
        for key, observation in zip(
            observation_to_embedding_keys, observations
        ):
            token_embedding = token_embeddings[key]
            observation_embeddings.append(token_embedding(observation))
    # (T, total_obs_num_tokens, N)
    return torch.cat(observation_embeddings, dim=1)


@dataclasses.dataclass(frozen=True)
class GatoEmbeddingMetaData:
    observation_positions: torch.Tensor  # (T, S)
    observation_masks: torch.Tensor  # (T, S, 1)
    action_masks: torch.Tensor  # (T, S, 1)


def create_embedding_metadata(
    observation_embeddings: torch.Tensor, action_embeddings: torch.Tensor
) -> GatoEmbeddingMetaData:
    num_steps, observation_token_size, _ = observation_embeddings.shape
    action_token_size = action_embeddings.shape[1]
    device = observation_embeddings.device

    # (T, total_obs_num_tokens)
    observation_positions = (
        torch.arange(observation_token_size, device=device)
        .view(1, observation_token_size)
        .tile([num_steps, 1])
    )
    # (T, S)
    observation_positions = torch.cat(
        [
            observation_positions,
            torch.zeros(
                (num_steps, action_token_size),
                device=device,
                dtype=torch.int32,
            ),
        ],
        dim=1,
    )
    # (T, S, 1)
    observation_masks = torch.cat(
        [
            torch.ones(
                (num_steps, observation_token_size, 1),
                device=device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (num_steps, action_token_size, 1),
                device=device,
                dtype=torch.float32,
            ),
        ],
        dim=1,
    )
    # (T, S, 1)
    action_masks = torch.cat(
        [
            torch.zeros(
                (num_steps, observation_token_size, 1),
                device=device,
                dtype=torch.float32,
            ),
            torch.ones(
                (num_steps, action_token_size, 1),
                device=device,
                dtype=torch.float32,
            ),
        ],
        dim=1,
    )
    return GatoEmbeddingMetaData(
        observation_positions=observation_positions,
        observation_masks=observation_masks,
        action_masks=action_masks,
    )


@dataclasses.dataclass(frozen=True)
class GatoInputEmbedding:
    embeddings: torch.Tensor  # (T, N)
    observation_positions: torch.Tensor  # (T,)
    observation_masks: torch.Tensor  # (T, 1)
    action_masks: torch.Tensor  # (T, 1)


@dataclasses.dataclass(frozen=True)
class GatoTrainingInputEmbedding(GatoInputEmbedding):
    action_tokens: torch.Tensor  # (T,)
    masks: torch.Tensor  # (T, 1)


class GatoTokenSlicer:
    r"""Standard trajectory slicer.

    This class implements a basic trajectory slicing.
    """

    def __call__(
        self,
        episode: GatoTokenEpisode,
        end_step: int,
        token_size: int,
        token_embeddings: Dict[str, TokenEmbedding],
    ) -> GatoTrainingInputEmbedding:
        num_steps = token_size // episode.one_step_block_size
        end_step = end_step + 1
        start_step = max(end_step - num_steps, 0)
        actual_num_steps = end_step - start_step

        # slice observations
        observations = slice_observations(
            episode.observations, start_step, end_step
        )
        # (T, total_obs_num_tokens, N)
        concat_observation_embeddings = convert_observations_to_embeddings(
            observations=observations,
            observation_to_embedding_keys=episode.observation_to_embedding_keys,
            token_embeddings=token_embeddings,
        )
        observation_token_size = concat_observation_embeddings.shape[1]
        device = concat_observation_embeddings.device

        # slice actions
        actions = episode.actions[start_step:end_step]
        action_token_embedding = token_embeddings[episode.action_embedding_key]
        # (T, action_num_tokens, N)
        action_embedding = action_token_embedding(actions)
        # (T, action_num_tokens)
        action_tokens = torch.tensor(
            action_token_embedding.get_tokens(actions),
            device=device,
            dtype=torch.int32,
        )

        # concat observations and actions
        # S = total_obs_num_tokens + action_num_tokens
        # (T, S, N)
        concat_embeddings = torch.cat(
            [concat_observation_embeddings, action_embedding], dim=1
        )
        metadata = create_embedding_metadata(
            observation_embeddings=concat_observation_embeddings,
            action_embeddings=action_embedding,
        )
        # (T, S)
        action_tokens = torch.cat(
            [
                torch.zeros(
                    (actual_num_steps, observation_token_size),
                    device=device,
                    dtype=torch.int32,
                ),
                action_tokens,
            ],
            dim=1,
        )

        # flatten tensors
        # (T, S, N) -> (T * S, N)
        embed_size = concat_embeddings.shape[2]
        flatten_embeddings = concat_embeddings.view(-1, embed_size)
        # (T, S) -> (T * S)
        flatten_observation_positions = metadata.observation_positions.view(-1)
        flatten_action_tokens = action_tokens.view(-1)
        # (T, S, 1) -> (T * S, 1)
        flatten_observation_masks = metadata.observation_masks.view(-1, 1)
        flatten_action_masks = metadata.action_masks.view(-1, 1)

        masks = torch.ones_like(flatten_observation_masks)

        # compute backward padding size
        pad_size = token_size - actual_num_steps * episode.one_step_block_size

        if pad_size == 0:
            return GatoTrainingInputEmbedding(
                embeddings=flatten_embeddings,
                observation_positions=flatten_observation_positions,
                observation_masks=flatten_observation_masks,
                action_masks=flatten_action_masks,
                action_tokens=flatten_action_tokens,
                masks=masks,
            )

        return GatoTrainingInputEmbedding(
            embeddings=torch_batch_pad_array(flatten_embeddings, pad_size),
            observation_positions=torch_batch_pad_array(
                flatten_observation_positions, pad_size
            ),
            observation_masks=torch_batch_pad_array(
                flatten_observation_masks, pad_size
            ),
            action_masks=torch_batch_pad_array(flatten_action_masks, pad_size),
            action_tokens=torch_batch_pad_array(
                flatten_action_tokens, pad_size
            ),
            masks=torch_batch_pad_array(masks, pad_size),
        )


@dataclasses.dataclass(frozen=True)
class GatoEmbeddingMiniBatch:
    embeddings: torch.Tensor  # (B, T, N)
    observation_positions: torch.Tensor  # (B, T)
    observation_masks: torch.Tensor  # (B, T, 1)
    action_masks: torch.Tensor  # (B, T, 1)
    action_tokens: torch.Tensor  # (B, T)
    masks: torch.Tensor  # (B, T, 1)

    @classmethod
    def from_sequences(
        cls, sequences: Sequence[GatoTrainingInputEmbedding]
    ) -> "GatoEmbeddingMiniBatch":
        embeddings = torch.stack(
            [embedding.embeddings for embedding in sequences], dim=0
        )
        observation_positions = torch.stack(
            [embedding.observation_positions for embedding in sequences], dim=0
        )
        observation_masks = torch.stack(
            [embedding.observation_masks for embedding in sequences], dim=0
        )
        action_masks = torch.stack(
            [embedding.action_masks for embedding in sequences], dim=0
        )
        action_tokens = torch.stack(
            [embedding.action_tokens for embedding in sequences], dim=0
        )
        masks = torch.stack([embedding.masks for embedding in sequences], dim=0)
        return GatoEmbeddingMiniBatch(
            embeddings=embeddings,
            observation_positions=observation_positions,
            observation_masks=observation_masks,
            action_masks=action_masks,
            action_tokens=action_tokens,
            masks=masks,
        )


@dataclasses.dataclass(frozen=True)
class ReplayBufferWithEmbeddingKeys:
    replay_buffer: ReplayBufferBase
    observation_to_embedding_keys: Union[str, Sequence[str]]
    action_embedding_key: str
    task_id: str


class GatoReplayBuffer:
    r"""Replay buffer for Gato."""
    _episodes: List[GatoTokenEpisode]
    _episodes_per_task: DefaultDict[str, List[GatoTokenEpisode]]
    _token_slicer: GatoTokenSlicer
    _token_embeddings: Dict[str, TokenEmbedding]

    def __init__(
        self,
        replay_buffers: Sequence[ReplayBufferWithEmbeddingKeys],
        token_embeddings: Dict[str, TokenEmbedding],
    ):
        self._token_slicer = GatoTokenSlicer()
        self._token_embeddings = token_embeddings
        self._episodes = []
        self._episodes_per_task = defaultdict(list)
        for replay_buffer in replay_buffers:
            for episode in replay_buffer.replay_buffer.episodes:
                token_episode = GatoTokenEpisode.from_episode(
                    episode=episode,
                    observation_to_embedding_keys=replay_buffer.observation_to_embedding_keys,
                    action_embedding_key=replay_buffer.action_embedding_key,
                    token_embeddings=token_embeddings,
                    task_id=replay_buffer.task_id,
                )
                self.append_episode(token_episode)

    def append_episode(self, episode: GatoTokenEpisode) -> None:
        self._episodes.append(episode)
        self._episodes_per_task[episode.task_id].append(episode)

    def sample_embedding_sequence(
        self, length: int
    ) -> GatoTrainingInputEmbedding:
        episode = self._episodes[int(np.random.randint(len(self._episodes)))]
        end_step = int(np.random.randint(episode.size()))
        return self._token_slicer(
            episode=episode,
            end_step=end_step,
            token_size=length,
            token_embeddings=self._token_embeddings,
        )

    def sample_embedding_mini_batch(
        self, batch_size: int, length: int
    ) -> GatoEmbeddingMiniBatch:
        embedding_sequences = [
            self.sample_embedding_sequence(length) for _ in range(batch_size)
        ]
        return GatoEmbeddingMiniBatch.from_sequences(embedding_sequences)

    def size(self) -> int:
        return sum([episode.size() for episode in self._episodes])
