from typing_extensions import Protocol

from .components import EpisodeBase, Transition
from .utils import (
    create_zero_observation,
    retrieve_observation,
    stack_recent_observations,
)

__all__ = [
    "TransitionPickerProtocol",
    "BasicTransitionPicker",
    "FrameStackTransitionPicker",
]


def _validate_index(episode: EpisodeBase, index: int) -> None:
    assert index < episode.transition_count


class TransitionPickerProtocol(Protocol):
    def __call__(self, episode: EpisodeBase, index: int) -> Transition:
        ...


class BasicTransitionPicker(TransitionPickerProtocol):
    def __call__(self, episode: EpisodeBase, index: int) -> Transition:
        _validate_index(episode, index)

        observation = retrieve_observation(episode.observations, index)
        is_terminal = index == episode.size() - 1
        if is_terminal:
            next_observation = create_zero_observation(observation)
        else:
            next_observation = retrieve_observation(
                episode.observations, index + 1
            )
        return Transition(
            observation=observation,
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=next_observation,
            terminal=float(is_terminal),
            interval=1,
        )


class FrameStackTransitionPicker(TransitionPickerProtocol):
    _n_frames: int

    def __init__(self, n_frames: int):
        self._n_frames = n_frames

    def __call__(self, episode: EpisodeBase, index: int) -> Transition:
        _validate_index(episode, index)

        observation = stack_recent_observations(
            episode.observations, index, self._n_frames
        )
        is_terminal = index == episode.size() - 1
        if is_terminal:
            next_observation = create_zero_observation(observation)
        else:
            next_observation = stack_recent_observations(
                episode.observations, index + 1, self._n_frames
            )
        return Transition(
            observation=observation,
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=next_observation,
            terminal=float(is_terminal),
            interval=1,
        )
