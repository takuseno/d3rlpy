import numpy as np
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
    "MultiStepTransitionPicker",
]


def _validate_index(episode: EpisodeBase, index: int) -> None:
    assert index < episode.transition_count


class TransitionPickerProtocol(Protocol):
    r"""Interface of TransitionPicker."""

    def __call__(self, episode: EpisodeBase, index: int) -> Transition:
        r"""Returns transition specified by ``index``.

        Args:
            episode: Episode.
            index: Index at the target transition.

        Returns:
            Transition.
        """
        raise NotImplementedError


class BasicTransitionPicker(TransitionPickerProtocol):
    r"""Standard transition picker.

    This class implements a basic transition picking.

    Args:
        gamma (float): Discount factor to compute return-to-go.
    """

    _gamma: float

    def __init__(self, gamma: float = 0.99):
        self._gamma = gamma

    def __call__(self, episode: EpisodeBase, index: int) -> Transition:
        _validate_index(episode, index)

        observation = retrieve_observation(episode.observations, index)
        is_terminal = episode.terminated and index == episode.size() - 1
        if is_terminal:
            next_observation = create_zero_observation(observation)
        else:
            next_observation = retrieve_observation(
                episode.observations, index + 1
            )

        # compute return-to-go
        length = episode.size() - index
        cum_gammas = np.expand_dims(self._gamma ** np.arange(length), axis=1)
        return_to_go = np.sum(cum_gammas * episode.rewards[index:], axis=0)

        return Transition(
            observation=observation,
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=next_observation,
            return_to_go=return_to_go,
            terminal=float(is_terminal),
            interval=1,
        )


class FrameStackTransitionPicker(TransitionPickerProtocol):
    r"""Frame-stacking transition picker.

    This class implements the frame-stacking logic. The observations are
    stacked with the last ``n_frames-1`` frames. When ``index`` specifies
    timestep below ``n_frames``, those frames are padded by zeros.

    .. code-block:: python

        episode = Episode(
            observations=np.random.random((100, 1, 84, 84)),
            actions=np.random.random((100, 2)),
            rewards=np.random.random((100, 1)),
            terminated=False,
        )

        frame_stacking_picker = FrameStackTransitionPicker(n_frames=4)
        transition = frame_stacking_picker(episode, 10)

        transition.observation.shape == (4, 84, 84)

    Args:
        n_frames (int): Number of frames to stack.
        gamma (float): Discount factor to compute return-to-go.
    """
    _n_frames: int
    _gamma: float

    def __init__(self, n_frames: int, gamma: float = 0.99):
        assert n_frames > 0
        self._n_frames = n_frames
        self._gamma = gamma

    def __call__(self, episode: EpisodeBase, index: int) -> Transition:
        _validate_index(episode, index)

        observation = stack_recent_observations(
            episode.observations, index, self._n_frames
        )
        is_terminal = episode.terminated and index == episode.size() - 1
        if is_terminal:
            next_observation = create_zero_observation(observation)
        else:
            next_observation = stack_recent_observations(
                episode.observations, index + 1, self._n_frames
            )

        # compute return-to-go
        length = episode.size() - index
        cum_gammas = np.expand_dims(self._gamma ** np.arange(length), axis=1)
        return_to_go = np.sum(cum_gammas * episode.rewards[index:], axis=0)

        return Transition(
            observation=observation,
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=next_observation,
            return_to_go=return_to_go,
            terminal=float(is_terminal),
            interval=1,
        )


class MultiStepTransitionPicker(TransitionPickerProtocol):
    r"""Multi-step transition picker.

    This class implements transition picking for the multi-step TD error.
    ``reward`` is computed as a multi-step discounted return.

    Args:
        n_steps: Delta timestep between ``observation`` and
            ``net_observation``.
        gamma: Discount factor to compute a multi-step return.
    """
    _n_steps: int
    _gamma: float

    def __init__(self, n_steps: int, gamma: float):
        self._n_steps = n_steps
        self._gamma = gamma

    def __call__(self, episode: EpisodeBase, index: int) -> Transition:
        _validate_index(episode, index)

        observation = retrieve_observation(episode.observations, index)

        # get observation N-step ahead
        if episode.terminated:
            next_index = min(index + self._n_steps, episode.size())
            is_terminal = next_index == episode.size()
            if is_terminal:
                next_observation = create_zero_observation(observation)
            else:
                next_observation = retrieve_observation(
                    episode.observations, next_index
                )
        else:
            is_terminal = False
            next_index = min(index + self._n_steps, episode.size() - 1)
            next_observation = retrieve_observation(
                episode.observations, next_index
            )

        # compute return-to-go
        length = episode.size() - index
        cum_gammas = np.expand_dims(self._gamma ** np.arange(length), axis=1)
        return_to_go = np.sum(cum_gammas * episode.rewards[index:], axis=0)

        # compute multi-step return
        interval = next_index - index
        cum_gammas = np.expand_dims(self._gamma ** np.arange(interval), axis=1)
        ret = np.sum(episode.rewards[index:next_index] * cum_gammas, axis=0)

        return Transition(
            observation=observation,
            action=episode.actions[index],
            reward=ret,
            next_observation=next_observation,
            return_to_go=return_to_go,
            terminal=float(is_terminal),
            interval=interval,
        )
