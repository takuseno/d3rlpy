from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence, cast

import gym
import numpy as np

from ..containers import FIFOQueue
from ..dataset import (
    Episode,
    MDPDataset,
    Transition,
    TransitionMiniBatch,
    trace_back_and_clear,
)
from ..envs import BatchEnv
from .utility import get_action_size_from_env


class _Buffer(metaclass=ABCMeta):

    _transitions: FIFOQueue[Transition]
    _observation_shape: Sequence[int]
    _action_size: int
    _create_mask: bool
    _mask_size: int

    def __init__(
        self,
        maxlen: int,
        env: Optional[gym.Env] = None,
        episodes: Optional[List[Episode]] = None,
        create_mask: bool = False,
        mask_size: int = 1,
    ):
        def drop_callback(transition: Transition) -> None:
            # remove links when dropping the last transition
            if transition.next_transition is None:
                trace_back_and_clear(transition)

        self._transitions = FIFOQueue(maxlen, drop_callback)

        # extract shape information
        if env:
            observation_shape = env.observation_space.shape
            action_size = get_action_size_from_env(env)
        elif episodes:
            observation_shape = episodes[0].get_observation_shape()
            action_size = episodes[0].get_action_size()
        else:
            raise ValueError("env or episodes are required to determine shape.")

        self._observation_shape = observation_shape
        self._action_size = action_size
        self._create_mask = create_mask
        self._mask_size = mask_size

        # add initial transitions
        if episodes:
            for episode in episodes:
                self.append_episode(episode)

    def append_episode(self, episode: Episode) -> None:
        """Append Episode object to buffer.

        Args:
            episode: episode.

        """
        assert episode.get_observation_shape() == self._observation_shape
        assert episode.get_action_size() == self._action_size
        for transition in episode.transitions:
            self._transitions.append(transition)
            # add mask if necessary
            if self._create_mask and transition.mask is None:
                transition.mask = np.random.randint(2, size=self._mask_size)

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
    ) -> TransitionMiniBatch:
        """Returns sampled mini-batch of transitions.

        If observation is image, you can stack arbitrary frames via
        ``n_frames``.

        .. code-block:: python

            buffer.observation_shape == (3, 84, 84)

            # stack 4 frames
            batch = buffer.sample(batch_size=32, n_frames=4)

            batch.observations.shape == (32, 12, 84, 84)

        Args:
            batch_size: mini-batch size.
            n_frames: the number of frames to stack for image observation.
            n_steps: the number of steps before the next observation.
            gamma: discount factor used in N-step return calculation.

        Returns:
            mini-batch.

        """

    def size(self) -> int:
        """Returns the number of appended elements in buffer.

        Returns:
            the number of elements in buffer.

        """
        return len(self._transitions)

    def to_mdp_dataset(self) -> MDPDataset:
        """Convert replay data into static dataset.

        The length of the dataset can be longer than the length of the replay
        buffer because this conversion is done by tracing ``Transition``
        objects.

        Returns:
            MDPDataset object.

        """
        # get the last transitions
        tail_transitions: List[Transition] = []
        for transition in self._transitions:
            if transition.next_transition is None:
                tail_transitions.append(transition)

        observations = []
        actions = []
        rewards = []
        terminals = []
        episode_terminals = []
        for transition in tail_transitions:

            # trace transition to the beginning
            episode_transitions: List[Transition] = []
            while True:
                episode_transitions.append(transition)
                if transition.prev_transition is None:
                    break
                transition = transition.prev_transition
            episode_transitions.reverse()

            # stack data
            for episode_transition in episode_transitions:
                observations.append(episode_transition.observation)
                actions.append(episode_transition.action)
                rewards.append(episode_transition.reward)
                terminals.append(0.0)
                episode_terminals.append(0.0)
            observations.append(episode_transitions[-1].next_observation)
            actions.append(episode_transitions[-1].next_action)
            rewards.append(episode_transitions[-1].next_reward)
            terminals.append(episode_transitions[-1].terminal)
            episode_terminals.append(1.0)

        if len(self._observation_shape) == 3:
            observations = np.asarray(observations, dtype=np.uint8)
        else:
            observations = np.asarray(observations, dtype=np.float32)

        return MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            episode_terminals=episode_terminals,
            create_mask=self._create_mask,
            mask_size=self._mask_size,
        )

    def __len__(self) -> int:
        return self.size()

    @property
    def transitions(self) -> FIFOQueue[Transition]:
        """Returns a FIFO queue of transitions.

        Returns:
            d3rlpy.online.buffers.FIFOQueue: FIFO queue of transitions.

        """
        return self._transitions


class Buffer(_Buffer):
    @abstractmethod
    def append(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: float,
        clip_episode: Optional[bool] = None,
    ) -> None:
        """Append observation, action, reward and terminal flag to buffer.

        If the terminal flag is True, Monte-Carlo returns will be computed with
        an entire episode and the whole transitions will be appended.

        Args:
            observation: observation.
            action: action.
            reward: reward.
            terminal: terminal flag.
            clip_episode: flag to clip the current episode. If ``None``, the
                episode is clipped based on ``terminal``.

        """


class BatchBuffer(_Buffer):
    @abstractmethod
    def append(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        clip_episodes: Optional[np.ndarray] = None,
    ) -> None:
        """Append observation, action, reward and terminal flag to buffer.

        If the terminal flag is True, Monte-Carlo returns will be computed with
        an entire episode and the whole transitions will be appended.

        Args:
            observations: observation.
            actions: action.
            rewards: reward.
            terminals: terminal flag.
            clip_episodes: flag to clip the current episode. If ``None``, the
                episode is clipped based on ``terminal``.

        """


class BasicSampleMixin:

    _transitions: FIFOQueue[Transition]

    def sample(
        self,
        batch_size: int,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
    ) -> TransitionMiniBatch:
        indices = np.random.choice(len(self._transitions), batch_size)
        transitions = [self._transitions[index] for index in indices]
        batch = TransitionMiniBatch(transitions, n_frames, n_steps, gamma)
        return batch


class ReplayBuffer(BasicSampleMixin, Buffer):
    """Standard Replay Buffer.

    Args:
        maxlen (int): the maximum number of data length.
        env (gym.Env): gym-like environment to extract shape information.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes to
            initialize buffer.
        create_mask (bool): flag to create bootstrapping mask.
        mask_size (int): ensemble size for binary mask.

    """

    _prev_observation: Optional[np.ndarray]
    _prev_action: Optional[np.ndarray]
    _prev_reward: float
    _prev_transition: Optional[Transition]

    def __init__(
        self,
        maxlen: int,
        env: Optional[gym.Env] = None,
        episodes: Optional[List[Episode]] = None,
        create_mask: bool = False,
        mask_size: int = 1,
    ):
        super().__init__(maxlen, env, episodes, create_mask, mask_size)
        self._prev_observation = None
        self._prev_action = None
        self._prev_reward = 0.0
        self._prev_transition = None

    def append(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: float,
        clip_episode: Optional[bool] = None,
    ) -> None:
        # if None, use terminal
        if clip_episode is None:
            clip_episode = bool(terminal)

        # validation
        assert observation.shape == self._observation_shape
        if isinstance(action, np.ndarray):
            assert action.shape[0] == self._action_size
        else:
            action = int(action)
            assert action < self._action_size
        # not allow terminal=True and clip_episode=False
        assert not (terminal and not clip_episode)

        # create Transition object
        if self._prev_observation is not None:
            if isinstance(terminal, bool):
                terminal = 1.0 if terminal else 0.0

            # create binary mask
            if self._create_mask:
                mask = np.random.randint(2, size=self._mask_size)
            else:
                mask = None

            transition = Transition(
                observation_shape=self._observation_shape,
                action_size=self._action_size,
                observation=self._prev_observation,
                action=self._prev_action,
                reward=self._prev_reward,
                next_observation=observation,
                next_action=action,
                next_reward=reward,
                terminal=terminal,
                mask=mask,
                prev_transition=self._prev_transition,
            )

            if self._prev_transition:
                self._prev_transition.next_transition = transition

            self._transitions.append(transition)
            self._prev_transition = transition

        self._prev_observation = observation
        self._prev_action = action
        self._prev_reward = reward

        if clip_episode:
            self._prev_observation = None
            self._prev_action = None
            self._prev_reward = 0.0
            self._prev_transition = None


class BatchReplayBuffer(BasicSampleMixin, BatchBuffer):
    """Standard Replay Buffer for batch training.

    Args:
        maxlen (int): the maximum number of data length.
        n_envs (int): the number of environments.
        env (gym.Env): gym-like environment to extract shape information.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes to
            initialize buffer
        create_mask (bool): flag to create bootstrapping mask.
        mask_size (int): ensemble size for binary mask.

    """

    _n_envs: int
    _prev_observations: List[Optional[np.ndarray]]
    _prev_actions: List[Optional[np.ndarray]]
    _prev_rewards: List[Optional[np.ndarray]]
    _prev_transitions: List[Optional[Transition]]

    def __init__(
        self,
        maxlen: int,
        env: BatchEnv,
        episodes: Optional[List[Episode]] = None,
        create_mask: bool = False,
        mask_size: int = 1,
    ):
        super().__init__(maxlen, env, episodes, create_mask, mask_size)
        self._n_envs = len(env)
        self._prev_observations = [None for _ in range(len(env))]
        self._prev_actions = [None for _ in range(len(env))]
        self._prev_rewards = [None for _ in range(len(env))]
        self._prev_transitions = [None for _ in range(len(env))]

    def append(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        clip_episodes: Optional[np.ndarray] = None,
    ) -> None:
        # if None, use terminal
        if clip_episodes is None:
            clip_episodes = terminals

        # validation
        assert observations.shape == (self._n_envs, *self._observation_shape)
        if actions.ndim == 2:
            assert actions.shape == (self._n_envs, self._action_size)
        else:
            assert actions.shape == (self._n_envs,)
        assert rewards.shape == (self._n_envs,)
        assert terminals.shape == (self._n_envs,)
        # not allow terminal=True and clip_episode=False
        assert np.all(terminals - clip_episodes < 1)

        # create Transition objects
        for i in range(self._n_envs):
            if self._prev_observations[i] is not None:

                prev_observation = self._prev_observations[i]
                prev_action = self._prev_actions[i]
                prev_reward = cast(np.ndarray, self._prev_rewards[i])
                prev_transition = self._prev_transitions[i]

                # create binary mask
                if self._create_mask:
                    mask = np.random.randint(2, size=self._mask_size)
                else:
                    mask = None

                transition = Transition(
                    observation_shape=self._observation_shape,
                    action_size=self._action_size,
                    observation=prev_observation,
                    action=prev_action,
                    reward=float(prev_reward),
                    next_observation=observations[i],
                    next_action=actions[i],
                    next_reward=float(rewards[i]),
                    terminal=float(terminals[i]),
                    mask=mask,
                    prev_transition=prev_transition,
                )

                if prev_transition:
                    prev_transition.next_transition = transition

                self._transitions.append(transition)
                self._prev_transitions[i] = transition

            self._prev_observations[i] = observations[i]
            self._prev_actions[i] = actions[i]
            self._prev_rewards[i] = rewards[i]

            if clip_episodes[i]:
                self._prev_observations[i] = None
                self._prev_actions[i] = None
                self._prev_rewards[i] = None
                self._prev_transitions[i] = None
