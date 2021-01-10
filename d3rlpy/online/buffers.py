from abc import ABCMeta, abstractmethod
from typing import Generic, List, Optional, TypeVar, Sequence

import numpy as np
import gym

from ..dataset import Episode, MDPDataset, Transition, TransitionMiniBatch
from .utility import get_action_size_from_env

T = TypeVar("T")


class FIFOQueue(Generic[T]):
    """Simple FIFO queue implementation.

    Random access of this queue object is O(1).

    """

    _maxlen: int
    _buffer: List[Optional[T]]
    _cursor: int
    _size: int

    def __init__(self, maxlen: int):
        self._maxlen = maxlen
        self._buffer = [None for _ in range(maxlen)]
        self._cursor = 0
        self._size = 0

    def append(self, item: T) -> None:
        self._buffer[self._cursor] = item
        self._cursor += 1
        if self._cursor == self._maxlen:
            self._cursor = 0
        self._size = min(self._size + 1, self._maxlen)

    def __getitem__(self, index: int) -> T:
        assert index < self._size
        item = self._buffer[index]
        assert item is not None
        return item

    def __len__(self) -> int:
        return self._size


class Buffer(metaclass=ABCMeta):
    @abstractmethod
    def append(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: float,
    ) -> None:
        """Append observation, action, reward and terminal flag to buffer.

        If the terminal flag is True, Monte-Carlo returns will be computed with
        an entire episode and the whole transitions will be appended.

        Args:
            observation (numpy.ndarray): observation.
            action (numpy.ndarray or int): action.
            reward (float): reward.
            terminal (bool or float): terminal flag.

        """

    @abstractmethod
    def append_episode(self, episode: Episode) -> None:
        """Append Episode object to buffer.

        Args:
            episode: episode.

        """

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

    @abstractmethod
    def size(self) -> int:
        """Returns the number of appended elements in buffer.

        Returns:
            the number of elements in buffer.

        """

    @abstractmethod
    def to_mdp_dataset(self) -> MDPDataset:
        """Convert replay data into static dataset.

        The length of the dataset can be longer than the length of the replay
        buffer because this conversion is done by tracing ``Transition``
        objects.

        Returns:
            MDPDataset object.

        """

    def __len__(self) -> int:
        return self.size()


class ReplayBuffer(Buffer):
    """Standard Replay Buffer.

    Args:
        maxlen (int): the maximum number of data length.
        env (gym.Env): gym-like environment to extract shape information.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes to
            initialize buffer

    """

    _prev_observation: Optional[np.ndarray]
    _prev_action: Optional[np.ndarray]
    _prev_reward: float
    _prev_transition: Optional[Transition]
    _transitions: FIFOQueue[Transition]
    _observation_shape: Sequence[int]
    _action_size: int

    def __init__(
        self,
        maxlen: int,
        env: gym.Env,
        episodes: Optional[List[Episode]] = None,
    ):
        # temporary cache to hold transitions for an entire episode
        self._prev_observation = None
        self._prev_action = None
        self._prev_reward = 0.0
        self._prev_transition = None

        self._transitions = FIFOQueue(maxlen=maxlen)

        # extract shape information
        self._observation_shape = env.observation_space.shape
        self._action_size = get_action_size_from_env(env)

        # add initial transitions
        if episodes:
            for episode in episodes:
                self.append_episode(episode)

    def append(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: float,
    ) -> None:
        # validation
        assert observation.shape == self._observation_shape
        if isinstance(action, np.ndarray):
            assert action.shape[0] == self._action_size
        else:
            action = int(action)
            assert action < self._action_size

        # create Transition object
        if self._prev_observation is not None:
            if isinstance(terminal, bool):
                terminal = 1.0 if terminal else 0.0

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
                prev_transition=self._prev_transition,
            )

            if self._prev_transition:
                self._prev_transition.next_transition = transition

            self._transitions.append(transition)

            self._prev_transition = transition

        self._prev_observation = observation
        self._prev_action = action
        self._prev_reward = reward

        if terminal:
            self._prev_observation = None
            self._prev_action = None
            self._prev_reward = 0.0
            self._prev_transition = None

    def append_episode(self, episode: Episode) -> None:
        assert episode.get_observation_shape() == self._observation_shape
        assert episode.get_action_size() == self._action_size
        for transition in episode.transitions:
            self._transitions.append(transition)

    def sample(
        self,
        batch_size: int,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
    ) -> TransitionMiniBatch:
        indices = np.random.choice(len(self._transitions), batch_size)
        transitions = [self._transitions[index] for index in indices]
        return TransitionMiniBatch(transitions, n_frames, n_steps, gamma)

    def size(self) -> int:
        return len(self._transitions)

    def to_mdp_dataset(self) -> MDPDataset:
        head_transitions: List[Transition] = []

        # get the first head transition
        if self._transitions[0].prev_transition:
            transition = self._transitions[0]
            while True:
                if transition.prev_transition:
                    transition = transition.prev_transition
                else:
                    head_transitions.append(transition)
                    break
        else:
            head_transitions.append(self._transitions[0])

        for i in range(1, self.size()):
            # check prev_transition=None
            transition = self._transitions[i]
            if transition.prev_transition is None:
                head_transitions.append(transition)

        observations = []
        actions = []
        rewards = []
        terminals = []
        for transition in head_transitions:
            # stack data
            while True:
                observations.append(transition.observation)
                actions.append(transition.action)
                rewards.append(transition.reward)
                terminals.append(0.0)
                if transition.next_transition:
                    transition = transition.next_transition
                else:
                    observations.append(transition.next_observation)
                    actions.append(transition.next_action)
                    rewards.append(transition.next_reward)
                    terminals.append(1.0)
                    break

        if len(self._observation_shape) == 3:
            observations = np.asarray(observations, dtype=np.uint8)
        else:
            observations = np.asarray(observations, dtype=np.float32)

        return MDPDataset(observations, actions, rewards, terminals)
