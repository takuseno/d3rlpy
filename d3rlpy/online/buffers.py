import numpy as np

from abc import ABCMeta, abstractmethod
from collections import deque
from random import sample
from ..dataset import Transition, TransitionMiniBatch
from .utility import get_action_size_from_env


class Buffer(metaclass=ABCMeta):
    @abstractmethod
    def append(self, observation, action, reward, terminal):
        """ Append observation, action, reward and terminal flag to buffer.

        If the terminal flag is True, Monte-Carlo returns will be computed with
        an entire episode and the whole transitions will be appended.

        Args:
            observation (numpy.ndarray): observation.
            action (numpy.ndarray or int): action.
            reward (float): reward.
            terminal (bool or float): terminal flag.

        """
        pass

    @abstractmethod
    def sample(self, batch_size, n_frames=1):
        """ Returns sampled mini-batch of transitions.

        If observation is image, you can stack arbitrary frames via
        ``n_frames``.

        .. code-block:: python

            buffer.observation_shape == (3, 84, 84)

            # stack 4 frames
            batch = buffer.sample(batch_size=32, n_frames=4)

            batch.observations.shape == (32, 12, 84, 84)

        Args:
            batch_size (int): mini-batch size.
            n_frames (int):
                the number of frames to stack for image observation.

        Returns:
            d3rlpy.dataset.TransitionMiniBatch: mini-batch.

        """
        pass

    @abstractmethod
    def size(self):
        """ Returns the number of appended elements in buffer.

        Returns:
            int: the number of elements in buffer.

        """
        pass


class ReplayBuffer(Buffer):
    """ Standard Replay Buffer.

    Args:
        maxlen (int): the maximum number of data length.
        env (gym.Env): gym-like environment to extract shape information.
        as_tensor (bool): flag to hold observations as ``torch.Tensor``.
        device (d3rlpy.gpu.Device or int): gpu device or device id for tensor.

    Attributes:
        prev_observation (numpy.ndarray): previously appended observation.
        prev_action (numpy.ndarray or int): previously appended action.
        prev_reward (float): previously appended reward.
        prev_transition (d3rlpy.dataset.Transition):
            previously appended transition.
        transitions (collections.deque): list of transitions.
        observation_shape (tuple): observation shape.
        action_size (int): action size.

    """
    def __init__(self, maxlen, env):
        # temporary cache to hold transitions for an entire episode
        self.prev_observation = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_transition = None

        self.transitions = deque(maxlen=maxlen)

        # extract shape information
        self.observation_shape = env.observation_space.shape
        self.action_size = get_action_size_from_env(env)

    def append(self, observation, action, reward, terminal):
        # validation
        assert observation.shape == self.observation_shape
        if isinstance(action, np.ndarray):
            assert action.shape[0] == self.action_size
        else:
            action = int(action)
            assert action < self.action_size

        # create Transition object
        if self.prev_observation is not None:
            if isinstance(terminal, bool):
                terminal = 1.0 if terminal else 0.0

            transition = Transition(observation_shape=self.observation_shape,
                                    action_size=self.action_size,
                                    observation=self.prev_observation,
                                    action=self.prev_action,
                                    reward=self.prev_reward,
                                    next_observation=observation,
                                    next_action=action,
                                    next_reward=reward,
                                    terminal=terminal,
                                    prev_transition=self.prev_transition)

            if self.prev_transition:
                self.prev_transition.next_transition = transition

            self.transitions.append(transition)

            self.prev_transition = transition

        self.prev_observation = observation
        self.prev_action = action
        self.prev_reward = reward

        if terminal:
            self.prev_observation = None
            self.prev_action = None
            self.prev_reward = None
            self.prev_transition = None

    def sample(self, batch_size, n_frames=1):
        transitions = sample(self.transitions, batch_size)
        return TransitionMiniBatch(transitions, n_frames)

    def size(self):
        return len(self.transitions)

    def __len__(self):
        return self.size()
