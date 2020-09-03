import numpy as np

from abc import ABCMeta, abstractmethod
from collections import deque
from random import sample
from ..gpu import Device
from ..dataset import Transition, TransitionMiniBatch, _numpy_to_tensor
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
    def sample(self, batch_size):
        """ Returns sampled mini-batch of transitions.

        Args:
            batch_size (int): mini-batch size.

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
        observations (list(numpy.ndarray)):
            list of observations in the current episode.
        actions (list(numpy.ndarray)): list of actions in the current episode.
        rewards (list(float)): list of rewards in the current episode.
        transitions (collections.deque): list of transitions.
        observation_shape (tuple): observation shape.
        action_size (int): action size.
        as_tensor (bool): flag to hold observations as ``torch.Tensor``.
        device (d3rlpy.gpu.Device): gpu device.

    """
    def __init__(self, maxlen, env, as_tensor=False, device=None):
        # temporary cache to hold transitions for an entire episode
        self.observations = []
        self.actions = []
        self.rewards = []

        self.transitions = deque(maxlen=maxlen)

        # extract shape information
        self.observation_shape = env.observation_space.shape
        self.action_size = get_action_size_from_env(env)

        # data type option
        if isinstance(device, int):
            self.device = Device(device)
        else:
            self.device = device
        self.as_tensor = as_tensor

    def append(self, observation, action, reward, terminal):
        # validation
        assert observation.shape == self.observation_shape
        if isinstance(action, np.ndarray):
            assert action.shape[0] == self.action_size
        else:
            action = int(action)
            assert action < self.action_size

        # numpy.ndarray to PyTorch conversion
        if self.as_tensor:
            observation = _numpy_to_tensor(observation, self.device)

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

        # commit data in the current episode
        if terminal:
            self._commit()

    def _commit(self):
        episode_size = len(self.observations)
        prev_transition = None
        for i in range(episode_size - 1):
            observation = self.observations[i]
            action = self.actions[i]
            reward = self.rewards[i]
            next_observation = self.observations[i + 1]
            next_action = self.actions[i + 1]
            next_reward = self.rewards[i + 1]
            terminal = 1.0 if i == episode_size - 2 else 0.0

            transition = Transition(observation_shape=self.observation_shape,
                                    action_size=self.action_size,
                                    observation=observation,
                                    action=action,
                                    reward=reward,
                                    next_observation=next_observation,
                                    next_action=next_action,
                                    next_reward=next_reward,
                                    terminal=terminal,
                                    prev_transition=prev_transition)

            # set transition to the next pointer
            if prev_transition:
                prev_transition.next_transition = transition

            prev_transition = transition

            self.transitions.append(transition)

        # refresh cache
        self.observations = []
        self.actions = []
        self.rewards = []

    def sample(self, batch_size):
        return TransitionMiniBatch(sample(self.transitions, batch_size))

    def size(self):
        return len(self.transitions)

    def __len__(self):
        return self.size()
