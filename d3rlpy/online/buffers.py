import numpy as np

from abc import ABCMeta, abstractmethod
from ..dataset import Transition, TransitionMiniBatch
from .utility import get_action_size_from_env


class Buffer(metaclass=ABCMeta):
    @abstractmethod
    def append(self, observation, action, reward, terminal):
        """ Append observation, action, reward and terminal flag to buffer.

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

    Attributes:
        maxlen (int): the maximum number of data length
        observations (list(numpy.ndarray)): list of observations.
        actions (list(numpy.ndarray) or list(int)): list of actions.
        rewards (list(float)): list of rewards.
        terminals (list(float)): list of terminal flags.
        cursor (int): current cursor pointing to list location to insert.
        observation_shape (tuple): observation shape.
        action_size (int): action size.

    """
    def __init__(self, maxlen, env):
        from gym.spaces import Discrete

        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []

        self.maxlen = maxlen
        self.cursor = 0

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

        # bool to float conversion
        if isinstance(terminal, bool):
            terminal = 1.0 if terminal else 0.0

        if self.size() < self.maxlen:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            self.cursor += 1
        else:
            self.observations[self.cursor] = observation
            self.actions[self.cursor] = action
            self.rewards[self.cursor] = reward
            self.terminals[self.cursor] = terminal
            self.cursor += 1

        if self.cursor >= self.maxlen:
            self.cursor = 0

    def sample(self, batch_size):
        transitions = []

        while len(transitions) < batch_size:
            index = np.random.randint(low=1, high=self.size())

            # skip if index points to the beginning of an episode.
            if self.terminals[index - 1]:
                continue

            observation = self.observations[index - 1]
            action = self.actions[index - 1]
            reward = self.rewards[index - 1]
            next_observation = self.observations[index]
            next_action = self.actions[index]
            next_reward = self.rewards[index]
            terminal = self.terminals[index]

            transition = Transition(self.observation_shape, self.action_size,
                                    observation, action, reward,
                                    next_observation, next_action, next_reward,
                                    terminal)

            transitions.append(transition)

        return TransitionMiniBatch(transitions)

    def size(self):
        return len(self.observations)

    def __len__(self):
        return self.size()
