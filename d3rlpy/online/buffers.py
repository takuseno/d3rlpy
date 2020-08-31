import numpy as np

from abc import ABCMeta, abstractmethod
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
        gamma (float): discount factor to compute Monte-Carlo returns.

    Attributes:
        maxlen (int): the maximum number of data length
        gamma (float): discount factor to compute Monte-Carlo returns.
        cur_observations (list(numpy.ndarray)):
            list of observations in the current episode.
        cur_actions (list(numpy.ndarray)):
            list of actions in the current episode.
        cur_rewards (list(float)):
            list of rewards in the current episode.
        observations (list(numpy.ndarray)): list of observations.
        actions (list(numpy.ndarray) or list(int)): list of actions.
        rewards (list(float)): list of rewards.
        terminals (list(float)): list of terminal flags.
        cursor (int): current cursor pointing to list location to insert.
        observation_shape (tuple): observation shape.
        action_size (int): action size.

    """
    def __init__(self, maxlen, env, gamma=0.99):
        # temporary cache to hold transitions for an entire episode
        self.cur_observations = []
        self.cur_actions = []
        self.cur_rewards = []

        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.consequent_observations = []
        self.returns = []

        self.maxlen = maxlen
        self.gamma = gamma
        self.cursor = 0
        self.episode_head = 0

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

        self.cur_observations.append(observation)
        self.cur_actions.append(action)
        self.cur_rewards.append(reward)

        # commit data in the current episode
        if terminal:
            self._commit()

    def _commit(self):
        episode_size = len(self.cur_observations)
        for i in range(episode_size):
            observation = self.cur_observations[i]
            action = self.cur_actions[i]
            reward = self.cur_rewards[i]
            terminal = 1.0 if i == episode_size - 1 else 0.0
            consq_observations = self.cur_observations[i + 1:]

            # compute Monte-Carlo returns
            R = 0.0
            returns = []
            for j in range(episode_size - i - 1):
                R += (self.gamma**j) * self.cur_rewards[i + j + 1]
                returns.append(R)

            if self.size() < self.maxlen:
                self.observations.append(observation)
                self.actions.append(action)
                self.rewards.append(reward)
                self.terminals.append(terminal)
                self.returns.append(returns)
                self.consequent_observations.append(consq_observations)
                self.cursor += 1
            else:
                self.observations[self.cursor] = observation
                self.actions[self.cursor] = action
                self.rewards[self.cursor] = reward
                self.terminals[self.cursor] = terminal
                self.returns[self.cursor] = returns
                self.consequent_observations[self.cursor] = consq_observations
                self.cursor += 1

            if self.cursor >= self.maxlen:
                self.cursor = 0

        # refresh cache
        self.cur_observations = []
        self.cur_actions = []
        self.cur_rewards = []

    def sample(self, batch_size):
        transitions = []

        while len(transitions) < batch_size:
            index = int(np.random.randint(low=1, high=self.size()))

            # skip if index points to the beginning of an episode.
            if self.terminals[index - 1]:
                continue

            transitions.append(self._get_transition(index - 1))

        return TransitionMiniBatch(transitions)

    def _get_transition(self, index):
        assert not self.terminals[index]

        observation = self.observations[index]
        action = self.actions[index]
        reward = self.rewards[index]
        next_observation = self.observations[index + 1]
        next_action = self.actions[index + 1]
        next_reward = self.rewards[index + 1]
        terminal = self.terminals[index + 1]
        returns = self.returns[index]
        consequent_observations = self.consequent_observations[index]

        transition = Transition(
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            next_action=next_action,
            next_reward=next_reward,
            terminal=terminal,
            returns=returns,
            consequent_observations=consequent_observations)

        return transition

    def size(self):
        return len(self.observations)

    def __len__(self):
        return self.size()
