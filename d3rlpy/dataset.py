import numpy as np
import pandas as pd

from PIL import Image


def _compute_rewards(reward_func, observations, actions, terminals):
    rewards = []
    for i in range(observations.shape[0]):
        if i == 0 or terminals[i - 1]:
            # reward will be zero at the beginning of each episode
            reward = 0.0
        else:
            obs_t = observations[i]
            obs_tm1 = observations[i - 1]
            act_t = actions[i]
            ter_t = terminals[i]
            reward = reward_func(obs_tm1, obs_t, act_t, ter_t)
        rewards.append(reward)
    return np.array(rewards)


def _load_images(paths):
    images = []
    for path in paths:
        image = np.array(Image.open(path), dtype=np.float32) / 255.0
        # currently, all images are resized to 84x84
        image = image.resize((84, 84))
        images.append(image)
    return np.array(images)


def read_csv(path,
             observation_keys,
             action_keys,
             reward_key=None,
             reward_func=None,
             episode_key='episode',
             use_image=False):
    """

    """
    df = pd.read_csv(path)

    observations = df[observation_keys].values
    if use_image:
        observations = _load_images(observations)

    actions = df[action_keys].values

    # make terminals
    terminal_indices = []
    cur_episode = None
    for index, row in df.iterrows():
        if cur_episode is None:
            cur_episode = row[episode_key]
        if cur_episode != row[episode_key] or index == df.shape[0] - 1:
            terminal_indices.append(index)
    terminals = np.zeros(df.shape[0])
    terminals[terminal_indices] = 1.0

    # make rewards
    if reward_key is not None:
        rewards = df[reward_key].values
    elif reward_func is not None:
        rewards = _compute_rewards(reward_func, observations, actions,
                                   terminals)
    else:  # binary reward
        rewards = np.zeros_like(terminals)
        rewards[terminals == 1.0] = 1.0

    return MDPDataset(observations, actions, rewards, terminals)


def _safe_size(array):
    if isinstance(array, (list, tuple)):
        return len(array)
    elif isinstance(array, np.ndarray):
        return array.shape[0]
    raise ValueError


class MDPDataset:
    """ Markov-Decision Process Dataset class.

    MDPDataset is deisnged for reinforcement learning datasets to use them like
    supervised learning datasets.

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset

        # 1000 steps of observations with shape of (100,)
        observations = np.random.random((1000, 100))
        # 1000 steps of actions with shape of (4,)
        actions = np.random.random((1000, 4))
        # 1000 steps of rewards
        rewards = np.random.random(1000)
        # 1000 steps of terminal flags
        terminals = np.random.randint(2, size=1000)

        dataset = MDPDataset(observations, actions, rewards, terminals)

    The MDPDataset object automatically splits the given data into list of
    :class:`d3rlpy.dataset.Episode` objects.
    Furthermore, the MDPDataset object behaves like a list in order to use with
    scikit-learn utilities.

    .. code-block:: python

        # returns the number of episodes
        len(dataset)

        # access to the first episode
        episode = dataset[0]

        # iterate through all episodes
        for episode in dataset:
            pass

    Args:
        observations (numpy.ndarray or list(numpy.ndarray)): N-D array. If the
            observation is a vector, the shape should be
            `(N, dim_observation)`. If the observations is an image, the shape
            should be `(N, C, H, W)`.
        actions (numpy.ndarray): N-D array. If the actions-space is
            continuous, the shape should be `(N, dim_action)`. If the
            action-space is discrete, the shpae should be `(N,)`.
        rewards (numpy.ndarray): array of scalar rewards.
        terminals (numpy.ndarray): array of binary terminal flags.
        discrete_action (bool): flag to use the given actions as discrete
            action-space actions.

    """
    def __init__(self,
                 observations,
                 actions,
                 rewards,
                 terminals,
                 discrete_action=False):
        self._observations = observations
        self._actions = np.asarray(actions)
        self._rewards = np.asarray(rewards)
        self._terminals = np.asarray(terminals)
        self.discrete_action = discrete_action
        self._episodes = None

    @property
    def observations(self):
        """ Returns the observations.

        Returns:
            (numpy.ndarray or list(numpy.ndarray)): array of observations.

        """
        return self._observations

    @property
    def actions(self):
        """ Returns the actions.

        Returns:
            numpy.ndarray: array of actions.

        """
        return self._actions

    @property
    def rewards(self):
        """ Returns the rewards.

        Returns:
            numpy.ndarray: array of rewards

        """
        return self._rewards

    @property
    def terminals(self):
        """ Returns the terminal flags.

        Returns:
            numpy.ndarray: array of terminal flags.

        """
        return self._terminals

    @property
    def episodes(self):
        """ Returns the episodes.

        Returns:
            list(d3rlpy.dataset.Episode):
                list of :class:`d3rlpy.dataset.Episode` objects.

        """
        if self._episodes is None:
            self._episodes = self._to_episodes()
        return self._episodes

    def _to_episodes(self):
        rets = []
        head_index = 0
        for i in range(_safe_size(self.observations)):
            if self._terminals[i]:
                episode = Episode(self.get_observation_shape(),
                                  self.get_action_size(),
                                  self._observations[head_index:i + 1],
                                  self._actions[head_index:i + 1],
                                  self._rewards[head_index:i + 1])
                rets.append(episode)
                head_index = i + 1
        return rets

    def size(self):
        """ Returns the number of episodes in the dataset.

        Returns:
            int: the number of episodes.

        """
        return len(self.episodes)

    def get_action_size(self):
        """ Returns dimension of action-space.

        If `discrete_action=True`, the return value will be the maximum index
        +1 in the give actions.

        Returns:
            int: dimension of action-space.

        """
        if self.discrete_action:
            return np.max(self._actions) + 1
        return self._actions.shape[1]

    def get_observation_shape(self):
        """ Returns observation shape.

        Returns:
            tuple: observation shape.

        """
        return self._observations[0].shape

    def is_action_discrete(self):
        """ Returns `discrete_action` flag.

        Returns:
            bool: `discrete_action` flag.

        """
        return self.discrete_action

    def compute_stats(self):
        """ Computes statistics of the dataset.

        .. code-block:: python

            stats = dataset.compute_stats()
            stats['return']['mean'] # mean of episode returns
            stats['return']['std'] # standard deviation of episode returns
            stats['return']['min'] # minimum of episode returns
            stats['return']['max'] # maximum of episode returns

        Returns:
            dict: statistics of the dataset.

        """
        episode_returns = []
        for episode in self.episodes:
            episode_returns.append(episode.compute_return())

        stats = {
            'return': {
                'mean': np.mean(episode_returns),
                'std': np.std(episode_returns),
                'min': np.min(episode_returns),
                'max': np.max(episode_returns),
            }
        }

        # avoid large copy when observations are huge data.
        if isinstance(self._observations, np.ndarray):
            stats['observation'] = {
                'mean': np.mean(self.observations, axis=0),
                'std': np.std(self.observations, axis=0)
            }

        return stats

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self.episodes[index]

    def __iter__(self):
        return iter(self.episodes)


class Episode:
    """ Episode class.

    This class is designed to hold data collected in a single episode.

    Episode object automatically splits data into list of
    :class:`d3rlpy.dataset.Transition` objects.
    Also Episode object behaves like a list object for ease of access to
    transitions.

    .. code-block:: python

        # return the number of transitions
        len(episode)

        # access to the first transition
        transitions = episode[0]

        # iterate through all transitions
        for transition in episode:
            pass

    Args:
        observation_shape (tuple): observation shape.
        action_size (int): dimension of action-space.
        observations (numpy.ndarray or list(numpy.ndarray)): observations.
        actions (numpy.ndarray): actions.
        rewards (numpy.ndarray): scalar rewards.
        terminals (numpy.ndarray): binary terminal flags.

    """
    def __init__(self, observation_shape, action_size, observations, actions,
                 rewards):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._transitions = None

    @property
    def observations(self):
        """ Returns the observations.

        Returns:
            (numpy.ndarray or list(numpy.ndarray)): array of observations.

        """
        return self._observations

    @property
    def actions(self):
        """ Returns the actions.

        Returns:
            numpy.ndarray: array of actions.

        """
        return self._actions

    @property
    def rewards(self):
        """ Returns the rewards.

        Returns:
            numpy.ndarray: array of rewards.

        """
        return self._rewards

    @property
    def transitions(self):
        """ Returns the transitions.

        Returns:
            list(d3rlpy.dataset.Transition):
                list of :class:`d3rlpy.dataset.Transition` objects.

        """
        if self._transitions is None:
            self._transitions = self._to_transitions()
        return self._transitions

    def _to_transitions(self):
        rets = []
        num_data = _safe_size(self._observations)
        for i in range(num_data - 1):
            observation = self._observations[i]
            action = self._actions[i]
            reward = self._rewards[i]
            next_observation = self._observations[i + 1]
            next_action = self._actions[i + 1]
            next_reward = self._rewards[i + 1]
            terminal = 1.0 if i == num_data - 2 else 0.0
            transition = Transition(self.observation_shape, self.action_size,
                                    observation, action, reward,
                                    next_observation, next_action, next_reward,
                                    terminal)
            rets.append(transition)
        return rets

    def size(self):
        """ Returns the number of transitions.

        Returns:
            int: the number of transitions.

        """
        return len(self.transitions)

    def get_observation_shape(self):
        """ Returns observation shape.

        Returns:
            tuple: observation shape.

        """
        return self.observation_shape

    def get_action_size(self):
        """ Returns dimension of action-space.

        Returns:
            int: dimension of action-space.

        """
        return self.action_size

    def compute_return(self):
        """ Computes sum of rewards.

        .. math::

            R = \sum_{i=1} r_i

        Returns:
            float: episode return.

        """
        return np.sum(self._rewards[1:])

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self.transitions[index]

    def __iter__(self):
        return iter(self.transitions)


class Transition:
    """ Transition class.

    This class is designed to hold data between two time steps, which is
    usually used as inputs of loss calculation in reinforcement learning.

    Args:
        observation_shape (tuple): observation shape.
        action_size (int): dimension of action-space.
        observation (numpy.ndarray): observation at `t`.
        action (numpy.ndarray or int): action at `t`.
        reward (float): reward at `t`.
        next_observation (numpy.ndarray): observation at `t+1`.
        next_action (numpy.ndarray or int): action at `t+1`.
        next_reward (float): reward at `t+1`.
        terminal (int): terminal flag at `t+1`.

    """
    def __init__(self, observation_shape, action_size, observation, action,
                 reward, next_observation, next_action, next_reward, terminal):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self._observation = observation
        self._action = action
        self._reward = reward
        self._next_observation = next_observation
        self._next_action = next_action
        self._next_reward = next_reward
        self._terminal = terminal

    def get_observation_shape(self):
        """ Returns observation shape.

        Returns:
            tuple: observation shape.

        """
        return self.observation_shape

    def get_action_size(self):
        """ Returns dimension of action-space.

        Returns:
            int: dimension of action-space.

        """
        return self.action_size

    @property
    def observation(self):
        """ Returns observation at `t`.

        Returns:
            numpy.ndarray: observation at `t`.

        """
        return self._observation

    @property
    def action(self):
        """ Returns action at `t`.

        Returns:
            (numpy.ndarray or int): action at `t`.

        """
        return self._action

    @property
    def reward(self):
        """ Returns reward at `t`.

        Returns:
            float: reward at `t`.

        """
        return self._reward

    @property
    def next_observation(self):
        """ Returns observation at `t+1`.

        Returns:
            numpy.ndarray: observation at `t+1`.

        """
        return self._next_observation

    @property
    def next_action(self):
        """ Returns action at `t+1`.

        Returns:
            (numpy.ndarray or int): action at `t+1`.

        """
        return self._next_action

    @property
    def next_reward(self):
        """ Returns reward at `t+1`.

        Returns:
            float: reward at `t+1`.

        """
        return self._next_reward

    @property
    def terminal(self):
        """ Returns terminal flag at `t+1`.

        Returns:
            int: terminal flag at `t+1`.

        """
        return self._terminal


class TransitionMiniBatch:
    """ mini-batch of Transition objects.

    This class is designed to hold :class:`d3rlpy.dataset.Transition` objects
    for being passed to algorithms during fitting.

    Args:
        transitions (list(d3rlpy.dataset.Transition)):
            mini-batch of transitions.

    """
    def __init__(self, transitions):
        self._transitions = transitions

        observations = []
        actions = []
        rewards = []
        next_observations = []
        next_actions = []
        next_rewards = []
        terminals = []
        for transition in transitions:
            observations.append(transition.observation)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_observations.append(transition.next_observation)
            next_actions.append(transition.next_action)
            next_rewards.append(transition.next_reward)
            terminals.append(transition.terminal)

        # convert list to ndarray and fix shapes
        self._observations = np.array(observations)
        self._actions = np.array(actions).reshape((self.size(), -1))
        self._rewards = np.array(rewards).reshape((self.size(), 1))
        self._next_observations = np.array(next_observations)
        self._next_rewards = np.array(next_rewards).reshape((self.size(), 1))
        self._next_actions = np.array(next_actions).reshape((self.size(), -1))
        self._terminals = np.array(terminals).reshape((self.size(), 1))

    @property
    def observations(self):
        """ Returns mini-batch of observations at `t`.

        Returns:
            numpy.ndarray: observations at `t`.

        """
        return self._observations

    @property
    def actions(self):
        """ Returns mini-batch of actions at `t`.

        Returns:
            numpy.ndarray: actions at `t`.

        """
        return self._actions

    @property
    def rewards(self):
        """ Returns mini-batch of rewards at `t`.

        Returns:
            numpy.ndarray: rewards at `t`.

        """
        return self._rewards

    @property
    def next_observations(self):
        """ Returns mini-batch of observations at `t+1`.

        Returns:
            numpy.ndarray: observations at `t+1`.

        """
        return self._next_observations

    @property
    def next_actions(self):
        """ Returns mini-batch of actions at `t+1`.

        Returns:
            numpy.ndarray: actions at `t+1`.

        """
        return self._next_actions

    @property
    def next_rewards(self):
        """ Returns mini-batch of rewards at `t+1`.

        Returns:
            numpy.ndarray: rewards at `t+1`.

        """
        return self._next_rewards

    @property
    def terminals(self):
        """ Returns mini-batch of terminal flags at `t+1`.

        Returns:
            numpy.ndarray: terminal flags at `t+1`.

        """
        return self._terminals

    @property
    def transitions(self):
        """ Returns transitions.

        Returns:
            d3rlpy.dataset.Transition: list of transitions.

        """
        return self._transitions

    def size(self):
        """ Returns size of mini-batch.

        Returns:
            int: mini-batch size.

        """
        return len(self._transitions)

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self._transitions[index]

    def __iter__(self):
        return iter(self._transitions)
