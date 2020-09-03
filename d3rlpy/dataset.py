import numpy as np
import torch
import h5py

from .gpu import Device


def _safe_size(array):
    if isinstance(array, (list, tuple)):
        return len(array)
    elif isinstance(array, (np.ndarray, torch.Tensor)):
        return array.shape[0]
    raise ValueError


def _device_to_string(device):
    if device is None:
        return 'cpu:0'
    return 'cuda:%d' % device.get_id()


def _numpy_to_tensor(ndarray, device):
    # return if ndarray is already torch.Tensor
    if isinstance(ndarray, torch.Tensor):
        return ndarray

    if ndarray.dtype == np.float32:
        dtype = torch.float32
    elif ndarray.dtype == np.float64:
        dtype = torch.float32
    elif ndarray.dtype == np.int32:
        dtype = torch.int32
    elif ndarray.dtype == np.uint8:
        dtype = torch.uint8
    device = _device_to_string(device)

    return torch.tensor(ndarray, dtype=dtype, device=device)


def _to_episodes(observation_shape, action_size, observations, actions,
                 rewards, terminals):
    rets = []
    head_index = 0
    for i in range(_safe_size(observations)):
        if terminals[i]:
            episode = Episode(observation_shape=observation_shape,
                              action_size=action_size,
                              observations=observations[head_index:i + 1],
                              actions=actions[head_index:i + 1],
                              rewards=rewards[head_index:i + 1])
            rets.append(episode)
            head_index = i + 1
    return rets


def _to_transitions(observation_shape, action_size, observations, actions,
                    rewards):
    rets = []
    num_data = _safe_size(observations)
    prev_transition = None
    for i in range(num_data - 1):
        observation = observations[i]
        action = actions[i]
        reward = rewards[i]
        next_observation = observations[i + 1]
        next_action = actions[i + 1]
        next_reward = rewards[i + 1]
        terminal = 1.0 if i == num_data - 2 else 0.0

        transition = Transition(observation_shape=observation_shape,
                                action_size=action_size,
                                observation=observation,
                                action=action,
                                reward=reward,
                                next_observation=next_observation,
                                next_action=next_action,
                                next_reward=next_reward,
                                terminal=terminal,
                                prev_transition=prev_transition)

        # set pointer to the next transition
        if prev_transition:
            prev_transition.next_transition = transition

        prev_transition = transition

        rets.append(transition)
    return rets


def _stack_frames(transition, n_frames):
    assert len(transition.observation.shape) == 3
    assert n_frames > 1
    assert isinstance(transition.observation, (np.ndarray, torch.Tensor))

    dtype = transition.observation.dtype
    n_channels = transition.observation.shape[0]
    image_size = transition.observation.shape[1:]
    shape = (n_frames * n_channels, *image_size)

    # create empty ndarray
    if isinstance(transition.observation, np.ndarray):
        observation = np.zeros(shape, dtype=dtype)
        next_observation = np.zeros(shape, dtype=dtype)
    else:
        device = transition.observation.device
        observation = torch.zeros(shape, dtype=dtype, device=device)
        next_observation = torch.zeros(shape, dtype=dtype, device=device)

    # stack frames
    t = transition
    for i in range(n_frames):
        tail_index = n_frames * n_channels - i * n_channels
        head_index = tail_index - n_channels
        observation[head_index:tail_index] = t.observation
        next_observation[head_index:tail_index] = t.next_observation
        if t.prev_transition is None:
            if i != n_frames - 1:
                tail_index -= n_channels
                head_index -= n_channels
                next_observation[head_index:tail_index] = t.observation
            break
        t = t.prev_transition

    return observation, next_observation


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
        as_tensor (bool): flag to hold observations as ``torch.Tensor``.
        device (d3rlpy.gpu.Device or int): gpu device or device id for tensors.

    """
    def __init__(self,
                 observations,
                 actions,
                 rewards,
                 terminals,
                 discrete_action=False,
                 as_tensor=False,
                 device=None):
        # data type option
        self._as_tensor = as_tensor
        if isinstance(device, int):
            self._device = Device(device)
        else:
            self._device = device

        # numpy to PyTorch conversion
        if as_tensor:
            self._observations = _numpy_to_tensor(observations, self._device)
        else:
            self._observations = observations

        self._rewards = np.asarray(rewards).reshape(-1)
        self._terminals = np.asarray(terminals).reshape(-1)
        self.discrete_action = discrete_action
        if discrete_action:
            self._actions = np.asarray(actions).reshape(-1)
        else:
            self._actions = np.asarray(actions)

        self._episodes = None

    @property
    def observations(self):
        """ Returns the observations.

        Returns:
            numpy.ndarray, list(numpy.ndarray) or torch.Tensor:
                array of observations.

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
    def as_tensor(self):
        """ Returns the flag to hold observations as ``torch.Tensor``.

        Returns:
            bool: flag to hold observations as ``torch.Tensor``.

        """
        return self._as_tensor

    @property
    def device(self):
        """ Returns the gpu device for tensors.

        Returns:
            d3rlpy.gpu.Device: gpu device.

        """
        return self._device

    @property
    def episodes(self):
        """ Returns the episodes.

        Returns:
            list(d3rlpy.dataset.Episode):
                list of :class:`d3rlpy.dataset.Episode` objects.

        """
        if self._episodes is None:
            self.build_episodes()
        return self._episodes

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
            return int(np.max(self._actions) + 1)
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

            # return statistics
            stats['return']['mean']
            stats['return']['std']
            stats['return']['min']
            stats['return']['max']

            # reward statistics
            stats['reward']['mean']
            stats['reward']['std']
            stats['reward']['min']
            stats['reward']['max']

            # action (only with continuous control actions)
            stats['action']['mean']
            stats['action']['std']
            stats['action']['min']
            stats['action']['max']

            # observation (only with numpy.ndarray observations)
            stats['observation']['mean']
            stats['observation']['std']
            stats['observation']['min']
            stats['observation']['max']

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
                'histogram': np.histogram(episode_returns, bins=20)
            },
            'reward': {
                'mean': np.mean(self._rewards),
                'std': np.std(self._rewards),
                'min': np.min(self._rewards),
                'max': np.max(self._rewards),
                'histogram': np.histogram(self._rewards, bins=20)
            }
        }

        # only for continuous control task
        if not self.discrete_action:
            # calculate histogram on each dimension
            hists = []
            for i in range(self.get_action_size()):
                hists.append(np.histogram(self.actions[:, i], bins=20))
            stats['action'] = {
                'mean': np.mean(self.actions, axis=0),
                'std': np.std(self.actions, axis=0),
                'min': np.min(self.actions, axis=0),
                'max': np.max(self.actions, axis=0),
                'histogram': hists
            }
        else:
            # count frequency of discrete actions
            freqs = []
            for i in range(self.get_action_size()):
                freqs.append((self.actions == i).sum())
            stats['action'] = {
                'histogram': [freqs, np.arange(self.get_action_size())]
            }

        # avoid large copy when observations are huge data.
        if isinstance(self._observations, np.ndarray):
            stats['observation'] = {
                'mean': np.mean(self.observations, axis=0),
                'std': np.std(self.observations, axis=0),
                'min': np.min(self.observations, axis=0),
                'max': np.max(self.observations, axis=0),
            }

        return stats

    def clip_reward(self, low=None, high=None):
        """ Clips rewards in the given range.

        Args:
            low (float): minimum value. If None, clipping is not performed on
                lower edge.
            high (float): maximum value. If None, clipping is not performed on
                upper edge.

        """
        self._rewards = np.clip(self._rewards, low, high)
        # rebuild Episode objects
        if self._episodes:
            self.build_episodes()

    def append(self, observations, actions, rewards, terminals):
        """ Appends new data.

        Args:
            observations (numpy.ndarray or list(numpy.ndarray)): N-D array.
            actions (numpy.ndarray): actions.
            rewards (numpy.ndarray): rewards.
            terminals (numpy.ndarray): terminals.

        """
        # validation
        for observation, action in zip(observations, actions):
            assert observation.shape == self.get_observation_shape()
            if self.discrete_action:
                assert int(action) < self.get_action_size()
            else:
                assert action.shape == (self.get_action_size(), )

        # append observations
        if isinstance(self._observations, list):
            self._observations += list(map(lambda x: x, observations))
        elif isinstance(self._observations, torch.Tensor):
            observations = _numpy_to_tensor(observations, self._device)
            self._observations = torch.cat([self._observations, observations],
                                           dim=0)
        else:
            self._observations = np.vstack([self._observations, observations])

        # append actions
        if self.discrete_action:
            self._actions = np.hstack([self._actions, actions])
        else:
            self._actions = np.vstack([self._actions, actions])

        # append rests
        self._rewards = np.hstack([self._rewards, rewards])
        self._terminals = np.hstack([self._terminals, terminals])

        # convert new data to list of episodes
        episodes = _to_episodes(observation_shape=self.get_observation_shape(),
                                action_size=self.get_action_size(),
                                observations=observations,
                                actions=actions,
                                rewards=rewards,
                                terminals=terminals)

        # append to episodes
        self._episodes += episodes

    def extend(self, dataset):
        """ Extend dataset by another dataset.

        Args:
            dataset (d3rlpy.dataset.MDPDataset): dataset.

        """
        assert self.is_action_discrete() == dataset.is_action_discrete()
        assert self.get_observation_shape() == dataset.get_observation_shape()
        assert self.get_action_size() == dataset.get_action_size()
        self.append(dataset.observations, dataset.actions, dataset.rewards,
                    dataset.terminals)

    def dump(self, fname):
        """ Saves dataset as HDF5.

        Args:
            fname (str): file path.

        """
        # make sure if data is numpy.ndarray or list
        if isinstance(self._observations, torch.Tensor):
            observations = self._observations.cpu().numpy()
        else:
            observations = self._observations

        with h5py.File(fname, 'w') as f:
            f.create_dataset('observations', data=observations)
            f.create_dataset('actions', data=self._actions)
            f.create_dataset('rewards', data=self._rewards)
            f.create_dataset('terminals', data=self._terminals)
            f.create_dataset('discrete_action', data=self.discrete_action)
            f.flush()

    @classmethod
    def load(cls, fname, as_tensor=False, device=None):
        """ Loads dataset from HDF5.

        .. code-block:: python

            import numpy as np
            from d3rlpy.dataset import MDPDataset

            dataset = MDPDataset(np.random.random(10, 4),
                                 np.random.random(10, 2),
                                 np.random.random(10),
                                 np.random.randint(2, size=10))

            # save as HDF5
            dataset.dump('dataset.h5')

            # load from HDF5
            new_dataset = MDPDataset.load('dataset.h5')

        Args:
            fname (str): file path.
            as_tensor (bool): flag to hold observations as ``torch.Tensor``.
            device (d3rlpy.gpu.Device or int):
                gpu device or device id for tensor.

        """
        with h5py.File(fname, 'r') as f:
            observations = f['observations'][()]
            actions = f['actions'][()]
            rewards = f['rewards'][()]
            terminals = f['terminals'][()]
            discrete_action = f['discrete_action'][()]

        dataset = cls(observations=observations,
                      actions=actions,
                      rewards=rewards,
                      terminals=terminals,
                      discrete_action=discrete_action,
                      as_tensor=as_tensor,
                      device=device)

        return dataset

    def build_episodes(self):
        """ Builds episode objects.

        This method will be internally called when accessing the episodes
        property at the first time.

        """
        self._episodes = _to_episodes(
            observation_shape=self.get_observation_shape(),
            action_size=self.get_action_size(),
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            terminals=self._terminals)

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
        observations (numpy.ndarray, list(numpy.ndarray) or torch.Tensor):
            observations.
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
            numpy.ndarray, list(numpy.ndarray) or torch.Tensor:
                array of observations.

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
            self.build_transitions()
        return self._transitions

    def build_transitions(self):
        """ Builds transition objects.

        This method will be internally called when accessing the transitions
        property at the first time.

        """
        self._transitions = _to_transitions(
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards)

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

            R = \\sum_{i=1} r_i

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
        observation (numpy.ndarray or torch.Tensor): observation at `t`.
        action (numpy.ndarray or int): action at `t`.
        reward (float): reward at `t`.
        next_observation (numpy.ndarray or torch.Tensor): observation at `t+1`.
        next_action (numpy.ndarray or int): action at `t+1`.
        next_reward (float): reward at `t+1`.
        terminal (int): terminal flag at `t+1`.
        prev_transition (d3rlpy.dataset.Transition):
            pointer to the previous transition.
        next_transition (d3rlpy.dataset.Transition):
            pointer to the next transition.

    """
    def __init__(self,
                 observation_shape,
                 action_size,
                 observation,
                 action,
                 reward,
                 next_observation,
                 next_action,
                 next_reward,
                 terminal,
                 prev_transition=None,
                 next_transition=None):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self._observation = observation
        self._action = action
        self._reward = reward
        self._next_observation = next_observation
        self._next_action = next_action
        self._next_reward = next_reward
        self._terminal = terminal
        self._prev_transition = prev_transition
        self._next_transition = next_transition

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
            numpy.ndarray or torch.Tensor: observation at `t`.

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
            numpy.ndarray or torch.Tensor: observation at `t+1`.

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

    @property
    def prev_transition(self):
        """ Returns pointer to the previous transition.

        If this is the first transition, this method should return ``None``.

        Returns:
            d3rlpy.dataset.Transition: previous transition.

        """
        return self._prev_transition

    @prev_transition.setter
    def prev_transition(self, transition):
        """ Sets transition to ``prev_transition``.

        Args:
            d3rlpy.dataset.Transition: previous transition.

        """
        assert isinstance(transition, Transition)
        self._prev_transition = transition

    @property
    def next_transition(self):
        """ Returns pointer to the next transition.

        If this is the last transition, this method should return ``None``.

        Returns:
            d3rlpy.dataset.Transition: next transition.

        """
        return self._next_transition

    @next_transition.setter
    def next_transition(self, transition):
        """ Sets transition to ``next_transition``.

        Args:
            d3rlpy.dataset.Dataset: next transition.

        """
        assert isinstance(transition, Transition)
        self._next_transition = transition


class TransitionMiniBatch:
    """ mini-batch of Transition objects.

    This class is designed to hold :class:`d3rlpy.dataset.Transition` objects
    for being passed to algorithms during fitting.

    Args:
        transitions (list(d3rlpy.dataset.Transition)):
            mini-batch of transitions.

    """
    def __init__(self, transitions, n_frames=1):
        self._transitions = transitions

        observations = []
        actions = []
        rewards = []
        next_observations = []
        next_actions = []
        next_rewards = []
        terminals = []
        for transition in transitions:
            # stack frames if necessary
            if n_frames > 1 and len(transition.observation.shape) == 3:
                stacked_data = _stack_frames(transition, n_frames)
                observation, next_observation = stacked_data
            else:
                observation = transition.observation
                next_observation = transition.next_observation

            observations.append(observation)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_observations.append(next_observation)
            next_actions.append(transition.next_action)
            next_rewards.append(transition.next_reward)
            terminals.append(transition.terminal)

        # convert list to ndarray or torch.Tensor and fix shapes
        if isinstance(observations[0], torch.Tensor):
            self._observations = torch.stack(observations, dim=0)
            self._next_observations = torch.stack(next_observations, dim=0)
        else:
            self._observations = np.array(observations)
            self._next_observations = np.array(next_observations)

        self._actions = np.array(actions).reshape((self.size(), -1))
        self._rewards = np.array(rewards).reshape((self.size(), 1))
        self._next_rewards = np.array(next_rewards).reshape((self.size(), 1))
        self._next_actions = np.array(next_actions).reshape((self.size(), -1))
        self._terminals = np.array(terminals).reshape((self.size(), 1))

    @property
    def observations(self):
        """ Returns mini-batch of observations at `t`.

        Returns:
            numpy.ndarray or torch.Tensor: observations at `t`.

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
            numpy.ndarray or torch.Tensor: observations at `t+1`.

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
