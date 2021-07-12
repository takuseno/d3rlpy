import copy
import warnings

import numpy as np

cimport numpy as np

import cython
import h5py

from cython cimport view

from cython.parallel import prange

from dataset cimport CTransition
from libc.string cimport memcpy
from libcpp cimport bool, nullptr
from libcpp.memory cimport make_shared, shared_ptr


def _safe_size(array):
    if isinstance(array, (list, tuple)):
        return len(array)
    elif isinstance(array, np.ndarray):
        return array.shape[0]
    raise ValueError


def _to_episodes(
    observation_shape,
    action_size,
    observations,
    actions,
    rewards,
    terminals,
    episode_terminals,
    create_mask,
    mask_size
):
    rets = []
    head_index = 0
    for i in range(_safe_size(observations)):
        if episode_terminals[i]:
            episode = Episode(
                observation_shape=observation_shape,
                action_size=action_size,
                observations=observations[head_index:i + 1],
                actions=actions[head_index:i + 1],
                rewards=rewards[head_index:i + 1],
                terminal=terminals[i],
                create_mask=create_mask,
                mask_size=mask_size
            )
            rets.append(episode)
            head_index = i + 1
    return rets


def _to_transitions(
    observation_shape,
    action_size,
    observations,
    actions,
    rewards,
    terminal,
    create_mask,
    mask_size
):
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
        env_terminal = terminal if i == num_data - 2 else 0.0

        if create_mask:
            mask = np.random.randint(2, size=mask_size)
        else:
            mask = None

        transition = Transition(
            observation_shape=observation_shape,
            action_size=action_size,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            next_action=next_action,
            next_reward=next_reward,
            terminal=env_terminal,
            mask=mask,
            prev_transition=prev_transition
        )

        # set pointer to the next transition
        if prev_transition:
            prev_transition.next_transition = transition

        prev_transition = transition

        rets.append(transition)
    return rets


def _check_discrete_action(actions):
    float_actions = np.array(actions, dtype=np.float32)
    int_actions = np.array(actions, dtype=np.int32)
    return np.all(float_actions == int_actions)


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
        observations (numpy.ndarray): N-D array. If the
            observation is a vector, the shape should be
            `(N, dim_observation)`. If the observations is an image, the shape
            should be `(N, C, H, W)`.
        actions (numpy.ndarray): N-D array. If the actions-space is
            continuous, the shape should be `(N, dim_action)`. If the
            action-space is discrete, the shape should be `(N,)`.
        rewards (numpy.ndarray): array of scalar rewards.
        terminals (numpy.ndarray): array of binary terminal flags.
        episode_terminals (numpy.ndarray): array of binary episode terminal
            flags. The given data will be splitted based on this flag.
            This is useful if you want to specify the non-environment
            terminations (e.g. timeout). If ``None``, the episode terminations
            match the environment terminations.
        discrete_action (bool): flag to use the given actions as discrete
            action-space actions. If ``None``, the action type is automatically
            determined.
        create_mask (bool): flag to create binary masks for bootstrapping.
        mask_size (int): ensemble size for mask. If ``create_mask`` is False,
            this will be ignored.

    """
    def __init__(
        self,
        observations,
        actions,
        rewards,
        terminals,
        episode_terminals=None,
        discrete_action=None,
        create_mask=False,
        mask_size=1
    ):
        # validation
        assert isinstance(observations, np.ndarray),\
            'Observations must be numpy array.'
        if len(observations.shape) == 4:
            assert observations.dtype == np.uint8,\
                'Image observation must be uint8 array.'
        else:
            if observations.dtype != np.float32:
                observations = np.asarray(observations, dtype=np.float32)

        self._observations = observations
        self._rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        self._terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)

        if episode_terminals is None:
            # if None, episode terminals match the environment terminals
            self._episode_terminals = self._terminals
        else:
            self._episode_terminals = np.asarray(
                episode_terminals, dtype=np.float32).reshape(-1)

        # automatic action type detection
        if discrete_action is None:
            discrete_action = _check_discrete_action(actions)

        self.discrete_action = discrete_action
        if discrete_action:
            self._actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        else:
            self._actions = np.asarray(actions, dtype=np.float32)

        self._episodes = None
        self._create_mask = create_mask
        self._mask_size = mask_size

    @property
    def observations(self):
        """ Returns the observations.

        Returns:
            numpy.ndarray: array of observations.

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
    def episode_terminals(self):
        """ Returns the episode terminal flags.

        Returns:
            numpy.ndarray: array of episode terminal flags.

        """
        return self._episode_terminals

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

    def append(
        self,
        observations,
        actions,
        rewards,
        terminals,
        episode_terminals=None
    ):
        """ Appends new data.

        Args:
            observations (numpy.ndarray): N-D array.
            actions (numpy.ndarray): actions.
            rewards (numpy.ndarray): rewards.
            terminals (numpy.ndarray): terminals.
            episode_terminals (numpy.ndarray): episode terminals.

        """
        # validation
        for observation, action in zip(observations, actions):
            assert observation.shape == self.get_observation_shape(),\
                f'Observation shape must be {self.get_observation_shape()}.'
            if self.discrete_action:
                if int(action) >= self.get_action_size():
                    message = f'New action size is higher than' \
                              f' {self.get_action_size()}.'
                    warnings.warn(message)
            else:
                assert action.shape == (self.get_action_size(), ),\
                    f'Action size must be {self.get_action_size()}.'

        # append observations
        self._observations = np.vstack([self._observations, observations])

        # append actions
        if self.discrete_action:
            self._actions = np.hstack([self._actions, actions])
        else:
            self._actions = np.vstack([self._actions, actions])

        # append rests
        self._rewards = np.hstack([self._rewards, rewards])
        self._terminals = np.hstack([self._terminals, terminals])
        if episode_terminals is None:
            episode_terminals = terminals
        self._episode_terminals = np.hstack(
            [self._episode_terminals, episode_terminals]
        )


        # convert new data to list of episodes
        episodes = _to_episodes(
            observation_shape=self.get_observation_shape(),
            action_size=self.get_action_size(),
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            terminals=self._terminals,
            episode_terminals=self._episode_terminals,
            create_mask=self._create_mask,
            mask_size=self._mask_size
        )

        self._episodes = episodes

    def extend(self, dataset):
        """ Extend dataset by another dataset.

        Args:
            dataset (d3rlpy.dataset.MDPDataset): dataset.

        """
        assert self.is_action_discrete() == dataset.is_action_discrete(),\
            'Dataset must have discrete action-space.'
        assert self.get_observation_shape() == dataset.get_observation_shape(),\
            f'Observation shape must be {self.get_observation_shape()}'

        self.append(
            dataset.observations,
            dataset.actions,
            dataset.rewards,
            dataset.terminals,
            dataset.episode_terminals
        )

    def dump(self, fname):
        """ Saves dataset as HDF5.

        Args:
            fname (str): file path.

        """
        with h5py.File(fname, 'w') as f:
            f.create_dataset('observations', data=self._observations)
            f.create_dataset('actions', data=self._actions)
            f.create_dataset('rewards', data=self._rewards)
            f.create_dataset('terminals', data=self._terminals)
            f.create_dataset('episode_terminals', data=self._episode_terminals)
            f.create_dataset('discrete_action', data=self.discrete_action)
            f.create_dataset('create_mask', data=self._create_mask)
            f.create_dataset('mask_size', data=self._mask_size)
            f.flush()

    @classmethod
    def load(cls, fname, create_mask=False, mask_size=1):
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
            create_mask (bool): flag to create bootstrapping masks.
            mask_size (int): size of bootstrapping masks.

        """
        with h5py.File(fname, 'r') as f:
            observations = f['observations'][()]
            actions = f['actions'][()]
            rewards = f['rewards'][()]
            terminals = f['terminals'][()]
            discrete_action = f['discrete_action'][()]

            # for backward compatibility
            if 'episode_terminals' in f:
                episode_terminals = f['episode_terminals'][()]
            else:
                episode_terminals = None

        dataset = cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            episode_terminals=episode_terminals,
            discrete_action=discrete_action,
            create_mask=create_mask,
            mask_size=mask_size
        )

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
            terminals=self._terminals,
            episode_terminals=self._episode_terminals,
            create_mask=self._create_mask,
            mask_size=self._mask_size
        )

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
        observations (numpy.ndarray): observations.
        actions (numpy.ndarray): actions.
        rewards (numpy.ndarray): scalar rewards.
        terminal (bool): binary terminal flag. If False, the episode is not
            terminated by the environment (e.g. timeout).
        create_mask (bool): flag to create binary masks for bootstrapping.
        mask_size (int): ensemble size for mask. If ``create_mask`` is False,
            this will be ignored.

    """
    def __init__(
        self,
        observation_shape,
        action_size,
        observations,
        actions,
        rewards,
        terminal=True,
        create_mask=False,
        mask_size=1
    ):
        # validation
        assert isinstance(observations, np.ndarray),\
            'Observation must be numpy array.'
        if len(observation_shape) == 3:
            assert observations.dtype == np.uint8,\
                'Image observation must be uint8 array.'
        else:
            if observations.dtype != np.float32:
                observations = np.asarray(observations, dtype=np.float32)

        # fix action dtype and shape
        if len(actions.shape) == 1:
            actions = np.asarray(actions, dtype=np.int32).reshape(-1)
        else:
            actions = np.asarray(actions, dtype=np.float32)

        self.observation_shape = observation_shape
        self.action_size = action_size
        self._observations = observations
        self._actions = actions
        self._rewards = np.asarray(rewards, dtype=np.float32)
        self._terminal = terminal
        self._create_mask = create_mask
        self._mask_size = mask_size
        self._transitions = None

    @property
    def observations(self):
        """ Returns the observations.

        Returns:
            numpy.ndarray: array of observations.

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
    def terminal(self):
        """ Returns the terminal flag.

        Returns:
            bool: the terminal flag.

        """
        return self._terminal

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
            rewards=self._rewards,
            terminal=self._terminal,
            create_mask=self._create_mask,
            mask_size=self._mask_size
        )

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


ctypedef np.uint8_t UINT8_t
ctypedef np.float32_t FLOAT_t
ctypedef np.int32_t INT_t
ctypedef shared_ptr[CTransition] TransitionPtr


cdef class Transition:
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
        mask (numpy.ndarray): binary mask for bootstrapping.
        prev_transition (d3rlpy.dataset.Transition):
            pointer to the previous transition.
        next_transition (d3rlpy.dataset.Transition):
            pointer to the next transition.

    """
    cdef TransitionPtr _thisptr
    cdef bool _is_image
    cdef bool _is_discrete
    cdef _observation
    cdef _action
    cdef _next_observation
    cdef _next_action
    cdef _mask
    cdef Transition _prev_transition
    cdef Transition _next_transition

    def __cinit__(
        self,
        vector[int] observation_shape,
        int action_size,
        np.ndarray observation,
        action not None,
        float reward,
        np.ndarray next_observation,
        next_action not None,
        float next_reward,
        float terminal,
        np.ndarray mask=None,
        Transition prev_transition=None,
        Transition next_transition=None
    ):
        cdef TransitionPtr prev_ptr
        cdef TransitionPtr next_ptr

        # validation
        if observation_shape.size() == 3:
            assert observation.dtype == np.uint8,\
                'Image observation must be uint8 array.'
            assert next_observation.dtype == np.uint8,\
                'Image observation must be uint8 array.'
        else:
            if observation.dtype != np.float32:
                observation = np.asarray(observation, dtype=np.float32)
            if next_observation.dtype != np.float32:
                next_observation = np.asarray(
                    next_observation, dtype=np.float32
                )

        if prev_transition:
            prev_ptr = prev_transition.get_ptr()
        if next_transition:
            next_ptr = next_transition.get_ptr()

        self._thisptr = make_shared[CTransition]()
        self._thisptr.get().observation_shape = observation_shape
        self._thisptr.get().action_size = action_size
        self._thisptr.get().reward = reward
        self._thisptr.get().next_reward = next_reward
        self._thisptr.get().terminal = terminal
        self._thisptr.get().prev_transition = prev_ptr
        self._thisptr.get().next_transition = next_ptr
        if mask is not None:
            self._thisptr.get().mask = np.array(mask, dtype=np.float32).tolist()

        # assign observation
        if observation_shape.size() == 3:
            self._thisptr.get().observation_i = <UINT8_t*> observation.data
            self._thisptr.get().next_observation_i = <UINT8_t*> next_observation.data
            self._is_image = True
        else:
            self._thisptr.get().observation_f = <FLOAT_t*> observation.data
            self._thisptr.get().next_observation_f = <FLOAT_t*> next_observation.data
            self._is_image = False

        # assign action
        cdef np.ndarray[FLOAT_t, ndim=1] action_f
        cdef np.ndarray[FLOAT_t, ndim=1] next_action_f
        cdef int action_i
        cdef int next_action_i
        if isinstance(action, np.ndarray):
            action_f = np.asarray(action, dtype=np.float32)
            next_action_f = np.asarray(next_action, dtype=np.float32)
            self._thisptr.get().action_f = <FLOAT_t*> action_f.data
            self._thisptr.get().next_action_f = <FLOAT_t*> next_action_f.data
            self._is_discrete = False
        else:
            action_i = action
            next_action_i = next_action
            self._thisptr.get().action_i = action_i
            self._thisptr.get().next_action_i = next_action_i
            self._is_discrete = True

        self._observation = observation
        self._action = action
        self._next_observation = next_observation
        self._next_action = next_action
        self._prev_transition = prev_transition
        self._next_transition = next_transition

    cdef TransitionPtr get_ptr(self):
        return self._thisptr

    def get_observation_shape(self):
        """ Returns observation shape.

        Returns:
            tuple: observation shape.

        """
        return tuple(self._thisptr.get().observation_shape)

    def get_action_size(self):
        """ Returns dimension of action-space.

        Returns:
            int: dimension of action-space.

        """
        return self._thisptr.get().action_size

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
        return self._thisptr.get().reward

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
        return self._thisptr.get().next_reward

    @property
    def terminal(self):
        """ Returns terminal flag at `t+1`.

        Returns:
            int: terminal flag at `t+1`.

        """
        return self._thisptr.get().terminal

    @property
    def mask(self):
        """ Returns binary mask for bootstrapping.

        Returns:
            np.ndarray: array of binary mask.

        """
        mask = self._thisptr.get().mask
        return np.asarray(mask, dtype=np.float32) if mask.size() > 0 else None

    @mask.setter
    def mask(self, mask):
        """ Sets binary mask for bootstrapping.

        Args:
            mask (np.ndarray): array of binary mask.

        """
        self._mask = np.ndarray(mask, dtype=np.float32)

    @property
    def prev_transition(self):
        """ Returns pointer to the previous transition.

        If this is the first transition, this method should return ``None``.

        Returns:
            d3rlpy.dataset.Transition: previous transition.

        """
        return self._prev_transition

    @prev_transition.setter
    def prev_transition(self, Transition transition):
        """ Sets transition to ``prev_transition``.

        Args:
            d3rlpy.dataset.Transition: previous transition.

        """
        assert isinstance(transition, Transition)
        cdef TransitionPtr ptr
        ptr = transition.get_ptr()
        self._thisptr.get().prev_transition = ptr
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
    def next_transition(self, Transition transition):
        """ Sets transition to ``next_transition``.

        Args:
            d3rlpy.dataset.Dataset: next transition.

        """
        assert isinstance(transition, Transition)
        cdef TransitionPtr ptr
        ptr = transition.get_ptr()
        self._thisptr.get().next_transition = ptr
        self._next_transition = transition

    def clear_links(self):
        """ Clears links to the next and previous transitions.

        This method is necessary to call when freeing this instance by GC.

        """
        self._prev_transition = None
        self._next_transition = None
        self._thisptr.get().prev_transition = <TransitionPtr> nullptr
        self._thisptr.get().next_transition = <TransitionPtr> nullptr


def trace_back_and_clear(transition):
    """ Traces transitions and clear all links.

    Args:
        transition (d3rlpy.dataset.Transition): transition.

    """
    while True:
        if transition is None:
            break
        prev_transition = transition.prev_transition
        transition.clear_links()
        transition = prev_transition


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _stack_frames(
    TransitionPtr transition,
    UINT8_t* stack,
    int n_frames,
    bool stack_next=False
) nogil:
    cdef UINT8_t* observation_ptr
    cdef int c = transition.get().observation_shape[0]
    cdef int h = transition.get().observation_shape[1]
    cdef int w = transition.get().observation_shape[2]
    cdef int image_size = c * h * w

    # stack frames
    cdef TransitionPtr t = transition
    cdef int i, j
    cdef int head_channel
    cdef int tail_channel
    cdef int offset
    cdef int index
    for i in range(n_frames):

        tail_channel = n_frames * c - i * c
        head_channel = tail_channel - c

        if stack_next:
            observation_ptr = t.get().next_observation_i
        else:
            observation_ptr = t.get().observation_i

        memcpy(stack + head_channel * h * w, observation_ptr, image_size)

        if t.get().prev_transition == nullptr:
            # fill rests with the last frame
            for j in range(n_frames - i - 1):
                tail_channel = n_frames * c - (i + j + 1) * c
                head_channel = tail_channel - c
                memcpy(stack + head_channel * h * w, t.get().observation_i, image_size)
            break
        t = t.get().prev_transition


cdef class TransitionMiniBatch:
    """ mini-batch of Transition objects.

    This class is designed to hold :class:`d3rlpy.dataset.Transition` objects
    for being passed to algorithms during fitting.

    If the observation is image, you can stack arbitrary frames via
    ``n_frames``.

    .. code-block:: python

        transition.observation.shape == (3, 84, 84)

        batch_size = len(transitions)

        # stack 4 frames
        batch = TransitionMiniBatch(transitions, n_frames=4)

        # 4 frames x 3 channels
        batch.observations.shape == (batch_size, 12, 84, 84)

    This is implemented by tracing previous transitions through
    ``prev_transition`` property.

    Args:
        transitions (list(d3rlpy.dataset.Transition)):
            mini-batch of transitions.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): length of N-step sampling.
        gamma (float): discount factor for N-step calculation.

    """
    cdef list _transitions
    cdef dict _additional_data
    cdef np.ndarray _observations
    cdef np.ndarray _actions
    cdef np.ndarray _rewards
    cdef np.ndarray _next_observations
    cdef np.ndarray _next_actions
    cdef np.ndarray _next_rewards
    cdef np.ndarray _terminals
    cdef np.ndarray _masks
    cdef np.ndarray _n_steps

    def __cinit__(
        self,
        list transitions not None,
        int n_frames=1,
        int n_steps=1,
        float gamma=0.99
    ):
        self._transitions = transitions

        # determine observation shape
        cdef tuple observation_shape = transitions[0].get_observation_shape()
        cdef int observation_ndim = len(observation_shape)
        observation_dtype = transitions[0].observation.dtype
        if len(observation_shape) == 3 and n_frames > 1:
            c, h, w = observation_shape
            observation_shape = (n_frames * c, h, w)

        # determine action shape
        cdef int action_size = transitions[0].get_action_size()
        cdef tuple action_shape = tuple()
        action_dtype = np.int32
        if isinstance(transitions[0].action, np.ndarray):
            action_shape = (action_size,)
            action_dtype = np.float32

        # allocate batch data
        cdef int size = len(transitions)
        self._observations = np.empty(
            (size,) + observation_shape, dtype=observation_dtype
        )
        self._actions = np.empty((size,) + action_shape, dtype=action_dtype)
        self._rewards = np.empty((size, 1), dtype=np.float32)
        self._next_observations = np.empty(
            (size,) + observation_shape, dtype=observation_dtype
        )
        self._next_actions = np.empty((size,) + action_shape, dtype=action_dtype)
        self._next_rewards = np.empty((size, 1), dtype=np.float32)
        self._terminals = np.empty((size, 1), dtype=np.float32)
        self._n_steps = np.empty((size, 1), dtype=np.float32)

        # determine flags
        cdef bool is_image
        cdef bool is_dicsrete
        is_image = len(transitions[0].get_observation_shape()) == 3
        is_discrete = not isinstance(transitions[0].action, np.ndarray)

        # prepare pointers to batch data
        cdef void* observations_ptr = self._observations.data
        cdef void* actions_ptr = self._actions.data
        cdef FLOAT_t* rewards_ptr = <FLOAT_t*> self._rewards.data
        cdef void* next_observations_ptr = self._next_observations.data
        cdef void* next_actions_ptr = self._next_actions.data
        cdef FLOAT_t* next_rewards_ptr = <FLOAT_t*> self._next_rewards.data
        cdef FLOAT_t* terminals_ptr = <FLOAT_t*> self._terminals.data
        cdef FLOAT_t* n_steps_ptr = <FLOAT_t*> self._n_steps.data

        # get pointers to transitions
        cdef int i
        cdef Transition transition
        cdef vector[TransitionPtr] transition_ptrs
        for i in range(size):
            transition = transitions[i]
            transition_ptrs.push_back(transition.get_ptr())

        # efficient memory copy
        cdef TransitionPtr ptr
        cdef vector[vector[float]] masks
        for i in prange(size, nogil=True):
            ptr = transition_ptrs[i]
            self._assign_to_batch(
                batch_index=i,
                ptr=ptr,
                observations_ptr=observations_ptr,
                actions_ptr=actions_ptr,
                rewards_ptr=rewards_ptr,
                next_observations_ptr=next_observations_ptr,
                next_actions_ptr=next_actions_ptr,
                next_rewards_ptr=next_rewards_ptr,
                terminals_ptr=terminals_ptr,
                n_steps_ptr=n_steps_ptr,
                n_frames=n_frames,
                n_steps=n_steps,
                gamma=gamma,
                is_image=is_image,
                is_discrete=is_discrete
            )
            # append valid mask
            if ptr.get().mask.size() > 0:
                masks.push_back(ptr.get().mask)

        # create binary mask only when every transition has masks
        if masks.size() == size:
            masks = np.array(masks, dtype=np.float32)
            self._masks = np.transpose(np.expand_dims(masks, axis=2), [1, 0, 2])
        else:
            self._masks = None

        # additional data
        self._additional_data = {}

    cdef void _assign_observation(
        self,
        int batch_index,
        TransitionPtr ptr,
        void* observations_ptr,
        int n_frames,
        bool is_image,
        bool is_next
    ) nogil:
        cdef int offset, channel, height, width
        cdef void* src_observation_ptr
        if is_image:
            channel = ptr.get().observation_shape[0]
            height = ptr.get().observation_shape[1]
            width = ptr.get().observation_shape[2]
            # stack frames if necessary
            if n_frames > 1:
                offset = n_frames * batch_index * channel * height * width
                _stack_frames(
                    transition=ptr,
                    stack=(<UINT8_t*> observations_ptr) + offset,
                    n_frames=n_frames,
                    stack_next=is_next
                )
            else:
                offset = batch_index * channel * height * width
                if is_next:
                    src_observation_ptr = ptr.get().next_observation_i
                else:
                    src_observation_ptr = ptr.get().observation_i
                memcpy(
                    (<UINT8_t*> observations_ptr) + offset,
                    <UINT8_t*> src_observation_ptr,
                    channel * height * width
                )
        else:
            offset = batch_index * ptr.get().observation_shape[0]
            if is_next:
                src_observation_ptr = ptr.get().next_observation_f
            else:
                src_observation_ptr = ptr.get().observation_f
            memcpy(
                (<FLOAT_t*> observations_ptr) + offset,
                <FLOAT_t*> src_observation_ptr,
                ptr.get().observation_shape[0] * sizeof(FLOAT_t)
            )

    cdef void _assign_action(
        self,
        int batch_index,
        TransitionPtr ptr,
        void* actions_ptr,
        bool is_discrete,
        bool is_next
    ) nogil:
        cdef int offset
        cdef void* src_action_ptr
        if is_discrete:
            if is_next:
                ((<INT_t*> actions_ptr) + batch_index)[0] = ptr.get().next_action_i
            else:
                ((<INT_t*> actions_ptr) + batch_index)[0] = ptr.get().action_i
        else:
            offset = batch_index * ptr.get().action_size
            if is_next:
                src_action_ptr = ptr.get().next_action_f
            else:
                src_action_ptr = ptr.get().action_f
            memcpy(
                (<FLOAT_t*> actions_ptr) + offset,
                <FLOAT_t*> src_action_ptr,
                ptr.get().action_size * sizeof(FLOAT_t)
            )

    cdef void _assign_to_batch(
        self,
        int batch_index,
        TransitionPtr ptr,
        void* observations_ptr,
        void* actions_ptr,
        float* rewards_ptr,
        void* next_observations_ptr,
        void* next_actions_ptr,
        float* next_rewards_ptr,
        float* terminals_ptr,
        float* n_steps_ptr,
        int n_frames,
        int n_steps,
        float gamma,
        bool is_image,
        bool is_discrete
    ) nogil:
        cdef int i
        cdef float n_step_return = 0.0
        cdef TransitionPtr next_ptr

        # assign data at t
        self._assign_observation(
            batch_index=batch_index,
            ptr=ptr,
            observations_ptr=observations_ptr,
            n_frames=n_frames,
            is_image=is_image,
            is_next=False
        )
        self._assign_action(
            batch_index=batch_index,
            ptr=ptr,
            actions_ptr=actions_ptr,
            is_discrete=is_discrete,
            is_next=False
        )
        rewards_ptr[batch_index] = ptr.get().reward

        # compute N-step return
        next_ptr = ptr
        for i in range(n_steps):
            n_step_return += next_ptr.get().next_reward * gamma ** i
            if next_ptr.get().next_transition == nullptr or i == n_steps - 1:
                break
            next_ptr = next_ptr.get().next_transition

        # assign data at t+N
        self._assign_observation(
            batch_index=batch_index,
            ptr=next_ptr,
            observations_ptr=next_observations_ptr,
            n_frames=n_frames,
            is_image=is_image,
            is_next=True
        )
        self._assign_action(
            batch_index=batch_index,
            ptr=next_ptr,
            actions_ptr=next_actions_ptr,
            is_discrete=is_discrete,
            is_next=True
        )
        next_rewards_ptr[batch_index] = n_step_return
        terminals_ptr[batch_index] = next_ptr.get().terminal
        n_steps_ptr[batch_index] = i + 1

    def add_additional_data(self, key, value):
        """Add arbitrary additional data.

        Args:
            key (str): key of data.
            value (any): value.

        """
        self._additional_data[key] = value

    def get_additional_data(self, key):
        """Returns specified additional data.

        Args:
            key (str): key of data.

        Returns:
            any: value.

        """
        assert key in self._additional_data, '%s does not exist.' % key
        return self._additional_data[key]

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
        """ Returns mini-batch of observations at `t+n`.

        Returns:
            numpy.ndarray or torch.Tensor: observations at `t+n`.

        """
        return self._next_observations

    @property
    def next_actions(self):
        """ Returns mini-batch of actions at `t+n`.

        Returns:
            numpy.ndarray: actions at `t+n`.

        """
        return self._next_actions

    @property
    def next_rewards(self):
        """ Returns mini-batch of rewards at `t+n`.

        Returns:
            numpy.ndarray: rewards at `t+n`.

        """
        return self._next_rewards

    @property
    def terminals(self):
        """ Returns mini-batch of terminal flags at `t+n`.

        Returns:
            numpy.ndarray: terminal flags at `t+n`.

        """
        return self._terminals

    @property
    def masks(self):
        """ Returns mini-batch of binary masks for bootstrapping.

        If any of transitions have an invalid mask, this will return ``None``.

        Returns:
            numpy.ndarray: binary mask.

        """
        return self._masks

    @property
    def n_steps(self):
        """ Returns mini-batch of the number of steps before next observations.

        This will always include only ones if ``n_steps=1``. If ``n_steps`` is
        bigger than ``1``. the values will depend on its episode length.

        Returns:
            numpy.ndarray: the number of steps before next observations.

        """
        return self._n_steps

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


cdef _compute_returns(Transition transition, float gamma, int n_frames):
    cdef vector[TransitionPtr] transitions
    cdef TransitionPtr ptr
    cdef int i, channel, width, height, offset
    cdef bool is_image
    cdef float R
    cdef void* observations_ptr
    cdef FLOAT_t* returns_ptr
    cdef FLOAT_t* terminals_ptr
    cdef np.ndarray observations, returns, terminals

    # iterate through transitions
    ptr = transition.get_ptr()
    with nogil:
        while True:
            transitions.push_back(ptr)
            ptr = ptr.get().next_transition
            if ptr == nullptr:
                break

    # prepare observations and returns
    is_image = len(transition.get_observation_shape()) == 3
    if is_image and n_frames > 1:
        channel, width, height = transition.get_observation_shape()
        shape = (transitions.size(), channel * n_frames, width, height)
    else:
        shape = (transitions.size(), *transition.get_observation_shape())
    dtype = np.uint8 if is_image else np.float32
    observations = np.zeros(shape, dtype=dtype)
    returns = np.empty(transitions.size(), dtype=np.float32)
    terminals = np.empty(transitions.size(), dtype=np.float32)

    observations_ptr = observations.data
    returns_ptr = <FLOAT_t*> returns.data
    terminals_ptr = <FLOAT_t*> terminals.data
    R = 0.0
    with nogil:
        for i in range(transitions.size()):
            ptr = transitions[i]

            # compute discounted return
            R += (gamma**i) * ptr.get().next_reward
            returns_ptr[i] = R
            terminals_ptr[i] = ptr.get().terminal

            # append observation
            if is_image:
                channel = ptr.get().observation_shape[0]
                height = ptr.get().observation_shape[1]
                width = ptr.get().observation_shape[2]
                # stack frames if necessary
                if n_frames > 1:
                    offset = n_frames * i * channel * height * width
                    _stack_frames(
                        transition=ptr,
                        stack=(<UINT8_t*> observations_ptr) + offset,
                        n_frames=n_frames,
                        stack_next=True
                    )
                else:
                    offset = i * channel * height * width
                    memcpy(
                        (<UINT8_t*> observations_ptr) + offset,
                        ptr.get().next_observation_i,
                        channel * height * width
                    )
            else:
                offset = i * ptr.get().observation_shape[0]
                memcpy(
                    (<FLOAT_t*> observations_ptr) + offset,
                    ptr.get().next_observation_f,
                    ptr.get().observation_shape[0] * sizeof(FLOAT_t)
                )

    return observations, returns, terminals


def compute_lambda_return(transition, algo, gamma, lam, n_frames):
    observations, returns, terminals = _compute_returns(
        transition, gamma, n_frames
    )

    values = algo.predict_value(observations)
    gammas = gamma ** (np.arange(returns.shape[0]) + 1)
    returns += gammas * values * (1.0 - terminals)

    lambdas = lam**np.arange(returns.shape[0])
    lambda_return = (1.0 - lam) * np.sum(lambdas[:-1] * returns[:-1])
    lambda_return += lambdas[-1] * returns[-1]

    return lambda_return
