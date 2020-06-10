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


class MDPDataset:
    def __init__(self,
                 observations,
                 actions,
                 rewards,
                 terminals,
                 discrete_action=False):
        self._observations = np.array(observations)
        self._actions = np.array(actions)
        self._rewards = np.array(rewards)
        self._terminals = np.array(terminals)
        self.discrete_action = discrete_action

        # array of Episode
        self._episodes = self._to_episodes()

    @property
    def observations(self):
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards

    @property
    def terminals(self):
        return self._terminals

    @property
    def episodes(self):
        return self._episodes

    def _to_episodes(self):
        rets = []
        observations = []
        actions = []
        rewards = []
        for i in range(self.observations.shape[0]):
            observations.append(self._observations[i])
            actions.append(self._actions[i])
            rewards.append(self._rewards[i])
            if self._terminals[i]:
                episode = Episode(self.get_observation_shape(),
                                  self.get_action_size(),
                                  np.array(observations),
                                  np.array(actions), np.array(rewards))
                rets.append(episode)
                observations = []
                actions = []
                rewards = []
        return rets

    def size(self):
        return len(self._episodes)

    def get_action_size(self):
        if self.discrete_action:
            return np.max(self._actions) + 1
        return self._actions.shape[1]

    def get_observation_shape(self):
        return self._observations.shape[1:]

    def is_action_discrete(self):
        return self.discrete_action

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self._episodes[index]

    def __iter__(self):
        return iter(self._episodes)


class Episode:
    def __init__(self, observation_shape, action_size, observations, actions,
                 rewards):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self._observations = observations
        self._actions = actions
        self._rewards = rewards

        # array of Transition
        self._transitions = self._to_transitions()

    @property
    def observations(self):
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards

    @property
    def transitions(self):
        return self._transitions

    def _to_transitions(self):
        rets = []
        num_data = self._observations.shape[0]
        for i in range(num_data - 1):
            obs_t = self._observations[i]
            act_t = self._actions[i]
            rew_t = self._rewards[i]
            obs_tp1 = self._observations[i + 1]
            act_tp1 = self._actions[i + 1]
            rew_tp1 = self._rewards[i + 1]
            ter_tp1 = 1.0 if i == num_data - 2 else 0.0
            transition = Transition(self.observation_shape, self.action_size,
                                    obs_t, act_t, rew_t, obs_tp1, act_tp1,
                                    rew_tp1, ter_tp1)
            rets.append(transition)
        return rets

    def size(self):
        return len(self._transitions)

    def get_observation_shape(self):
        return self.observation_shape

    def get_action_size(self):
        return self.action_size

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self._transitions[index]

    def __iter__(self):
        return iter(self._transitions)


class Transition:
    def __init__(self, observation_shape, action_size, obs_t, act_t, rew_t,
                 obs_tp1, act_tp1, rew_tp1, ter_tp1):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self._obs_t = obs_t
        self._act_t = act_t
        self._rew_t = rew_t
        self._obs_tp1 = obs_tp1
        self._act_tp1 = act_tp1
        self._rew_tp1 = rew_tp1
        self._ter_tp1 = ter_tp1

    def get_observation_shape(self):
        return self.observation_shape

    def get_action_size(self):
        return self.action_size

    @property
    def obs_t(self):
        return self._obs_t

    @property
    def act_t(self):
        return self._act_t

    @property
    def rew_t(self):
        return self._rew_t

    @property
    def obs_tp1(self):
        return self._obs_tp1

    @property
    def act_tp1(self):
        return self._act_tp1

    @property
    def rew_tp1(self):
        return self._rew_tp1

    @property
    def ter_tp1(self):
        return self._ter_tp1


class TransitionMiniBatch:
    def __init__(self, transitions):
        self._transitions = transitions

        obs_ts = []
        act_ts = []
        rew_ts = []
        obs_tp1s = []
        act_tp1s = []
        rew_tp1s = []
        ter_tp1s = []
        for transition in transitions:
            obs_ts.append(transition.obs_t)
            act_ts.append(transition.act_t)
            rew_ts.append(transition.rew_t)
            obs_tp1s.append(transition.obs_tp1)
            act_tp1s.append(transition.act_tp1)
            rew_tp1s.append(transition.rew_tp1)
            ter_tp1s.append(transition.ter_tp1)

        # convert list to ndarray and fix shapes
        self._obs_ts = np.array(obs_ts)
        self._act_ts = np.array(act_ts).reshape((self.size(), -1))
        self._rew_ts = np.array(rew_ts).reshape((self.size(), 1))
        self._obs_tp1s = np.array(obs_tp1s)
        self._rew_tp1s = np.array(rew_tp1s).reshape((self.size(), 1))
        self._act_tp1s = np.array(act_tp1s).reshape((self.size(), -1))
        self._ter_tp1s = np.array(ter_tp1s).reshape((self.size(), 1))

    @property
    def obs_t(self):
        return self._obs_ts

    @property
    def act_t(self):
        return self._act_ts

    @property
    def rew_t(self):
        return self._rew_ts

    @property
    def obs_tp1(self):
        return self._obs_tp1s

    @property
    def act_tp1(self):
        return self._act_tp1s

    @property
    def rew_tp1(self):
        return self._rew_tp1s

    @property
    def ter_tp1(self):
        return self._ter_tp1s

    @property
    def transitions(self):
        return self._transitions

    def size(self):
        return len(self._transitions)

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self._transitions[index]

    def __iter__(self):
        return iter(self._transitions)
