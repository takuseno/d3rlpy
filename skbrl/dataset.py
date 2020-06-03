import numpy as np
import pandas as pd

from PIL import Image


def _compute_rewards(reward_func, observations, actions, terminals):
    rewards = []
    for i in range(observations.shape[0]):
        if terminals[i]:
            continue
        obs_t = observations[i]
        obs_tp1 = observations[i + 1]
        act_t = actions[i]
        reward = reward_func(obs_t, act_t, obs_tp1)
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
    """

    """
    def __init__(self,
                 observations,
                 actions,
                 rewards,
                 terminals,
                 discrete=False):
        """
        """
        self._observations = np.array(observations)
        self._actions = np.array(actions)
        self._rewards = np.array(rewards)
        self._terminals = np.array(terminals)
        self.discrete = discrete

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

    def get_episodes(self):
        """
        """
        rets = []
        observations = []
        actions = []
        rewards = []
        for i in range(self.size()):
            observations.append(self._observations[i])
            actions.append(self._actions[i])
            rewards.append(self._rewards[i])
            if self._terminals[i]:
                episode = Episode(self.get_observation_shape(),
                                  self.get_action_size(), observations,
                                  actions, rewards)
                rets.append(episode)
                observations = []
                actions = []
                rewards = []
        return rets

    def size(self):
        """
        """
        return self._observations.shape[0]

    def get_action_size(self):
        """
        """
        if self.discrete:
            return np.max(self._actions) + 1
        return self._actions.shape[1]

    def get_observation_shape(self):
        """
        """
        return self._observations.shape[1:]


class Episode:
    """
    """
    def __init__(self, observation_shape, action_size, observations, actions,
                 rewards):
        """
        """
        self.observation_shape = observation_shape
        self.action_size = action_size
        self._observations = observations
        self._actions = actions
        self._rewards = rewards

    @property
    def observations(self):
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards

    def get_transitions(self):
        """
        """
        rets = []
        for i in range(self.size() - 1):
            obs_t = self._observations[i]
            act_t = self._actions[i]
            rew_t = self._rewards[i]
            obs_tp1 = self._observations[i + 1]
            act_tp1 = self._actions[i + 1]
            rew_tp1 = self._rewards[i + 1]
            ter_tp1 = i == self.size() - 2
            transition = Transition(self.observation_shape, self.action_size,
                                    obs_t, act_t, rew_t, obs_tp1, act_tp1,
                                    rew_tp1, ter_tp1)
            rets.append(transition)
        return rets

    def size(self):
        return self._observations.shape[0]

    def get_observation_shape(self):
        return self.observation_shape

    def get_action_size(self):
        return self.action_size


class Transition:
    """
    """
    def __init__(self, observation_shape, action_size, obs_t, act_t, rew_t,
                 obs_tp1, act_tp1, rew_tp1, ter_tp1):
        """
        """
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.obs_t = obs_t
        self.act_t = act_t
        self.rew_t = rew_t
        self.obs_tp1 = obs_tp1
        self.act_tp1 = act_tp1
        self.rew_tp1 = rew_tp1
        self.ter_tp1 = ter_tp1
