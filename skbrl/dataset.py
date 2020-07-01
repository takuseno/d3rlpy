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
        self._actions = np.asarray(actions)
        self._rewards = np.asarray(rewards)
        self._terminals = np.asarray(terminals)
        self.discrete_action = discrete_action
        self._episodes = None

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
        if self._episodes is None:
            self._episodes = self._to_episodes()
        return self._episodes

    def _to_episodes(self):
        rets = []
        head_index = 0
        for i in range(self.observations.shape[0]):
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
        return len(self.episodes)

    def get_action_size(self):
        if self.discrete_action:
            return np.max(self._actions) + 1
        return self._actions.shape[1]

    def get_observation_shape(self):
        return self._observations.shape[1:]

    def is_action_discrete(self):
        return self.discrete_action

    def compute_stats(self):
        episode_returns = []
        for episode in self.episodes:
            episode_returns.append(episode.compute_return())

        stats = {
            'return': {
                'mean': np.mean(episode_returns),
                'std': np.std(episode_returns),
                'min': np.min(episode_returns),
                'max': np.max(episode_returns),
            },
            'observation': {
                'mean': np.mean(self.observations, axis=0),
                'std': np.std(self.observations, axis=0)
            }
        }

        return stats

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self.episodes[index]

    def __iter__(self):
        return iter(self.episodes)


class Episode:
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
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards

    @property
    def transitions(self):
        if self._transitions is None:
            self._transitions = self._to_transitions()
        return self._transitions

    def _to_transitions(self):
        rets = []
        num_data = self._observations.shape[0]
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
        return len(self.transitions)

    def get_observation_shape(self):
        return self.observation_shape

    def get_action_size(self):
        return self.action_size

    def compute_return(self):
        return np.sum(self._rewards[1:])

    def __len__(self):
        return self.size()

    def __getitem__(self, index):
        return self.transitions[index]

    def __iter__(self):
        return iter(self.transitions)


class Transition:
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
        return self.observation_shape

    def get_action_size(self):
        return self.action_size

    @property
    def observation(self):
        return self._observation

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    @property
    def next_observation(self):
        return self._next_observation

    @property
    def next_action(self):
        return self._next_action

    @property
    def next_reward(self):
        return self._next_reward

    @property
    def terminal(self):
        return self._terminal


class TransitionMiniBatch:
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
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards

    @property
    def next_observations(self):
        return self._next_observations

    @property
    def next_actions(self):
        return self._next_actions

    @property
    def next_rewards(self):
        return self._next_rewards

    @property
    def terminals(self):
        return self._terminals

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
