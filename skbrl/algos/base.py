import numpy as np


class AlgoBase:
    def __init__(self, n_epochs, batch_size):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.impl = None

    def set_params(self, **params):
        raise NotImplementedError

    def get_params(self, deep=True):
        raise NotImplementedError

    def save_model(self, fname):
        self.impl.save_model(fname)

    def load_model(self, fname):
        self.impl.load_model(fname)

    def save_policy(self, fname):
        self.impl.save_policy(fname)

    def fit(self, episodes):
        transitions = []
        for episode in episodes:
            transitions += episode.get_transitions()

        # instantiate implementation
        if self.impl is None:
            observation_shape = transitions[0].get_observation_shape()
            action_size = transitions[0].get_action_size()
            self.create_impl(observation_shape, action_size)

        # training loop
        for epoch in self.n_epochs:
            indices = np.permutation(np.arange(len(transitions)))
            for itr in range(len(transitions) // self.batch_size):
                obs_ts = []
                act_ts = []
                rew_tp1s = []
                obs_tp1s = []
                ter_tp1s = []
                head_index = itr * self.batch_size
                for index in indices[head_index:head_index + self.batch_size]:
                    obs_ts.append(transitions[index].obs_t)
                    act_ts.append(transitions[index].act_t)
                    rew_tp1s.append(transitions[index].rew_tp1)
                    obs_tp1s.append(transitions[index].obs_tp1)
                    ter_tp1s.append(transitions[index].ter_tp1)
                loss = self.update(epoch, itr, obs_ts, act_ts, rew_tp1s,
                                   obs_tp1s, ter_tp1s)

    def predict(self, x):
        return self.impl.predict_best_action(x)

    def create_impl(self, observation_shape, action_size):
        raise NotImplementedError

    def update(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        raise NotImplementedError
