import numpy as np
import copy

from skbrl.dataset import TransitionMiniBatch


class AlgoBase:
    def __init__(self, n_epochs, batch_size):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.impl = None

    def set_params(self, **params):
        for key, val in params.items():
            assert hasattr(self, key)
            setattr(self, key, val)

    def get_params(self, deep=True):
        rets = {}
        for key in dir(self):
            # remove magic properties
            if key == '__module__':
                continue
            # pick scalar parameters
            if isinstance(getattr(self, key), (str, int, float, bool)):
                rets[key] = getattr(self, key)
        if deep:
            rets['impl'] = copy.deepcopy(self.impl)
        else:
            rets['impl'] = self.impl
        return rets

    def save_model(self, fname):
        self.impl.save_model(fname)

    def load_model(self, fname):
        self.impl.load_model(fname)

    def save_policy(self, fname):
        self.impl.save_policy(fname)

    def fit(self, episodes):
        transitions = []
        for episode in episodes:
            transitions += episode.transitions

        # instantiate implementation
        if self.impl is None:
            observation_shape = transitions[0].get_observation_shape()
            action_size = transitions[0].get_action_size()
            self.create_impl(observation_shape, action_size)

        # training loop
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(np.arange(len(transitions)))
            for itr in range(len(transitions) // self.batch_size):

                # pick transitions
                batch = []
                head_index = itr * self.batch_size
                for index in indices[head_index:head_index + self.batch_size]:
                    batch.append(transitions[index])

                loss = self.update(epoch, itr, TransitionMiniBatch(batch))

    def predict(self, x):
        return self.impl.predict_best_action(x)

    def predict_value(self, x, action):
        return self.impl.predict_value(x, action)

    def create_impl(self, observation_shape, action_size):
        raise NotImplementedError

    def update(self, epoch, itr, batch):
        raise NotImplementedError


class ImplBase:
    def save_model(self, fname):
        raise NotImplementedError

    def load_model(self, fname):
        raise NotImplementedError

    def save_policy(self, fname):
        raise NotImplementedError

    def predict_best_action(self, x):
        raise NotImplementedError

    def predict_value(self, x, action):
        raise NotImplementedError
