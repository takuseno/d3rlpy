import numpy as np
import copy

from skbrl.dataset import TransitionMiniBatch
from skbrl.logger import SkbrlLogger


class AlgoBase:
    def __init__(self, n_epochs, batch_size):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.impl = None

    def set_params(self, **params):
        for key, val in params.items():
            assert hasattr(self, key)
            setattr(self, key, val)
        return self

    def get_params(self, deep=True):
        rets = {}
        for key in dir(self):
            # remove magic properties
            if key == '__module__':
                continue
            # pick scalar parameters
            if np.isscalar(getattr(self, key)):
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

    def fit(self,
            episodes,
            experiment_name=None,
            logdir='logs',
            verbose=True,
            tensorboard=True,
            eval_episodes=None,
            scorers=None):

        transitions = []
        for episode in episodes:
            transitions += episode.transitions

        # instantiate implementation
        if self.impl is None:
            observation_shape = transitions[0].get_observation_shape()
            action_size = transitions[0].get_action_size()
            self.create_impl(observation_shape, action_size)

        # setup logger
        logger = self._prepare_logger(experiment_name, logdir, verbose,
                                      tensorboard)

        # save hyperparameters
        logger.add_params(self.get_params(deep=False))

        # training loop
        total_step = 0
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(np.arange(len(transitions)))
            for itr in range(len(transitions) // self.batch_size):

                # pick transitions
                batch = []
                head_index = itr * self.batch_size
                for index in indices[head_index:head_index + self.batch_size]:
                    batch.append(transitions[index])

                loss = self.update(epoch, total_step,
                                   TransitionMiniBatch(batch))

                # record metrics
                for name, val in zip(self._get_loss_labels(), loss):
                    if val is not None:
                        logger.add_metric(name, val)

                total_step += 1

            if scorers and eval_episodes:
                for name, scorer in scorers.items():
                    # evaluation with test data
                    test_score = scorer(self, eval_episodes)
                    logger.add_metric(name, test_score)

            # save metrics
            logger.commit(epoch, total_step)

    def predict(self, x):
        return self.impl.predict_best_action(x)

    def predict_value(self, x, action):
        return self.impl.predict_value(x, action)

    def create_impl(self, observation_shape, action_size):
        raise NotImplementedError

    def update(self, epoch, total_step, batch):
        raise NotImplementedError

    def _get_loss_labels(self):
        raise NotImplementedError

    def _prepare_logger(self, experiment_name, logdir, verbose, tensorboard):
        if experiment_name is None:
            experiment_name = self.__class__.__name__

        logger = SkbrlLogger(experiment_name,
                             root_dir=logdir,
                             verbose=verbose,
                             tensorboard=tensorboard)

        return logger


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
