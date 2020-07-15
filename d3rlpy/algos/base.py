import numpy as np
import copy
import json

from abc import ABCMeta, abstractmethod
from tqdm import trange
from ..preprocessing import create_scaler
from ..dataset import TransitionMiniBatch
from ..logger import D3RLPyLogger
from ..metrics.scorer import NEGATED_SCORER


class ImplBase(metaclass=ABCMeta):
    @abstractmethod
    def save_model(self, fname):
        pass

    @abstractmethod
    def load_model(self, fname):
        pass

    @abstractmethod
    def save_policy(self, fname):
        pass

    @abstractmethod
    def predict_best_action(self, x):
        pass

    @abstractmethod
    def predict_value(self, x, action):
        pass


class AlgoBase:
    """ Algorithm base class.

    All algorithms have the shared interfaces same as scikit-learn.

    Attributes:
        n_epochs (int): the number of epochs to train.
        batch_size (int): the batch size of training.
        impl (d3rlpy.algos.base.ImplBase): implementation of the algorithm.

    """
    def __init__(self, n_epochs, batch_size, scaler):
        """ __init__ method.

        Args:
            n_epochs (int): the number of epochs to train.
            batch_size (int): mini-batch size.
            scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
                The available options are `['pixel', 'min_max', 'standard']`

        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        if isinstance(scaler, str):
            self.scaler = create_scaler(scaler)
        else:
            self.scaler = scaler

        self.impl = None

    @classmethod
    def from_json(cls, fname):
        """ Returns algorithm configured with json file.

        The Json file should be the one saved during fitting.

        .. code-block:: python

            from d3rlpy.algos import Algo

            # create algorithm with saved configuration
            algo = Algo.from_json('d3rlpy_logs/<path-to-json>/params.json')

            # ready to load
            algo.load_model('d3rlpy_logs/<path-to-model>/model_100.pt')

            # ready to predict
            algo.predict(...)

        Args:
            fname (str): file path to `params.json`.

        Returns:
            d3rlpy.algos.base.AlgoBase: algorithm.

        """
        with open(fname, 'r') as f:
            params = json.load(f)

        observation_shape = tuple(params['observation_shape'])
        action_size = params['action_size']
        del params['observation_shape']
        del params['action_size']

        if params['scaler']:
            scaler_type = params['scaler']['type']
            scaler_params = params['scaler']['params']
            scaler = create_scaler(scaler_type, **scaler_params)
            params['scaler'] = scaler

        algo = cls(**params)
        algo.create_impl(observation_shape, action_size)
        return algo

    def set_params(self, **params):
        """ Sets the given arguments to the attributes if they exist.

        This method sets the given values to the attributes including ones in
        subclasses. If the values that don't exist as attributes are
        passed, they are ignored.
        Some of scikit-learn utilities will use this method.

        .. code-block:: python

            algo.set_params(n_epochs=10, batch_size=100)

        Args:
            **params: arbitrary inputs to set as attributes.

        Returns:
            d3rlpy.algos.base.AlgoBase: itself.

        """
        for key, val in params.items():
            assert hasattr(self, key)
            setattr(self, key, val)
        return self

    def get_params(self, deep=True):
        """ Returns the all attributes.

        This method returns the all attributes including ones in subclasses.
        Some of scikit-learn utilities will use this method.

        .. code-block:: python

            params = algo.get_params(deep=True)

            # the returned values can be used to instantiate the new object.
            algo2 = AlgoBase(**params)

        Args:
            deep (bool): flag to deeply copy objects such as `impl`.

        Returns:
            dict: attribute values in dictionary.

        """
        rets = {}
        for key in dir(self):
            # remove magic properties
            if key[:2] == '__':
                continue
            # pick scalar parameters
            value = getattr(self, key)
            if np.isscalar(value):
                rets[key] = value
            elif isinstance(value, object) and not callable(value):
                if deep:
                    rets[key] = copy.deepcopy(value)
                else:
                    rets[key] = value
        return rets

    def save_model(self, fname):
        """ Saves neural network parameters.

        .. code-block:: python

            algo.save_model('model.pt')

        Args:
            fname (str): destination file path.

        """
        self.impl.save_model(fname)

    def load_model(self, fname):
        """ Load neural network parameters.

        .. code-block:: python

            algo.load_model('model.pt')

        Args:
            fname (str): source file path.

        """
        self.impl.load_model(fname)

    def save_policy(self, fname):
        """ Save the greedy-policy computational graph as TorchScript.

        .. code-block:: python

            algo.save_policy('policy.pt')

        The artifacts saved with this method will work without any dependencies
        except pytorch.
        This method is especially useful to deploy the learned policy to
        production environments or embedding systems.

        See also

            * https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html (for Python).
            * https://pytorch.org/tutorials/advanced/cpp_export.html (for C++).

        Args:
            fname (str): destination file path.

        """
        self.impl.save_policy(fname)

    def fit(self,
            episodes,
            experiment_name=None,
            logdir='d3rlpy_logs',
            verbose=True,
            show_progress=True,
            tensorboard=True,
            eval_episodes=None,
            save_interval=1,
            scorers=None):
        """ Trains with the given dataset.

        .. code-block:: python

            algo.fit(episodes)

        Args:
            episodes (list(d3rlpy.dataset.Episode)): list of episodes to train.
            experiment_name (str): experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            logdir (str): root directory name to save logs.
            verbose (bool): flag to show logged information on stdout.
            show_progress (bool): flag to show progress bar for iterations.
            tensorboard (bool): flag to save logged information in tensorboard
                (additional to the csv data)
            eval_episodes (list(d3rlpy.dataset.Episode)):
                list of episodes to test.
            save_interval (int): interval to save parameters.
            scorers (list(callable)):
                list of scorer functions used with `eval_episodes`.

        """

        transitions = []
        for episode in episodes:
            transitions += episode.transitions

        # initialize scaler
        if self.scaler:
            self.scaler.fit(episodes)

        # instantiate implementation
        observation_shape = transitions[0].get_observation_shape()
        action_size = transitions[0].get_action_size()
        if self.impl is None:
            self.create_impl(observation_shape, action_size)

        # setup logger
        logger = self._prepare_logger(experiment_name, logdir, verbose,
                                      tensorboard)

        # save hyperparameters
        self._save_params(logger, observation_shape, action_size)

        # training loop
        total_step = 0
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(np.arange(len(transitions)))
            n_iters = len(transitions) // self.batch_size
            range_gen = trange(n_iters) if show_progress else range(n_iters)
            for itr in range_gen:
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
                self._evaluate(eval_episodes, scorers, logger)

            # save metrics
            logger.commit(epoch, total_step)

            # save model parameters and greedy policy
            if epoch % save_interval == 0:
                logger.save_model(epoch, self)

    def predict(self, x):
        """ Returns greedy actions.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            actions = algo.predict(x)
            # actions.shape == (100, action size) for continuous control
            # actions.shape == (100,) for discrete control

        Args:
            x (numpy.ndarray): observations

        Returns:
            numpy.ndarray: greedy actions

        """
        return self.impl.predict_best_action(x)

    def predict_value(self, x, action):
        """ Returns predicted action-values.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            # for continuous control
            # 100 actions with shape of (2,)
            actions = np.random.random((100, 2))

            # for discrete control
            # 100 actions in integer values
            actions = np.random.randint(2, size=100)

            values = algo.predict_value(x, actions)
            # values.shape == (100,)

        Args:
            x (numpy.ndarray): observations
            action (numpy.ndarray): actions

        Returns:
            numpy.ndarray: predicted action-values

        """
        return self.impl.predict_value(x, action)

    def create_impl(self, observation_shape, action_size):
        """ Instantiate implementation objects with the dataset shapes.

        This method will be used internally when `fit` method is called.

        Args:
            observation_shape (tuple): observation shape.
            action_size (int): dimension of action-space.

        """
        raise NotImplementedError

    def update(self, epoch, total_step, batch):
        """ Update parameters with mini-batch of data.

        Args:
            epoch (int): the current number of epochs.
            total_step (int): the current number of total iterations.
            batch (d3rlpy.dataset.TransitionMiniBatch): mini-batch data.

        Returns:
            list: loss values. 

        """
        raise NotImplementedError

    def _get_loss_labels(self):
        raise NotImplementedError

    def _prepare_logger(self, experiment_name, logdir, verbose, tensorboard):
        if experiment_name is None:
            experiment_name = self.__class__.__name__

        logger = D3RLPyLogger(experiment_name,
                              root_dir=logdir,
                              verbose=verbose,
                              tensorboard=tensorboard)

        return logger

    def _evaluate(self, episodes, scorers, logger):
        for name, scorer in scorers.items():
            # evaluation with test data
            test_score = scorer(self, episodes)

            # higher scorer's scores are better in scikit-learn.
            # make it back to its original sign here.
            if scorer in NEGATED_SCORER:
                test_score *= -1

            logger.add_metric(name, test_score)

    def _save_params(self, logger, observation_shape, action_size):
        # get hyperparameters without impl
        params = self.get_params(deep=False)
        params = {k: v for k, v in params.items() if k != 'impl'}

        # save shapes
        params['observation_shape'] = observation_shape
        params['action_size'] = action_size

        # save scaler
        if self.scaler:
            params['scaler'] = {
                'type': self.scaler.get_type(),
                'params': self.scaler.get_params()
            }

        logger.add_params(params)
