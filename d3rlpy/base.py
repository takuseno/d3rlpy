import numpy as np
import copy
import json

from abc import ABCMeta, abstractmethod
from tqdm import trange
from .preprocessing import create_scaler
from .augmentation import create_augmentation, AugmentationPipeline
from .dataset import TransitionMiniBatch
from .logger import D3RLPyLogger
from .metrics.scorer import NEGATED_SCORER
from .context import disable_parallel
from .gpu import Device


class ImplBase(metaclass=ABCMeta):
    @abstractmethod
    def save_model(self, fname):
        pass

    @abstractmethod
    def load_model(self, fname):
        pass


class LearnableBase:
    """ Algorithm base class.

    All algorithms have the shared interfaces same as scikit-learn.

    Attributes:
        n_epochs (int): the number of epochs to train.
        batch_size (int): the batch size of training.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor
        augmentation (list(str or d3rlpy.augmentation.base.Augmentation)):
            list of data augmentations.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        impl (d3rlpy.base.ImplBase): implementation object.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self, n_epochs, batch_size, scaler, augmentation, use_gpu):
        """ __init__ method.

        Args:
            n_epochs (int): the number of epochs to train.
            batch_size (int): mini-batch size.
            scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
                The available options are `['pixel', 'min_max', 'standard']`
            augmentation (list(str or d3rlpy.augmentation.base.Augmentation)):
                list of data augmentations.
            use_gpu (bool, int or d3rlpy.gpu.Device):
                flag to use GPU, device ID or device.

        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # prepare preprocessor
        if isinstance(scaler, str):
            self.scaler = create_scaler(scaler)
        else:
            self.scaler = scaler

        # prepare augmentations
        if isinstance(augmentation, AugmentationPipeline):
            self.augmentation = augmentation
        else:
            self.augmentation = AugmentationPipeline()
            for aug in augmentation:
                if isinstance(aug, str):
                    aug = create_augmentation(aug)
                self.augmentation.append(aug)

        # prepare GPU device
        # isinstance cannot tell difference between bool and int
        if type(use_gpu) == bool and use_gpu:
            self.use_gpu = Device(0)
        elif type(use_gpu) == int:
            self.use_gpu = Device(use_gpu)
        elif isinstance(use_gpu, Device):
            self.use_gpu = use_gpu
        else:
            self.use_gpu = None

        self.impl = None
        self.eval_results_ = {}

    @classmethod
    def from_json(cls, fname, use_gpu=False):
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
            use_gpu (bool, int or d3rlpy.gpu.Device):
                flag to use GPU, device ID or device.

        Returns:
            d3rlpy.base.LearnableBase: algorithm.

        """
        with open(fname, 'r') as f:
            params = json.load(f)

        observation_shape = tuple(params['observation_shape'])
        action_size = params['action_size']
        del params['observation_shape']
        del params['action_size']

        # create scaler object
        if params['scaler']:
            scaler_type = params['scaler']['type']
            scaler_params = params['scaler']['params']
            scaler = create_scaler(scaler_type, **scaler_params)
            params['scaler'] = scaler

        # create augmentation objects
        augmentations = []
        for param in params['augmentation']:
            aug_type = param['type']
            aug_params = param['params']
            augmentation = create_augmentation(aug_type, **aug_params)
            augmentations.append(augmentation)
        params['augmentation'] = AugmentationPipeline(augmentations)

        # overwrite use_gpu flag
        params['use_gpu'] = use_gpu

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
            # remove protected properties
            if key[-1] == '_':
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

    def fit(self,
            episodes,
            experiment_name=None,
            with_timestamp=True,
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
            with_timestamp (bool): flag to add timestamp string to the last of
                directory name.
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
        logger = self._prepare_logger(experiment_name, with_timestamp, logdir,
                                      verbose, tensorboard)

        # save hyperparameters
        self._save_params(logger)

        # refresh evaluation metrics
        self.eval_results_ = {}

        # hold original dataset
        env_transitions = transitions

        # training loop
        total_step = 0
        for epoch in range(self.n_epochs):

            # data augmentation
            new_transitions = self._generate_new_data(env_transitions)
            if new_transitions:
                transitions = env_transitions + new_transitions

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

    def _generate_new_data(self, transitions):
        """ Returns generated transitions for data augmentation.

        This method is called at the beginning of every epoch.

        Args:
            transitions (list(d3rlpy.dataset.Transition)): list of transitions.

        Returns:
            list(d3rlpy.dataset.Transition): list of new transitions.

        """
        return None

    def _get_loss_labels(self):
        raise NotImplementedError

    def _prepare_logger(self, experiment_name, with_timestamp, logdir, verbose,
                        tensorboard):
        if experiment_name is None:
            experiment_name = self.__class__.__name__

        logger = D3RLPyLogger(experiment_name,
                              root_dir=logdir,
                              verbose=verbose,
                              tensorboard=tensorboard,
                              with_timestamp=with_timestamp)

        return logger

    def _evaluate(self, episodes, scorers, logger):
        for name, scorer in scorers.items():
            # evaluation with test data
            test_score = scorer(self, episodes)

            # higher scorer's scores are better in scikit-learn.
            # make it back to its original sign here.
            if scorer in NEGATED_SCORER:
                test_score *= -1

            # logging metrics
            logger.add_metric(name, test_score)

            # store metric locally
            if name not in self.eval_results_:
                self.eval_results_[name] = []
            self.eval_results_[name].append(test_score)

    def _save_params(self, logger):
        # get hyperparameters without impl
        params = {}
        with disable_parallel():
            for k, v in self.get_params(deep=False).items():
                if isinstance(v, (ImplBase, LearnableBase)):
                    continue
                params[k] = v

        # save shapes
        params['observation_shape'] = self.impl.observation_shape
        params['action_size'] = self.impl.action_size

        # save scaler
        if self.scaler:
            params['scaler'] = {
                'type': self.scaler.get_type(),
                'params': self.scaler.get_params()
            }

        # save augmentations
        params['augmentation'] = []
        aug_types = self.augmentation.get_type()
        aug_params = self.augmentation.get_params()
        for aug_type, aug_param in zip(aug_types, aug_params):
            params['augmentation'].append({
                'type': aug_type,
                'params': aug_param
            })

        # save GPU device id
        if self.use_gpu:
            params['use_gpu'] = self.use_gpu.get_id()

        logger.add_params(params)
