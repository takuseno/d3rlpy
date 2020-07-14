import numpy as np
import torch

from abc import ABCMeta, abstractmethod


class Scaler(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, episodes):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def get_type(self):
        pass

    @abstractmethod
    def get_params(self):
        pass


def create_scaler(scaler_type, **kwargs):
    if scaler_type == 'pixel':
        return PixelScaler()
    elif scaler_type == 'min_max':
        return MinMaxScaler(**kwargs)
    elif scaler_type == 'standard':
        return StandardScaler(**kwargs)
    raise ValueError


class PixelScaler(Scaler):
    """ Pixel normalization preprocessing.

    .. math::

        x' = x / 255

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with PixelScaler
        cql = CQL(scaler='pixel')

        cql.fit(dataset.episodes)

    """
    def fit(self, episodes):
        pass

    def transform(self, x):
        """ Returns normalized pixel observations.

        Args:
            x (torch.Tensor): pixel observation tensor.

        Returns:
            torch.Tensor: normalized pixel observation tensor.

        """
        return x.float() / 255.0

    def get_params(self):
        """ Returns scaling parameters.

        PixelScaler returns empty dictiornary.

        Returns:
            dict: empty dictionary.

        """
        return {}

    def get_type(self):
        """ Returns scaler type.

        Returns:
            str: `pixel`.

        """
        return 'pixel'


class MinMaxScaler(Scaler):
    """ Min-Max normalization preprocessing.

    .. math::

        x' = (x - \\min{x}) / (\\max{x} - \\min{x})

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with MinMaxScaler
        cql = CQL(scaler='min_max')

        # scaler is initialized from the given episodes
        cql.fit(dataset.episodes)

    You can also initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxScaler

        # initialize with dataset
        scaler = MinMaxScaler(dataset)

        # initialize manually
        minimum = observations.min(axis=0)
        maximum = observations.max(axis=0)
        scaler = MinMaxScaler(minimum=minimum, maximum=maximum)

        cql = CQL(scaler=scaler)

    Args:
        dataset (d3rlpy.dataset.MDPDataset): dataset object.
        min (numpy.ndarray): minimum values at each entry.
        max (numpy.ndarray): maximum values at each entry.

    Attributes:
        minimum (numpy.ndarray): minimum values at each entry.
        maximum (numpy.ndarray): maximum values at each entry.

    """
    def __init__(self, dataset=None, maximum=None, minimum=None):
        if dataset:
            stats = dataset.compute_stats()
            self.minimum = stats['observation']['min']
            self.maximum = stats['observation']['max']
        elif maximum is not None and minimum is not None:
            self.minimum = np.asarray(minimum)
            self.maximum = np.asarray(maximum)
        else:
            self.minimum = None
            self.maximum = None

    def fit(self, episodes):
        """ Fits minimum and maximum from list of episodes.

        Args:
            episodes (list(d3rlpy.dataset.Episode)): list of episodes.

        """
        if self.minimum is not None and self.maximum is not None:
            return

        for i, e in enumerate(episodes):
            observations = np.asarray(e.observations)
            if i == 0:
                self.minimum = observations.min(axis=0)
                self.maximum = observations.max(axis=0)
                continue
            self.minimum = np.minimum(self.minimum, observations.min(axis=0))
            self.maximum = np.maximum(self.maximum, observations.max(axis=0))

    def transform(self, x):
        """ Returns normalized observation tensor.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: normalized observation tensor.

        """
        minimum = torch.tensor(self.minimum,
                               dtype=torch.float32,
                               device=x.device)
        maximum = torch.tensor(self.maximum,
                               dtype=torch.float32,
                               device=x.device)
        return (x - minimum.view(1, -1)) / (maximum - minimum).view(1, -1)

    def get_params(self):
        """ Returns scaling parameters.

        Returns:
            dict: `maximum` and `minimum`.

        """
        return {'maximum': self.maximum, 'minimum': self.minimum}

    def get_type(self):
        """ Returns scaler type.

        Returns:
            str: `min_max`.

        """
        return 'min_max'


class StandardScaler(Scaler):
    """ Standardization preprocessing.

    .. math::

        x' = (x - \\mu) / \\sigma

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with StandardScaler
        cql = CQL(scaler='standard')

        # scaler is initialized from the given episodes
        cql.fit(dataset.episodes)

    You can initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import StandardScaler

        # initialize with dataset
        scaler = StandardScaler(dataset)

        # initialize manually
        mean = observations.mean(axis=0)
        std = observations.std(axis=0)
        scaler = StandardScaler(mean=mean, std=std)

        cql = CQL(scaler=scaler)

    Args:
        dataset (d3rlpy.dataset.MDPDataset): dataset object.
        mean (numpy.ndarray): mean values at each entry.
        std (numpy.ndarray): standard deviation at each entry.

    Attributes:
        mean (numpy.ndarray): mean values at each entry.
        std (numpy.ndarray): standard deviation values at each entry.

    """
    def __init__(self, dataset=None, mean=None, std=None):
        if dataset:
            stats = dataset.compute_stats()
            self.mean = stats['observation']['mean']
            self.std = stats['observation']['std']
        elif mean is not None and std is not None:
            self.mean = np.asarray(mean)
            self.std = np.asarray(std)
        else:
            self.mean = None
            self.std = None

    def fit(self, episodes):
        """ Fits mean and standard deviation from list of episodes.

        Args:
            episodes (list(d3rlpy.dataset.Episode)): list of episodes.

        """
        if self.mean is not None and self.std is not None:
            return

        # compute mean
        total_sum = np.zeros(episodes[0].observation_shape)
        total_count = 0
        for e in episodes:
            observations = np.asarray(e.observations)
            total_sum += observations.sum(axis=0)
            total_count += observations.shape[0]
        self.mean = total_sum / total_count

        # compute stdandard deviation
        total_sqsum = np.zeros(episodes[0].observation_shape)
        expanded_mean = self.mean.reshape((1, -1))
        for e in episodes:
            observations = np.asarray(e.observations)
            total_sqsum += ((observations - expanded_mean)**2).sum(axis=0)
        self.std = np.sqrt(total_sqsum / total_count)

    def transform(self, x):
        """ Returns standardized observation tensor.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: standardized observation tensor.

        """
        mean = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self.std, dtype=torch.float32, device=x.device)
        return (x - mean.view(1, -1)) / std.view(1, -1)

    def get_params(self):
        """ Returns scaling parameters.

        Returns:
            dict: `mean` and `std`.

        """
        return {'mean': self.mean, 'std': self.std}

    def get_type(self):
        """ Returns scaler type.

        Returns:
            str: `standard`.

        """
        return 'standard'
