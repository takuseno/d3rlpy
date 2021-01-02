import numpy as np
import torch

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional
from ..dataset import Episode, MDPDataset


class Scaler(metaclass=ABCMeta):

    TYPE: ClassVar[str] = "none"

    @abstractmethod
    def fit(self, episodes: List[Episode]) -> None:
        pass

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_type(self) -> str:
        """Returns a scaler type.

        Returns:
            str: scaler type.

        """
        return self.TYPE

    @abstractmethod
    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        pass


class PixelScaler(Scaler):
    """Pixel normalization preprocessing.

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

    TYPE: ClassVar[str] = "pixel"

    def fit(self, episodes: List[Episode]) -> None:
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized pixel observations.

        Args:
            x (torch.Tensor): pixel observation tensor.

        Returns:
            torch.Tensor: normalized pixel observation tensor.

        """
        return x.float() / 255.0

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed observations.

        Args:
            x (torch.Tensor): normalized observation tensor.

        Returns:
            torch.Tensor: unnormalized pixel observation tensor.

        """
        return (x * 255.0).long()

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns scaling parameters.

        PixelScaler returns empty dictiornary.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: empty dictionary.

        """
        return {}


class MinMaxScaler(Scaler):
    r"""Min-Max normalization preprocessing.

    .. math::

        x' = (x - \min{x}) / (\max{x} - \min{x})

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

    TYPE: ClassVar[str] = "min_max"
    _minimum: Optional[np.ndarray]
    _maximum: Optional[np.ndarray]

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        maximum: Optional[np.ndarray] = None,
        minimum: Optional[np.ndarray] = None,
    ):
        self._minimum = None
        self._maximum = None
        if dataset:
            self.fit(dataset.episodes)
        elif maximum is not None and minimum is not None:
            self._minimum = np.asarray(minimum)
            self._maximum = np.asarray(maximum)

    def fit(self, episodes: List[Episode]) -> None:
        """Fits minimum and maximum from list of episodes.

        Args:
            episodes (list(d3rlpy.dataset.Episode)): list of episodes.

        """
        if self._minimum is not None and self._maximum is not None:
            return

        for i, e in enumerate(episodes):
            observations = np.asarray(e.observations)
            if i == 0:
                minimum = observations.min(axis=0)
                maximum = observations.max(axis=0)
                continue
            minimum = np.minimum(minimum, observations.min(axis=0))
            maximum = np.maximum(maximum, observations.max(axis=0))

        self._minimum = minimum.reshape((1,) + minimum.shape)
        self._maximum = maximum.reshape((1,) + maximum.shape)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized observation tensor.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: normalized observation tensor.

        """
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=x.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=x.device
        )
        return (x - minimum) / (maximum - minimum)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed observations.

        Args:
            x (torch.Tensor): normalized observation tensor.

        Returns:
            torch.Tensor: unnormalized observation tensor.

        """
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=x.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=x.device
        )
        return ((maximum - minimum) * x) + minimum

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns scaling parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: `maximum` and `minimum`.

        """
        if self._maximum is not None:
            maximum = self._maximum.copy() if deep else self._maximum
        else:
            maximum = None

        if self._minimum is not None:
            minimum = self._minimum.copy() if deep else self._minimum
        else:
            minimum = None

        return {"maximum": maximum, "minimum": minimum}


class StandardScaler(Scaler):
    r"""Standardization preprocessing.

    .. math::

        x' = (x - \mu) / \sigma

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

    TYPE = "standard"
    _mean: Optional[np.ndarray]
    _std: Optional[np.ndarray]

    def __init__(
        self,
        dataset: Optional[MDPDataset] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self._mean = None
        self._std = None
        if dataset:
            self.fit(dataset.episodes)
        elif mean is not None and std is not None:
            self._mean = np.asarray(mean)
            self._std = np.asarray(std)

    def fit(self, episodes: List[Episode]) -> None:
        """Fits mean and standard deviation from list of episodes.

        Args:
            episodes (list(d3rlpy.dataset.Episode)): list of episodes.

        """
        if self._mean is not None and self._std is not None:
            return

        # compute mean
        total_sum = np.zeros(episodes[0].get_observation_shape())
        total_count = 0
        for e in episodes:
            observations = np.asarray(e.observations)
            total_sum += observations.sum(axis=0)
            total_count += observations.shape[0]
        mean = total_sum / total_count

        # compute stdandard deviation
        total_sqsum = np.zeros(episodes[0].get_observation_shape())
        expanded_mean = mean.reshape((1,) + mean.shape)
        for e in episodes:
            observations = np.asarray(e.observations)
            total_sqsum += ((observations - expanded_mean) ** 2).sum(axis=0)
        std = np.sqrt(total_sqsum / total_count)

        self._mean = mean.reshape((1,) + mean.shape)
        self._std = std.reshape((1,) + std.shape)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns standardized observation tensor.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: standardized observation tensor.

        """
        assert self._mean is not None and self._std is not None
        mean = torch.tensor(self._mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self._std, dtype=torch.float32, device=x.device)
        return (x - mean) / std

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed observation tensor.

        Args:
            x (torch.Tensor): standardized observation tensor.

        Returns:
            torch.Tensor: unstandardized observation tensor.

        """
        assert self._mean is not None and self._std is not None
        mean = torch.tensor(self._mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self._std, dtype=torch.float32, device=x.device)
        return (std * x) + mean

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns scaling parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: `mean` and `std`.

        """
        if self._mean is not None:
            mean = self._mean.copy() if deep else self._mean
        else:
            mean = None

        if self._std is not None:
            std = self._std.copy() if deep else self._std
        else:
            std = None

        return {"mean": mean, "std": std}


SCALER_LIST = {}


def register_scaler(cls: Any) -> None:
    """Registers scaler class.

    Args:
        cls (type): scaler class inheriting ``Scaler``.

    """
    is_registered = cls.TYPE in SCALER_LIST
    assert not is_registered, "%s seems to be already registered" % cls.TYPE
    SCALER_LIST[cls.TYPE] = cls


def create_scaler(name: str, **kwargs: Dict[str, Any]) -> Scaler:
    """Returns registered scaler object.

    Args:
        name (str): regsitered scaler type name.
        kwargs (any): scaler arguments.

    Returns:
        d3rlpy.preprocessing.scalers: scaler object.

    """
    assert name in SCALER_LIST, "%s seems not to be registered." % name
    scaler = SCALER_LIST[name](**kwargs)
    assert isinstance(scaler, Scaler)
    return scaler


register_scaler(PixelScaler)
register_scaler(MinMaxScaler)
register_scaler(StandardScaler)
