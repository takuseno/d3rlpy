import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Union
from typing_extensions import Protocol


class _ActionProtocol(Protocol):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    @property
    def action_size(self) -> Optional[int]:
        ...


class Explorer(metaclass=ABCMeta):
    @abstractmethod
    def sample(
        self, algo: _ActionProtocol, x: np.ndarray, step: int
    ) -> np.ndarray:
        pass


class LinearDecayEpsilonGreedy(Explorer):
    """:math:`\\epsilon`-greedy explorer with linear decay schedule.

    Args:
        start_epsilon (float): the beginning :math:`\\epsilon`.
        end_epsilon (float): the end :math:`\\epsilon`.
        duration (int): the scheduling duration.

    """

    _start_epsilon: float
    _end_epsilon: float
    _duration: int

    def __init__(
        self,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.1,
        duration: int = 1000000,
    ):
        self._start_epsilon = start_epsilon
        self._end_epsilon = end_epsilon
        self._duration = duration

    def sample(
        self, algo: _ActionProtocol, x: np.ndarray, step: int
    ) -> np.ndarray:
        """Returns :math:`\\epsilon`-greedy action.

        Args:
            algo (d3rlpy.algos.base.AlgoBase): algorithm.
            x (numpy.ndarray): observation.
            step (int): current environment step.

        Returns:
            int: :math:`\\epsilon`-greedy action.

        """
        if np.random.random() < self.compute_epsilon(step):
            assert algo.action_size is not None
            return np.random.randint(algo.action_size)
        return algo.predict([x])[0]

    def compute_epsilon(self, step: int) -> float:
        """Returns decayed :math:`\\epsilon`.

        Returns:
            float: :math:`\\epsilon`.

        """
        if step >= self._duration:
            return self._end_epsilon
        base = self._start_epsilon - self._end_epsilon
        return base * (1.0 - step / self._duration) + self._end_epsilon


class NormalNoise(Explorer):
    """Normal noise explorer.

    Args:
        mean (float): mean.
        std (float): standard deviation.

    """

    _mean: float
    _std: float

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self._mean = mean
        self._std = std

    def sample(
        self, algo: _ActionProtocol, x: np.ndarray, step: int
    ) -> np.ndarray:
        """Returns action with noise injection.

        Args:
            algo (d3rlpy.algos.base.AlgoBase): algorithm.
            x (numpy.ndarray): observation.

        Returns:
            numpy.ndarray: action with noise injection.

        """
        action = algo.sample_action([x])[0]
        noise = np.random.normal(self._mean, self._std, size=action.shape)
        return np.clip(action + noise, -1.0, 1.0)
