import numpy as np

from abc import ABCMeta, abstractmethod


class Explorer(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, algo, x, step):
        pass


class LinearDecayEpsilonGreedy(Explorer):
    """ :math:`\\epsilon`-greedy explorer with linear decay schedule.

    Args:
        start_epsilon (float): the beginning :math:`\\epsilon`.
        end_epsilon (float): the end :math:`\\epsilon`.
        duration (int): the scheduling duration.

    Attributes:
        start_epsilon (float): the beginning :math:`\\epsilon`.
        end_epsilon (float): the end :math:`\\epsilon`.
        duration (int): the scheduling duration.

    """
    def __init__(self, start_epsilon=1.0, end_epsilon=0.1, duration=1000000):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.duration = duration

    def sample(self, algo, x, step):
        """ Returns :math:`\\epsilon`-greedy action.

        Args:
            algo (d3rlpy.algos.base.AlgoBase): algorithm.
            x (numpy.ndarray): observation.
            step (int): current environment step.

        Returns:
            int: :math:`\\epsilon`-greedy action.

        """
        if np.random.random() < self.compute_epsilon(step):
            return np.random.randint(algo.impl.action_size)
        return algo.predict([x])[0]

    def compute_epsilon(self, step):
        """ Returns decayed :math:`\\epsilon`.

        Returns:
            float: :math:`\\epsilon`.

        """
        if step >= self.duration:
            return self.end_epsilon
        base = self.start_epsilon - self.end_epsilon
        return base * (1.0 - step / self.duration) + self.end_epsilon


class NormalNoise(Explorer):
    """ Normal noise explorer.

    Args:
        mean (float): mean.
        std (float): standard deviation.

    Attributes:
        mean (float): mean.
        std (float): standard deviation.

    """
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def sample(self, algo, x, *args):
        """ Returns action with noise injection.

        Args:
            algo (d3rlpy.algos.base.AlgoBase): algorithm.
            x (numpy.ndarray): observation.

        Returns:
            numpy.ndarray: action with noise injection.

        """
        action = algo.sample_action([x])[0]
        noise = np.random.normal(self.mean, self.std, size=action.shape)
        return np.clip(action + noise, -1.0, 1.0)
