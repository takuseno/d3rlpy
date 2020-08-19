import torch

from .base import Augmentation


class SingleAmplitudeScaling(Augmentation):
    """ Single Amplitude Scaling augmentation.

    .. math::

        x' = x + z

    where :math:`z \\sim \\text{Unif}(minimum, maximum)`.

    Args:
        minimum (float): minimum amplitude scale.
        maximum (float): maximum amplitude scale.

    Attributes:
        minimum (float): minimum amplitude scale.
        maximum (float): maximum amplitude scale.

    """
    def __init__(self, minimum=0.8, maximum=1.2):
        self.minimum = minimum
        self.maximum = maximum

    def transform(self, x):
        """ Returns scaled observation.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        z = torch.empty(x.shape[0], 1, device=x.device)
        z.uniform_(self.minimum, self.maximum)
        return x * z

    def get_type(self):
        """ Returns augmentation type.

        Returns:
            str: `single_amplitude_scaling`.

        """
        return 'single_amplitude_scaling'

    def get_params(self):
        """ Returns augmentation parameters.

        Returns:
            dict: augmentation parameters.

        """
        return {'minimum': self.minimum, 'maximum': self.maximum}


class MultipleAmplitudeScaling(SingleAmplitudeScaling):
    """ Multiple Amplitude Scaling augmentation.

    .. math::

        x' = x + z

    where :math:`z \\sim \\text{Unif}(minimum, maximum)` and :math:`z`
    is a vector with different amplitude scale on each.

    Args:
        minimum (float): minimum amplitude scale.
        maximum (float): maximum amplitude scale.

    Attributes:
        minimum (float): minimum amplitude scale.
        maximum (float): maximum amplitude scale.

    """
    def transform(self, x):
        z = torch.empty(*x.shape, device=x.device)
        z.uniform_(self.minimum, self.maximum)
        return x * z

    def get_type(self):
        """ Returns augmentation type.

        Returns:
            str: `multiple_amplitude_scaling`.

        """
        return 'multiple_amplitude_scaling'
