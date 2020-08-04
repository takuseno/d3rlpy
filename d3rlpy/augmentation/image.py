import torch
import torch.nn as nn
import kornia.augmentation as aug

from .base import Augmentation


class RandomShift(Augmentation):
    """ Random shift augmentation.

    Args:
        shift_size (int): size to shift image.

    Attributes:
        shift_size (int): size to shift image.

    """
    def __init__(self, shift_size=4):
        self.shift_size = shift_size
        self._operation = None

    def _setup(self, x):
        height, width = x.shape[-2:]
        self._operation = nn.Sequential(nn.ReplicationPad2d(self.shift_size),
                                        aug.RandomCrop((height, width)))

    def transform(self, x):
        """ Returns shifted images.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        if not self._operation:
            self._setup(x)
        return self._operation(x)

    def get_type(self):
        """ Returns augmentation type.

        Returns:
            str: `random_shift`.

        """
        return 'random_shift'

    def get_params(self):
        """ Returns augmentation parameters.

        Returns:
            dict: augmentation parameters.

        """
        return {'shift_size': self.shift_size}


class Cutout(Augmentation):
    """ Cutout augmentation.

    Args:
        probability (float): probability to cutout.

    Attributes:
        probability (float): probability to cutout.

    """
    def __init__(self, probability=0.5):
        self.probability = probability
        self._operation = aug.RandomErasing(p=probability)

    def transform(self, x):
        """ Returns observation performed Cutout.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        return self._operation(x)

    def get_type(self):
        """ Returns augmentation type.

        Returns:
            str: `cutout`.

        """
        return 'cutout'

    def get_params(self):
        """ Returns augmentation parameters.

        Returns:
            dict: augmentation parameters.

        """
        return {'probability': self.probability}


class HorizontalFlip(Augmentation):
    """ Horizontal flip augmentation.

    Args:
        probability (float): probability to flip horizontally.

    Attributes:
        probability (float): probability to flip horizontally.

    """
    def __init__(self, probability=0.1):
        self.probability = probability
        self._operation = aug.RandomHorizontalFlip(p=probability)

    def transform(self, x):
        """ Returns horizontally flipped image.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        return self._operation(x)

    def get_type(self):
        """ Returns augmentation type.

        Returns:
            str: `horizontal_flip`.

        """
        return 'horizontal_flip'

    def get_params(self):
        """ Returns augmentation parameters.

        Returns:
            dict: augmentation parameters.

        """
        return {'probability': self.probability}


class VerticalFlip(Augmentation):
    """ Vertical flip augmentation.

    Args:
        probability (float): probability to flip vertically.

    Attributes:
        probability (float): probability to flip vertically.

    """
    def __init__(self, probability=0.1):
        self.probability = probability
        self._operation = aug.RandomVerticalFlip(p=probability)

    def transform(self, x):
        """ Returns vertically flipped image.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        return self._operation(x)

    def get_type(self):
        """ Returns augmentation type.

        Returns:
            str: `vertical_flip`.

        """
        return 'vertical_flip'

    def get_params(self):
        """ Returns augmentation parameters.

        Returns:
            dict: augmentation parameters.

        """
        return {'probability': self.probability}


class RandomRotation(Augmentation):
    """ Random rotation augmentation.

    Args:
        degree (float): range of degrees to rotate image.

    Attributes:
        degree (float): range of degrees to rotate image.

    """
    def __init__(self, degree=5.0):
        self.degree = degree
        self._operation = aug.RandomRotation(degrees=degree)

    def transform(self, x):
        """ Returns rotated image.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        return self._operation(x)

    def get_type(self):
        """ Returns augmentation type.

        Returns:
            str: `random_rotation`.

        """
        return 'random_rotation'

    def get_params(self):
        """ Returns augmentation parameters.

        Returns:
            dict: augmentation parameters.

        """
        return {'degree': self.degree}


class Intensity(Augmentation):
    """ Intensity augmentation.

    .. math::

        x' = x + n

    where :math:`n \\sim N(0, scale)`.

    Args:
        scale (float): scale of multiplier.

    Attributes:
        scale (float): scale of multiplier.

    """
    def __init__(self, scale=0.1):
        self.scale = scale

    def transform(self, x):
        """ Returns multiplied image.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        r = torch.randn(x.size(0), 1, 1, 1, device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise

    def get_type(self):
        """ Returns augmentation type.

        Returns:
            str: `intensity`.

        """
        return 'intensity'

    def get_params(self):
        """ Returns augmentation parameters.

        Returns:
            dict: augmentation parameters.

        """
        return {'scale': self.scale}
