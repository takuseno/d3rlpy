import numpy as np
import torch
import torch.nn as nn
import kornia.augmentation as aug

from kornia.color.hsv import hsv_to_rgb, rgb_to_hsv
from .base import Augmentation


class RandomShift(Augmentation):
    """ Random shift augmentation.

    References:
        * `Kostrikov et al., Image Augmentation Is All You Need: Regularizing
          Deep Reinforcement Learning from Pixels.
          <https://arxiv.org/abs/2004.13649>`_

    Args:
        shift_size (int): size to shift image.

    Attributes:
        shift_size (int): size to shift image.

    """

    TYPE = 'random_shift'

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

    def get_params(self, deep=False):
        """ Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: augmentation parameters.

        """
        return {'shift_size': self.shift_size}


class Cutout(Augmentation):
    """ Cutout augmentation.

    References:
        * `Kostrikov et al., Image Augmentation Is All You Need: Regularizing
          Deep Reinforcement Learning from Pixels.
          <https://arxiv.org/abs/2004.13649>`_

    Args:
        probability (float): probability to cutout.

    Attributes:
        probability (float): probability to cutout.

    """

    TYPE = 'cutout'

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

    def get_params(self, deep=False):
        """ Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: augmentation parameters.

        """
        return {'probability': self.probability}


class HorizontalFlip(Augmentation):
    """ Horizontal flip augmentation.

    References:
        * `Kostrikov et al., Image Augmentation Is All You Need: Regularizing
          Deep Reinforcement Learning from Pixels.
          <https://arxiv.org/abs/2004.13649>`_

    Args:
        probability (float): probability to flip horizontally.

    Attributes:
        probability (float): probability to flip horizontally.

    """

    TYPE = 'horizontal_flip'

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

    def get_params(self, deep=False):
        """ Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: augmentation parameters.

        """
        return {'probability': self.probability}


class VerticalFlip(Augmentation):
    """ Vertical flip augmentation.

    References:
        * `Kostrikov et al., Image Augmentation Is All You Need: Regularizing
          Deep Reinforcement Learning from Pixels.
          <https://arxiv.org/abs/2004.13649>`_

    Args:
        probability (float): probability to flip vertically.

    Attributes:
        probability (float): probability to flip vertically.

    """

    TYPE = 'vertical_flip'

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

    def get_params(self, deep=False):
        """ Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: augmentation parameters.

        """
        return {'probability': self.probability}


class RandomRotation(Augmentation):
    """ Random rotation augmentation.

    References:
        * `Kostrikov et al., Image Augmentation Is All You Need: Regularizing
          Deep Reinforcement Learning from Pixels.
          <https://arxiv.org/abs/2004.13649>`_

    Args:
        degree (float): range of degrees to rotate image.

    Attributes:
        degree (float): range of degrees to rotate image.

    """

    TYPE = 'random_rotation'

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

    def get_params(self, deep=False):
        """ Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: augmentation parameters.

        """
        return {'degree': self.degree}


class Intensity(Augmentation):
    r""" Intensity augmentation.

    .. math::

        x' = x + n

    where :math:`n \sim N(0, scale)`.

    References:
        * `Kostrikov et al., Image Augmentation Is All You Need: Regularizing
          Deep Reinforcement Learning from Pixels.
          <https://arxiv.org/abs/2004.13649>`_

    Args:
        scale (float): scale of multiplier.

    Attributes:
        scale (float): scale of multiplier.

    """

    TYPE = 'intensity'

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

    def get_params(self, deep=False):
        """ Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: augmentation parameters.

        """
        return {'scale': self.scale}


class ColorJitter(Augmentation):
    """ Color Jitter augmentation.

    This augmentation modifies the given images in the HSV channel spaces
    as well as a contrast change.
    This augmentation will be useful with the real world images.

    References:
        * `Laskin et al., Reinforcement Learning with Augmented Data.
          <https://arxiv.org/abs/2004.14990>`_

    Args:
        brightness (tuple): brightness scale range.
        contrast (tuple): contrast scale range.
        saturation (tuple): saturation scale range.
        hue (tuple): hue scale range.

    Attributes:
        brightness (tuple): brightness scale range.
        contrast (tuple): contrast scale range.
        saturation (tuple): saturation scale range.
        hue (tuple): hue scale range.

    """

    TYPE = 'color_jitter'

    def __init__(self,
                 brightness=(0.6, 1.4),
                 contrast=(0.6, 1.4),
                 saturation=(0.6, 1.4),
                 hue=(-0.5, 0.5)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def transform(self, x):
        """ Returns jittered images.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        # check if channel can be devided by three
        if x.shape[1] % 3 > 0:
            raise ValueError('color jitter is used with stacked RGB images')

        # flag for transformation order
        is_transforming_rgb_first = np.random.randint(2)

        # (batch, C, W, H) -> (batch, stack, 3, W, H)
        flat_rgb = x.view(x.shape[0], -1, 3, x.shape[2], x.shape[3])

        if is_transforming_rgb_first:
            # transform contrast
            flat_rgb = self._transform_contrast(flat_rgb)

        # (batch, stack, 3, W, H) -> (batch * stack, 3, W, H)
        rgb_images = flat_rgb.view(-1, 3, x.shape[2], x.shape[3])

        # RGB -> HSV
        hsv_images = rgb_to_hsv(rgb_images)

        # apply same transformation within the stacked images
        # (batch * stack, 3, W, H) -> (batch, stack, 3, W, H)
        flat_hsv = hsv_images.view(x.shape[0], -1, 3, x.shape[2], x.shape[3])

        # transform hue
        flat_hsv = self._transform_hue(flat_hsv)
        # transform saturate
        flat_hsv = self._transform_saturate(flat_hsv)
        # transform brightness
        flat_hsv = self._transform_brightness(flat_hsv)

        # (batch, stack, 3, W, H) -> (batch * stack, 3, W, H)
        hsv_images = flat_hsv.view(-1, 3, x.shape[2], x.shape[3])

        # HSV -> RGB
        rgb_images = hsv_to_rgb(hsv_images)

        # (batch * stack, 3, W, H) -> (batch, stack, 3, W, H)
        flat_rgb = rgb_images.view(x.shape[0], -1, 3, x.shape[2], x.shape[3])

        if not is_transforming_rgb_first:
            # transform contrast
            flat_rgb = self._transform_contrast(flat_rgb)

        return flat_rgb.view(*x.shape)

    def _transform_hue(self, hsv):
        scale = torch.empty(hsv.shape[0], 1, 1, 1, device=hsv.device)
        scale = scale.uniform_(*self.hue) * 255.0 / 360.0
        hsv[:, :, 0, :, :] = (hsv[:, :, 0, :, :] + scale) % 1
        return hsv

    def _transform_saturate(self, hsv):
        scale = torch.empty(hsv.shape[0], 1, 1, 1, device=hsv.device)
        scale.uniform_(*self.saturation)
        hsv[:, :, 1, :, :] *= scale
        return hsv.clamp(0, 1)

    def _transform_brightness(self, hsv):
        scale = torch.empty(hsv.shape[0], 1, 1, 1, device=hsv.device)
        scale.uniform_(*self.brightness)
        hsv[:, :, 2, :, :] *= scale
        return hsv.clamp(0, 1)

    def _transform_contrast(self, rgb):
        scale = torch.empty(rgb.shape[0], 1, 1, 1, 1, device=rgb.device)
        scale.uniform_(*self.contrast)
        means = rgb.mean(dim=(3, 4), keepdims=True)
        return ((rgb - means) * (scale + means)).clamp(0, 1)

    def get_params(self, deep=False):
        """ Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            dict: augmentation parameters.

        """
        return {
            'brightness': self.brightness,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'hue': self.hue
        }
