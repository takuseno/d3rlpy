from .base import Augmentation
from .image import RandomShift
from .image import Cutout
from .image import HorizontalFlip
from .image import VerticalFlip
from .image import RandomRotation
from .image import Intensity
from .vector import SingleAmplitudeScaling
from .vector import MultipleAmplitudeScaling


def create_augmentation(augmentation_type, **kwargs):
    if augmentation_type == 'random_shift':
        return RandomShift(**kwargs)
    elif augmentation_type == 'cutout':
        return Cutout(**kwargs)
    elif augmentation_type == 'horizontal_flip':
        return HorizontalFlip(**kwargs)
    elif augmentation_type == 'vertical_flip':
        return VerticalFlip(**kwargs)
    elif augmentation_type == 'random_rotation':
        return RandomRotation(**kwargs)
    elif augmentation_type == 'intensity':
        return Intensity(**kwargs)
    elif augmentation_type == 'single_amplitude_scaling':
        return SingleAmplitudeScaling(**kwargs)
    elif augmentation_type == 'multiple_amplitude_scaling':
        return MultipleAmplitudeScaling(**kwargs)
    raise ValueError('invalid augmentation_type.')


class AugmentationPipeline(Augmentation):
    """ Augmentation pipeline.

    Args:
        augmentations (list(d3rlpy.augmentation.base.Augmentation)):
            list of augmentations.

    Attributes:
        augmentations (list(d3rlpy.augmentation.base.Augmentation)):
            list of augmentations.

    """
    def __init__(self, augmentations=[]):
        self.augmentations = list(augmentations)

    def append(self, augmentation):
        """ Append augmentation to pipeline.

        Args:
            augmentation (d3rlpy.augmentation.base.Augmentation): augmentation.

        """
        self.augmentations.append(augmentation)

    def transform(self, x):
        """ Returns observation processed by all augmentations.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        if not self.augmentations:
            return x

        for augmentation in self.augmentations:
            x = augmentation.transform(x)

        return x

    def get_type(self):
        """ Returns augmentation types.

        Returns:
            list(str): list of augmentation types.

        """
        return [aug.get_type() for aug in self.augmentations]

    def get_params(self):
        """ Returns augmentation parameters.

        Returns:
            list(dict): list of augmentation parameters.

        """
        return [aug.get_params() for aug in self.augmentations]
