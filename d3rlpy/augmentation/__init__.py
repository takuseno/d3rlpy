from .base import Augmentation
from .image import RandomShift
from .image import Cutout
from .image import HorizontalFlip
from .image import VerticalFlip
from .image import RandomRotation
from .image import Intensity
from .image import ColorJitter
from .vector import SingleAmplitudeScaling
from .vector import MultipleAmplitudeScaling

AUGMENTATION_LIST = {}


def register_augmentation(cls):
    """ Registers augmentation class.

    Args:
        cls (type): augmentation class inheriting ``Augmentation``.

    """
    is_registered = cls.TYPE in AUGMENTATION_LIST
    assert not is_registered, '%s seems to be already registered' % cls.TYPE
    AUGMENTATION_LIST[cls.TYPE] = cls


def create_augmentation(name, **kwargs):
    """ Returns registered encoder factory object.

    Args:
        name (str): regsitered encoder factory type name.
        kwargs (any): encoder arguments.

    Returns:
        d3rlpy.encoders.EncoderFactory: encoder factory object.

    """
    assert name in AUGMENTATION_LIST, '%s seems not to be registered.' % name
    augmentation = AUGMENTATION_LIST[name](**kwargs)
    assert isinstance(augmentation, Augmentation)
    return augmentation


register_augmentation(RandomShift)
register_augmentation(Cutout)
register_augmentation(HorizontalFlip)
register_augmentation(VerticalFlip)
register_augmentation(RandomRotation)
register_augmentation(Intensity)
register_augmentation(ColorJitter)
register_augmentation(SingleAmplitudeScaling)
register_augmentation(MultipleAmplitudeScaling)


class AugmentationPipeline:
    """ Augmentation pipeline.

    Args:
        augmentations (list(d3rlpy.augmentation.base.Augmentation or str)):
            list of augmentations or augmentation types.

    Attributes:
        augmentations (list(d3rlpy.augmentation.base.Augmentation)):
            list of augmentations.

    """
    def __init__(self, augmentations=None):
        if augmentations is None:
            self.augmentations = []
        else:
            self.augmentations = augmentations

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

    def get_augmentation_types(self):
        """ Returns augmentation types.

        Returns:
            list(str): list of augmentation types.

        """
        return [aug.get_type() for aug in self.augmentations]

    def get_augmentation_params(self):
        """ Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            list(dict): list of augmentation parameters.

        """
        return [aug.get_params() for aug in self.augmentations]
