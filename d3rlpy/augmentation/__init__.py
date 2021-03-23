from typing import Any, Dict, Type

from .base import Augmentation
from .image import (
    ColorJitter,
    Cutout,
    HorizontalFlip,
    Intensity,
    RandomRotation,
    RandomShift,
    VerticalFlip,
)
from .pipeline import AugmentationPipeline, DrQPipeline
from .vector import MultipleAmplitudeScaling, SingleAmplitudeScaling

__all__ = [
    "Augmentation",
    "AugmentationPipeline",
    "DrQPipeline",
    "RandomShift",
    "Cutout",
    "HorizontalFlip",
    "VerticalFlip",
    "RandomRotation",
    "Intensity",
    "ColorJitter",
    "SingleAmplitudeScaling",
    "MultipleAmplitudeScaling",
    "register_augmentation",
    "create_augmentation",
]

AUGMENTATION_LIST: Dict[str, Type[Augmentation]] = {}


def register_augmentation(cls: Type[Augmentation]) -> None:
    """Registers augmentation class.

    Args:
        cls (type): augmentation class inheriting ``Augmentation``.

    """
    is_registered = cls.TYPE in AUGMENTATION_LIST
    assert not is_registered, "%s seems to be already registered" % cls.TYPE
    AUGMENTATION_LIST[cls.TYPE] = cls


def create_augmentation(name: str, **kwargs: Dict[str, Any]) -> Augmentation:
    """Returns registered encoder factory object.

    Args:
        name (str): regsitered encoder factory type name.
        kwargs (any): encoder arguments.

    Returns:
        d3rlpy.encoders.EncoderFactory: encoder factory object.

    """
    assert name in AUGMENTATION_LIST, "%s seems not to be registered." % name
    augmentation = AUGMENTATION_LIST[name](**kwargs)  # type: ignore
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
