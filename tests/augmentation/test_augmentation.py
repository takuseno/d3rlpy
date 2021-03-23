import numpy as np
import pytest

from d3rlpy.augmentation import create_augmentation
from d3rlpy.augmentation.image import (
    ColorJitter,
    Cutout,
    HorizontalFlip,
    Intensity,
    RandomRotation,
    RandomShift,
    VerticalFlip,
)
from d3rlpy.augmentation.vector import (
    MultipleAmplitudeScaling,
    SingleAmplitudeScaling,
)


@pytest.mark.parametrize(
    "aug_type",
    [
        "random_shift",
        "cutout",
        "horizontal_flip",
        "vertical_flip",
        "random_rotation",
        "intensity",
        "color_jitter",
        "single_amplitude_scaling",
        "multiple_amplitude_scaling",
    ],
)
def test_create_augmentation(aug_type):
    augmentation = create_augmentation(aug_type)
    if aug_type == "random_shift":
        assert isinstance(augmentation, RandomShift)
    elif aug_type == "cutout":
        assert isinstance(augmentation, Cutout)
    elif aug_type == "horizontal_flip":
        assert isinstance(augmentation, HorizontalFlip)
    elif aug_type == "vertical_flip":
        assert isinstance(augmentation, VerticalFlip)
    elif aug_type == "random_rotation":
        assert isinstance(augmentation, RandomRotation)
    elif aug_type == "intensity":
        assert isinstance(augmentation, Intensity)
    elif aug_type == "color_jitter":
        assert isinstance(augmentation, ColorJitter)
    elif aug_type == "single_amplitude_scaling":
        assert isinstance(augmentation, SingleAmplitudeScaling)
    elif aug_type == "multiple_amplitude_scaling":
        assert isinstance(augmentation, MultipleAmplitudeScaling)
