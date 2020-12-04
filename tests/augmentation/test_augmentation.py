import numpy as np
import pytest

from unittest.mock import Mock
from d3rlpy.augmentation import AugmentationPipeline, create_augmentation
from d3rlpy.augmentation.base import Augmentation
from d3rlpy.augmentation.image import RandomShift
from d3rlpy.augmentation.image import Cutout
from d3rlpy.augmentation.image import HorizontalFlip
from d3rlpy.augmentation.image import VerticalFlip
from d3rlpy.augmentation.image import RandomRotation
from d3rlpy.augmentation.image import Intensity
from d3rlpy.augmentation.image import ColorJitter
from d3rlpy.augmentation.vector import SingleAmplitudeScaling
from d3rlpy.augmentation.vector import MultipleAmplitudeScaling


class DummyAugmentation(Augmentation):
    def transform(self, x):
        return x

    def get_type(self):
        return 'dummy'

    def get_params(self):
        return {'param': 0.1}


@pytest.mark.parametrize('aug_type', [
    'random_shift', 'cutout', 'horizontal_flip', 'vertical_flip',
    'random_rotation', 'intensity', 'color_jitter', 'single_amplitude_scaling',
    'multiple_amplitude_scaling'
])
def test_create_augmentation(aug_type):
    augmentation = create_augmentation(aug_type)
    if aug_type == 'random_shift':
        assert isinstance(augmentation, RandomShift)
    elif aug_type == 'cutout':
        assert isinstance(augmentation, Cutout)
    elif aug_type == 'horizontal_flip':
        assert isinstance(augmentation, HorizontalFlip)
    elif aug_type == 'vertical_flip':
        assert isinstance(augmentation, VerticalFlip)
    elif aug_type == 'random_rotation':
        assert isinstance(augmentation, RandomRotation)
    elif aug_type == 'intensity':
        assert isinstance(augmentation, Intensity)
    elif aug_type == 'color_jitter':
        assert isinstance(augmentation, ColorJitter)
    elif aug_type == 'single_amplitude_scaling':
        assert isinstance(augmentation, SingleAmplitudeScaling)
    elif aug_type == 'multiple_amplitude_scaling':
        assert isinstance(augmentation, MultipleAmplitudeScaling)


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('observation_shape', [(4, 84, 84)])
def test_augmentation_pipeline(batch_size, observation_shape):
    aug1 = DummyAugmentation()
    aug1.transform = Mock(side_effect=lambda x: x + 0.1)

    aug2 = DummyAugmentation()
    aug2.transform = Mock(side_effect=lambda x: x + 0.2)

    aug = AugmentationPipeline([aug1])
    aug.append(aug2)

    x = np.random.random((batch_size, *observation_shape))
    y = aug.transform(x)

    aug1.transform.assert_called_once()
    aug2.transform.assert_called_once()
    assert np.allclose(y, x + 0.3)

    assert aug.get_augmentation_types() == ['dummy', 'dummy']
    assert aug.get_augmentation_params() == [{'param': 0.1}, {'param': 0.1}]
