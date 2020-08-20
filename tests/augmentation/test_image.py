import pytest
import torch

from d3rlpy.augmentation import create_augmentation
from d3rlpy.augmentation.image import RandomShift
from d3rlpy.augmentation.image import Cutout
from d3rlpy.augmentation.image import HorizontalFlip
from d3rlpy.augmentation.image import VerticalFlip
from d3rlpy.augmentation.image import RandomRotation
from d3rlpy.augmentation.image import Intensity
from d3rlpy.augmentation.image import ColorJitter


@pytest.mark.parametrize('augmentation_type', [
    'random_shift', 'cutout', 'horizontal_flip', 'vertical_flip',
    'random_rotation', 'intensity', 'color_jitter'
])
def test_create_augmentation(augmentation_type):
    augmentation = create_augmentation(augmentation_type)
    if augmentation_type == 'random_shift':
        assert isinstance(augmentation, RandomShift)
    elif augmentation_type == 'cutout':
        assert isinstance(augmentation, Cutout)
    elif augmentation_type == 'horizontal_flip':
        assert isinstance(augmentation, HorizontalFlip)
    elif augmentation_type == 'vertical_flip':
        assert isinstance(augmentation, VerticalFlip)
    elif augmentation_type == 'random_rotation':
        assert isinstance(augmentation, RandomRotation)
    elif augmentation_type == 'intensity':
        assert isinstance(augmentation, Intensity)
    elif augmentation_type == 'color_jitter':
        assert isinstance(augmentation, ColorJitter)


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('observation_shape', [(4, 84, 84)])
@pytest.mark.parametrize('shift_size', [4])
def test_random_shift(batch_size, observation_shape, shift_size):
    augmentation = RandomShift(shift_size)

    x = torch.rand(batch_size, *observation_shape)

    y = augmentation.transform(x)

    assert not torch.all(x == y)

    assert augmentation.get_type() == 'random_shift'
    assert augmentation.get_params()['shift_size'] == shift_size


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('observation_shape', [(4, 4, 4)])
@pytest.mark.parametrize('probability', [1.0])
def test_cutout(batch_size, observation_shape, probability):
    augmentation = Cutout(probability)

    x = torch.rand(batch_size, *observation_shape)

    y = augmentation.transform(x)

    assert not torch.all(x == y)

    assert augmentation.get_type() == 'cutout'
    assert augmentation.get_params()['probability'] == probability


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('observation_shape', [(4, 4, 4)])
@pytest.mark.parametrize('probability', [1.0])
def test_horizontal_flip(batch_size, observation_shape, probability):
    augmentation = HorizontalFlip(probability)

    x = torch.rand(batch_size, *observation_shape)

    y = augmentation.transform(x)

    assert not torch.all(x == y)

    assert augmentation.get_type() == 'horizontal_flip'
    assert augmentation.get_params()['probability'] == probability


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('observation_shape', [(4, 4, 4)])
@pytest.mark.parametrize('probability', [1.0])
def test_vertical_flip(batch_size, observation_shape, probability):
    augmentation = VerticalFlip(probability)

    x = torch.rand(batch_size, *observation_shape)

    y = augmentation.transform(x)

    assert not torch.all(x == y)

    assert augmentation.get_type() == 'vertical_flip'
    assert augmentation.get_params()['probability'] == probability


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('observation_shape', [(4, 4, 4)])
@pytest.mark.parametrize('degree', [5.0])
def test_random_rotation(batch_size, observation_shape, degree):
    augmentation = RandomRotation(degree)

    x = torch.rand(batch_size, *observation_shape)

    y = augmentation.transform(x)

    assert not torch.all(x == y)

    assert augmentation.get_type() == 'random_rotation'
    assert augmentation.get_params()['degree'] == degree


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('observation_shape', [(4, 4, 4)])
@pytest.mark.parametrize('scale', [0.1])
def test_intensity(batch_size, observation_shape, scale):
    augmentation = Intensity(scale)

    x = torch.rand(batch_size, *observation_shape)

    y = augmentation.transform(x)

    assert not torch.all(x == y)

    assert augmentation.get_type() == 'intensity'
    assert augmentation.get_params()['scale'] == scale


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('observation_shape', [(3, 4, 4), (9, 4, 4)])
@pytest.mark.parametrize('hue', [0.4])
@pytest.mark.parametrize('saturation', [0.4])
@pytest.mark.parametrize('brightness', [0.4])
@pytest.mark.parametrize('contrast', [0.4])
def test_color_jitter(batch_size, observation_shape, hue, saturation,
                      brightness, contrast):
    augmentation = ColorJitter(brightness, contrast, saturation, hue)

    x = torch.rand(batch_size, *observation_shape)

    y = augmentation.transform(x)

    assert not torch.all(x == y)

    assert augmentation.get_type() == 'color_jitter'
    assert augmentation.get_params()['hue'] == hue
    assert augmentation.get_params()['saturation'] == saturation
    assert augmentation.get_params()['brightness'] == brightness
    assert augmentation.get_params()['contrast'] == contrast
