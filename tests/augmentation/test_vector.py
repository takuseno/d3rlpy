import pytest
import torch

from d3rlpy.augmentation import create_augmentation
from d3rlpy.augmentation.vector import SingleAmplitudeScaling
from d3rlpy.augmentation.vector import MultipleAmplitudeScaling


@pytest.mark.parametrize(
    "augmentation_type",
    ["single_amplitude_scaling", "multiple_amplitude_scaling"],
)
def test_create_augmentation(augmentation_type):
    augmentation = create_augmentation(augmentation_type)
    if augmentation_type == "single_amplitude_scaling":
        assert isinstance(augmentation, SingleAmplitudeScaling)
    elif augmentation_type == "multiple_amplitude_scaling":
        assert isinstance(augmentation, MultipleAmplitudeScaling)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("minimum", [0.8])
@pytest.mark.parametrize("maximum", [1.2])
def test_single_amplitude_scaling(
    batch_size, observation_shape, minimum, maximum
):
    augmentation = SingleAmplitudeScaling(minimum, maximum)

    x = torch.rand(batch_size, *observation_shape)

    y = augmentation.transform(x)

    assert not torch.all(x == y)

    assert augmentation.get_type() == "single_amplitude_scaling"
    assert augmentation.get_params()["minimum"] == minimum
    assert augmentation.get_params()["maximum"] == maximum


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("minimum", [0.8])
@pytest.mark.parametrize("maximum", [1.2])
def test_multiple_amplitude_scaling(
    batch_size, observation_shape, minimum, maximum
):
    augmentation = MultipleAmplitudeScaling(minimum, maximum)

    x = torch.rand(batch_size, *observation_shape)

    y = augmentation.transform(x)

    assert not torch.all(x == y)

    assert augmentation.get_type() == "multiple_amplitude_scaling"
    assert augmentation.get_params()["minimum"] == minimum
    assert augmentation.get_params()["maximum"] == maximum
