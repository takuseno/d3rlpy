import numpy as np
import pytest

from unittest.mock import Mock
from d3rlpy.augmentation import AugmentationPipeline
from d3rlpy.augmentation.base import Augmentation


class DummyAugmentation(Augmentation):
    def transform(self, x):
        return x

    def get_type(self):
        return 'dummy'

    def get_params(self):
        return {'param': 0.1}


@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('observation_shape', [(4, 84, 84)])
def test_augmentation_pipeline(batch_size, observation_shape):
    aug1 = DummyAugmentation()
    aug1.transform = Mock(side_effect=lambda x: x + 0.1)

    aug2 = DummyAugmentation()
    aug2.transform = Mock(side_effect=lambda x: x + 0.2)

    aug = AugmentationPipeline()
    aug.append(aug1)
    aug.append(aug2)

    x = np.random.random((batch_size, *observation_shape))
    y = aug.transform(x)

    aug1.transform.assert_called_once()
    aug2.transform.assert_called_once()
    assert np.allclose(y, x + 0.3)

    assert aug.get_augmentation_types() == ['dummy', 'dummy']
    assert aug.get_augmentation_params() == [{'param': 0.1}, {'param': 0.1}]
