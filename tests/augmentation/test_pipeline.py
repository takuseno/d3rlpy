import torch
import pytest

from unittest.mock import Mock
from d3rlpy.augmentation.pipeline import DrQPipeline
from d3rlpy.augmentation.base import Augmentation


class DummyAugmentation(Augmentation):
    def transform(self, x):
        return x

    def get_type(self):
        return "dummy"

    def get_params(self):
        return {"param": 0.1}


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(4, 84, 84)])
@pytest.mark.parametrize("n_mean", [1, 2])
def test_drq_pipeline(batch_size, observation_shape, n_mean):
    aug1 = DummyAugmentation()
    aug1.transform = Mock(side_effect=lambda x: x + 0.1)

    aug2 = DummyAugmentation()
    aug2.transform = Mock(side_effect=lambda x: x + 0.2)

    aug = DrQPipeline([aug1], n_mean=n_mean)
    aug.append(aug2)

    x = torch.rand((batch_size, *observation_shape))
    y = aug.transform(x)

    aug1.transform.assert_called_once()
    aug2.transform.assert_called_once()
    assert torch.allclose(y, x + 0.3)

    assert aug.get_augmentation_types() == ["dummy", "dummy"]
    assert aug.get_augmentation_params() == [{"param": 0.1}, {"param": 0.1}]
    assert aug.get_params() == {"n_mean": n_mean}

    def func(x):
        return x

    x = torch.rand((batch_size, *observation_shape))
    y = aug.process(func, {"x": x}, targets=["x"])
    assert torch.allclose(y, x + 0.3)
    assert aug1.transform.call_count == 1 + n_mean
    assert aug2.transform.call_count == 1 + n_mean
