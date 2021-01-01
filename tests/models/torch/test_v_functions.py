import pytest
import torch
import torch.nn.functional as F

from d3rlpy.models.torch.v_functions import ValueFunction
from .model_test import check_parameter_updates, DummyEncoder


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("batch_size", [32])
def test_value_function(feature_size, batch_size):
    encoder = DummyEncoder(feature_size)
    v_func = ValueFunction(encoder)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = v_func(x)
    assert y.shape == (batch_size, 1)

    # check compute_error
    returns = torch.rand(batch_size, 1)
    loss = v_func.compute_error(x, returns)
    assert torch.allclose(loss, F.mse_loss(y, returns))

    # check layer connections
    check_parameter_updates(
        v_func,
        (
            x,
            returns,
        ),
    )
