import pytest
import torch
import torch.nn.functional as F

from d3rlpy.models.torch.v_functions import create_value_function
from d3rlpy.models.torch.v_functions import ValueFunction
from tests import create_encoder_factory
from .model_test import check_parameter_updates, DummyEncoder


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('use_encoder_factory', [False, True])
@pytest.mark.parametrize('batch_size', [32])
def test_create_value_function(observation_shape, use_encoder_factory,
                               batch_size):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)

    v_func = create_value_function(observation_shape, encoder_factory)

    assert isinstance(v_func, ValueFunction)

    x = torch.rand((batch_size, ) + observation_shape)
    y = v_func(x)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('batch_size', [32])
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
    check_parameter_updates(v_func, (
        x,
        returns,
    ))
