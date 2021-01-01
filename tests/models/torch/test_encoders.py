import pytest
import torch

from d3rlpy.models.torch.encoders import PixelEncoder, PixelEncoderWithAction
from d3rlpy.models.torch.encoders import VectorEncoder, VectorEncoderWithAction
from .model_test import check_parameter_updates


@pytest.mark.parametrize("shapes", [((4, 84, 84), 3136)])
@pytest.mark.parametrize("filters", [[(32, 8, 4), (64, 4, 2), (64, 3, 1)]])
@pytest.mark.parametrize("feature_size", [512])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("activation", [torch.relu])
def test_pixel_encoder(
    shapes, filters, feature_size, batch_size, use_batch_norm, activation
):
    observation_shape, linear_input_size = shapes

    encoder = PixelEncoder(
        observation_shape, filters, feature_size, use_batch_norm, activation
    )
    x = torch.rand((batch_size,) + observation_shape)
    y = encoder(x)

    # check output shape
    assert encoder._get_linear_input_size() == linear_input_size
    assert y.shape == (batch_size, feature_size)

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x)
    assert not use_batch_norm == torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x,))


@pytest.mark.parametrize("shapes", [((4, 84, 84), 3136)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("filters", [[(32, 8, 4), (64, 4, 2), (64, 3, 1)]])
@pytest.mark.parametrize("feature_size", [512])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("discrete_action", [False, True])
@pytest.mark.parametrize("activation", [torch.relu])
def test_pixel_encoder_with_action(
    shapes,
    action_size,
    filters,
    feature_size,
    batch_size,
    use_batch_norm,
    discrete_action,
    activation,
):
    observation_shape, linear_input_size = shapes

    encoder = PixelEncoderWithAction(
        observation_shape,
        action_size,
        filters,
        feature_size,
        use_batch_norm,
        discrete_action,
        activation,
    )
    x = torch.rand((batch_size,) + observation_shape)
    if discrete_action:
        action = torch.randint(0, action_size, size=(batch_size, 1))
    else:
        action = torch.rand((batch_size, action_size))
    y = encoder(x, action)

    # check output shape
    assert encoder._get_linear_input_size() == linear_input_size + action_size
    assert y.shape == (batch_size, feature_size)

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x, action)
    assert not use_batch_norm == torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x, action))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("hidden_units", [[256, 256]])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("use_dense", [False, True])
@pytest.mark.parametrize("activation", [torch.relu])
def test_vector_encoder(
    observation_shape,
    hidden_units,
    batch_size,
    use_batch_norm,
    use_dense,
    activation,
):
    encoder = VectorEncoder(
        observation_shape, hidden_units, use_batch_norm, use_dense, activation
    )

    x = torch.rand((batch_size,) + observation_shape)
    y = encoder(x)

    # check output shape
    assert encoder.get_feature_size() == hidden_units[-1]
    assert y.shape == (batch_size, hidden_units[-1])

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x)
    assert not use_batch_norm == torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x,))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("hidden_units", [[256, 256]])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("use_dense", [False, True])
@pytest.mark.parametrize("discrete_action", [False, True])
@pytest.mark.parametrize("activation", [torch.relu])
def test_vector_encoder(
    observation_shape,
    action_size,
    hidden_units,
    batch_size,
    use_batch_norm,
    use_dense,
    discrete_action,
    activation,
):
    encoder = VectorEncoderWithAction(
        observation_shape,
        action_size,
        hidden_units,
        use_batch_norm,
        use_dense,
        discrete_action,
        activation,
    )

    x = torch.rand((batch_size,) + observation_shape)
    if discrete_action:
        action = torch.randint(0, action_size, size=(batch_size, 1))
    else:
        action = torch.rand((batch_size, action_size))
    y = encoder(x, action)

    # check output shape
    assert encoder.get_feature_size() == hidden_units[-1]
    assert y.shape == (batch_size, hidden_units[-1])

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x, action)
    assert not use_batch_norm == torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x, action))
