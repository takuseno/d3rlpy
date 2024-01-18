# pylint: disable=protected-access
from typing import List, Optional, Sequence, Tuple

import pytest
import torch

from d3rlpy.models.torch.encoders import (
    PixelEncoder,
    PixelEncoderWithAction,
    VectorEncoder,
    VectorEncoderWithAction,
)

from .model_test import check_parameter_updates


@pytest.mark.parametrize("shapes", [((4, 84, 84), 3136)])
@pytest.mark.parametrize("filters", [[(32, 8, 4), (64, 4, 2), (64, 3, 1)]])
@pytest.mark.parametrize("feature_size", [512])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("dropout_rate", [None, 0.2])
@pytest.mark.parametrize("activation", [torch.nn.ReLU()])
@pytest.mark.parametrize("last_activation", [None, torch.nn.ReLU()])
def test_pixel_encoder(
    shapes: Tuple[Sequence[int], int],
    filters: List[List[int]],
    feature_size: int,
    batch_size: int,
    use_batch_norm: bool,
    dropout_rate: Optional[float],
    activation: torch.nn.Module,
    last_activation: Optional[torch.nn.Module],
) -> None:
    observation_shape, _ = shapes

    encoder = PixelEncoder(
        observation_shape=observation_shape,
        filters=filters,
        feature_size=feature_size,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        activation=activation,
        last_activation=last_activation,
    )
    x = torch.rand((batch_size, *observation_shape))
    y = encoder(x)

    # check output shape
    assert y.shape == (batch_size, feature_size)

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x)
    if use_batch_norm or dropout_rate:
        assert not torch.allclose(y, eval_y)
    else:
        assert torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x,))


@pytest.mark.parametrize("shapes", [((4, 84, 84), 3136)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("filters", [[(32, 8, 4), (64, 4, 2), (64, 3, 1)]])
@pytest.mark.parametrize("feature_size", [512])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("dropout_rate", [None, 0.2])
@pytest.mark.parametrize("discrete_action", [False, True])
@pytest.mark.parametrize("activation", [torch.nn.ReLU()])
@pytest.mark.parametrize("last_activation", [None, torch.nn.ReLU()])
def test_pixel_encoder_with_action(
    shapes: Tuple[Sequence[int], int],
    action_size: int,
    filters: List[List[int]],
    feature_size: int,
    batch_size: int,
    use_batch_norm: bool,
    dropout_rate: Optional[float],
    discrete_action: bool,
    activation: torch.nn.Module,
    last_activation: Optional[torch.nn.Module],
) -> None:
    observation_shape, _ = shapes

    encoder = PixelEncoderWithAction(
        observation_shape=observation_shape,
        action_size=action_size,
        filters=filters,
        feature_size=feature_size,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        discrete_action=discrete_action,
        activation=activation,
        last_activation=last_activation,
    )
    x = torch.rand((batch_size, *observation_shape))
    if discrete_action:
        action = torch.randint(0, action_size, size=(batch_size, 1))
    else:
        action = torch.rand((batch_size, action_size))
    y = encoder(x, action)

    # check output shape
    assert y.shape == (batch_size, feature_size)

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x, action)
    if use_batch_norm or dropout_rate:
        assert not torch.allclose(y, eval_y)
    else:
        assert torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x, action))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("hidden_units", [[256, 256]])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("dropout_rate", [None, 0.2])
@pytest.mark.parametrize("activation", [torch.nn.ReLU()])
@pytest.mark.parametrize("last_activation", [None, torch.nn.ReLU()])
def test_vector_encoder(
    observation_shape: Sequence[int],
    hidden_units: Sequence[int],
    batch_size: int,
    use_batch_norm: bool,
    dropout_rate: Optional[float],
    activation: torch.nn.Module,
    last_activation: Optional[torch.nn.Module],
) -> None:
    encoder = VectorEncoder(
        observation_shape=observation_shape,
        hidden_units=hidden_units,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        activation=activation,
        last_activation=last_activation,
    )

    x = torch.rand((batch_size, *observation_shape))
    y = encoder(x)

    # check output shape
    assert y.shape == (batch_size, hidden_units[-1])

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x)
    if use_batch_norm or dropout_rate:
        assert not torch.allclose(y, eval_y)
    else:
        assert torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x,))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("hidden_units", [[256, 256]])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("use_batch_norm", [False, True])
@pytest.mark.parametrize("dropout_rate", [None, 0.2])
@pytest.mark.parametrize("discrete_action", [False, True])
@pytest.mark.parametrize("activation", [torch.nn.ReLU()])
@pytest.mark.parametrize("last_activation", [None, torch.nn.ReLU()])
def test_vector_encoder_with_action(
    observation_shape: Sequence[int],
    action_size: int,
    hidden_units: Sequence[int],
    batch_size: int,
    use_batch_norm: bool,
    dropout_rate: Optional[float],
    discrete_action: bool,
    activation: torch.nn.Module,
    last_activation: Optional[torch.nn.Module],
) -> None:
    encoder = VectorEncoderWithAction(
        observation_shape=observation_shape,
        action_size=action_size,
        hidden_units=hidden_units,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        discrete_action=discrete_action,
        activation=activation,
        last_activation=last_activation,
    )

    x = torch.rand((batch_size, *observation_shape))
    if discrete_action:
        action = torch.randint(0, action_size, size=(batch_size, 1))
    else:
        action = torch.rand((batch_size, action_size))
    y = encoder(x, action)

    # check output shape
    assert y.shape == (batch_size, hidden_units[-1])

    # check use of batch norm
    encoder.eval()
    eval_y = encoder(x, action)
    if use_batch_norm or dropout_rate:
        assert not torch.allclose(y, eval_y)
    else:
        assert torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(encoder, (x, action))
