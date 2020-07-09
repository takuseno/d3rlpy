import pytest
import torch

from d3rlpy.models.torch.heads import create_head
from d3rlpy.models.torch.heads import PixelHead, PixelHeadWithAction
from d3rlpy.models.torch.heads import VectorHead, VectorHeadWithAction
from .model_test import check_parameter_updates


@pytest.mark.parametrize('observation_shape', [(4, 84, 84)])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_head_with_pixel_head(observation_shape, batch_size,
                                     use_batch_norm):
    head = create_head(observation_shape, use_batch_norm=use_batch_norm)
    x = torch.rand((batch_size, ) + observation_shape)
    y = head(x)

    assert head.use_batch_norm == use_batch_norm
    assert isinstance(head, PixelHead)
    assert y.shape == (batch_size, head.feature_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_head_with_pixel_head_with_action(observation_shape,
                                                 action_size, batch_size,
                                                 use_batch_norm):
    head = create_head(observation_shape,
                       action_size,
                       use_batch_norm=use_batch_norm)
    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand((batch_size, action_size))
    y = head(x, action)

    assert head.use_batch_norm == use_batch_norm
    assert isinstance(head, PixelHeadWithAction)
    assert y.shape == (batch_size, head.feature_size)


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_head_with_vector_head(observation_shape, batch_size,
                                      use_batch_norm):
    head = create_head(observation_shape, use_batch_norm=use_batch_norm)
    x = torch.rand((batch_size, ) + observation_shape)
    y = head(x)

    assert head.use_batch_norm == use_batch_norm
    assert isinstance(head, VectorHead)
    assert y.shape == (batch_size, head.feature_size)


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_head_with_vector_head_with_action(observation_shape,
                                                  action_size, batch_size,
                                                  use_batch_norm):
    head = create_head(observation_shape,
                       action_size,
                       use_batch_norm=use_batch_norm)
    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand((batch_size, action_size))
    y = head(x, action)

    assert head.use_batch_norm == use_batch_norm
    assert isinstance(head, VectorHead)
    assert y.shape == (batch_size, head.feature_size)


@pytest.mark.parametrize('shapes', [((4, 84, 84), 3136)])
@pytest.mark.parametrize('filters', [[(32, 8, 4), (64, 4, 2), (64, 3, 1)]])
@pytest.mark.parametrize('feature_size', [512])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_pixel_head(shapes, filters, feature_size, batch_size, use_batch_norm):
    observation_shape, linear_input_size = shapes

    head = PixelHead(observation_shape, filters, feature_size, use_batch_norm)
    x = torch.rand((batch_size, ) + observation_shape)
    y = head(x)

    # check output shape
    assert head._get_linear_input_size() == linear_input_size
    assert y.shape == (batch_size, feature_size)

    # check use of batch norm
    head.eval()
    eval_y = head(x)
    assert not use_batch_norm == torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(head, (x, ))


@pytest.mark.parametrize('shapes', [((4, 84, 84), 3136)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('filters', [[(32, 8, 4), (64, 4, 2), (64, 3, 1)]])
@pytest.mark.parametrize('feature_size', [512])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_pixel_head_with_action(shapes, action_size, filters, feature_size,
                                batch_size, use_batch_norm):
    observation_shape, linear_input_size = shapes

    head = PixelHeadWithAction(observation_shape, action_size, filters,
                               feature_size, use_batch_norm)
    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand((batch_size, action_size))
    y = head(x, action)

    # check output shape
    assert head._get_linear_input_size() == linear_input_size + action_size
    assert y.shape == (batch_size, feature_size)

    # check use of batch norm
    head.eval()
    eval_y = head(x, action)
    assert not use_batch_norm == torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(head, (x, action))


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('hidden_units', [[256, 256]])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_vector_head(observation_shape, hidden_units, batch_size,
                     use_batch_norm):
    head = VectorHead(observation_shape, hidden_units, use_batch_norm)

    x = torch.rand((batch_size, ) + observation_shape)
    y = head(x)

    # check output shape
    assert head.feature_size == hidden_units[-1]
    assert y.shape == (batch_size, hidden_units[-1])

    # check use of batch norm
    head.eval()
    eval_y = head(x)
    assert not use_batch_norm == torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(head, (x, ))


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('hidden_units', [[256, 256]])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_vector_head(observation_shape, action_size, hidden_units, batch_size,
                     use_batch_norm):
    head = VectorHeadWithAction(observation_shape, action_size, hidden_units,
                                use_batch_norm)

    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand((batch_size, action_size))
    y = head(x, action)

    # check output shape
    assert head.feature_size == hidden_units[-1]
    assert y.shape == (batch_size, hidden_units[-1])

    # check use of batch norm
    head.eval()
    eval_y = head(x, action)
    assert not use_batch_norm == torch.allclose(y, eval_y)

    # check layer connection
    check_parameter_updates(head, (x, action))
