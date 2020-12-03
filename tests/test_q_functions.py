import pytest

from d3rlpy.models.torch.q_functions import DiscreteMeanQFunction
from d3rlpy.models.torch.q_functions import DiscreteQRQFunction
from d3rlpy.models.torch.q_functions import DiscreteIQNQFunction
from d3rlpy.models.torch.q_functions import DiscreteFQFQFunction
from d3rlpy.models.torch.q_functions import ContinuousMeanQFunction
from d3rlpy.models.torch.q_functions import ContinuousQRQFunction
from d3rlpy.models.torch.q_functions import ContinuousIQNQFunction
from d3rlpy.models.torch.q_functions import ContinuousFQFQFunction
from d3rlpy.q_functions import create_q_func_factory
from d3rlpy.q_functions import MeanQFunctionFactory
from d3rlpy.q_functions import QRQFunctionFactory
from d3rlpy.q_functions import IQNQFunctionFactory
from d3rlpy.q_functions import FQFQFunctionFactory
from d3rlpy.encoders import VectorEncoderFactory


def _create_encoder(observation_shape, action_size):
    factory = VectorEncoderFactory()
    if action_size is None:
        encoder = factory.create(observation_shape, 2)
    else:
        encoder = factory.create(observation_shape, None)
    return encoder


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [None, 2])
def test_mean_q_function_factory(observation_shape, action_size):
    encoder = _create_encoder(observation_shape, action_size)

    factory = MeanQFunctionFactory()
    q_func = factory.create(encoder, action_size)

    if action_size is None:
        assert isinstance(q_func, ContinuousMeanQFunction)
    else:
        assert isinstance(q_func, DiscreteMeanQFunction)

    assert factory.get_type() == 'mean'

    params = factory.get_params()
    new_factory = MeanQFunctionFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [None, 2])
def test_qr_q_function_factory(observation_shape, action_size):
    encoder = _create_encoder(observation_shape, action_size)

    factory = QRQFunctionFactory()
    q_func = factory.create(encoder, action_size)

    if action_size is None:
        assert isinstance(q_func, ContinuousQRQFunction)
    else:
        assert isinstance(q_func, DiscreteQRQFunction)

    assert factory.get_type() == 'qr'

    params = factory.get_params()
    new_factory = QRQFunctionFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [None, 2])
def test_iqn_q_function_factory(observation_shape, action_size):
    encoder = _create_encoder(observation_shape, action_size)

    factory = IQNQFunctionFactory()
    q_func = factory.create(encoder, action_size)

    if action_size is None:
        assert isinstance(q_func, ContinuousIQNQFunction)
    else:
        assert isinstance(q_func, DiscreteIQNQFunction)

    assert factory.get_type() == 'iqn'

    params = factory.get_params()
    new_factory = IQNQFunctionFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [None, 2])
def test_fqf_q_function_factory(observation_shape, action_size):
    encoder = _create_encoder(observation_shape, action_size)

    factory = FQFQFunctionFactory()
    q_func = factory.create(encoder, action_size)

    if action_size is None:
        assert isinstance(q_func, ContinuousFQFQFunction)
    else:
        assert isinstance(q_func, DiscreteFQFQFunction)

    assert factory.get_type() == 'fqf'

    params = factory.get_params()
    new_factory = FQFQFunctionFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize('name', ['mean', 'qr', 'iqn', 'fqf'])
def test_create_q_func_factory(name):
    factory = create_q_func_factory(name)
    if name == 'mean':
        assert isinstance(factory, MeanQFunctionFactory)
    elif name == 'qr':
        assert isinstance(factory, QRQFunctionFactory)
    elif name == 'iqn':
        assert isinstance(factory, IQNQFunctionFactory)
    elif name == 'fqf':
        assert isinstance(factory, FQFQFunctionFactory)
