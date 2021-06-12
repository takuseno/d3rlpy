import pytest
import torch

from d3rlpy.models.builders import create_continuous_q_function
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory, QRQFunctionFactory
from d3rlpy.models.torch.q_functions import compute_max_with_n_actions


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [3])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("n_ensembles", [2])
@pytest.mark.parametrize("batch_size", [100])
@pytest.mark.parametrize("n_quantiles", [32])
@pytest.mark.parametrize("n_actions", [10])
@pytest.mark.parametrize("lam", [0.75])
def test_compute_max_with_n_actions(
    observation_shape,
    action_size,
    encoder_factory,
    q_func_factory,
    n_ensembles,
    batch_size,
    n_quantiles,
    n_actions,
    lam,
):
    q_func = create_continuous_q_function(
        observation_shape,
        action_size,
        encoder_factory,
        q_func_factory,
        n_ensembles=n_ensembles,
    )
    x = torch.rand(batch_size, *observation_shape)
    actions = torch.rand(batch_size, n_actions, action_size)

    y = compute_max_with_n_actions(x, actions, q_func, lam)

    if isinstance(q_func_factory, MeanQFunctionFactory):
        assert y.shape == (batch_size, 1)
    else:
        assert y.shape == (batch_size, q_func_factory.n_quantiles)
