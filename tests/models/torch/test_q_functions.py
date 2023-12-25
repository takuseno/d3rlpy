import pytest
import torch

from d3rlpy.models.builders import create_continuous_q_function
from d3rlpy.models.q_functions import (
    MeanQFunctionFactory,
    QFunctionFactory,
    QRQFunctionFactory,
)
from d3rlpy.models.torch.q_functions import compute_max_with_n_actions
from d3rlpy.types import Shape

from ...testing_utils import create_torch_observations
from .model_test import DummyEncoderFactory


@pytest.mark.parametrize(
    "observation_shape", [(4, 84, 84), (100,), ((100,), (200,))]
)
@pytest.mark.parametrize("action_size", [3])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("n_ensembles", [2])
@pytest.mark.parametrize("batch_size", [100])
@pytest.mark.parametrize("n_actions", [10])
@pytest.mark.parametrize("lam", [0.75])
def test_compute_max_with_n_actions(
    observation_shape: Shape,
    action_size: int,
    q_func_factory: QFunctionFactory,
    n_ensembles: int,
    batch_size: int,
    n_actions: int,
    lam: float,
) -> None:
    _, forwarder = create_continuous_q_function(
        observation_shape,
        action_size,
        DummyEncoderFactory(),
        q_func_factory,
        n_ensembles=n_ensembles,
        device="cpu:0",
    )
    x = create_torch_observations(observation_shape, batch_size)
    actions = torch.rand(batch_size, n_actions, action_size)

    y = compute_max_with_n_actions(x, actions, forwarder, lam)

    if isinstance(q_func_factory, MeanQFunctionFactory):
        assert y.shape == (batch_size, 1)
    else:
        assert isinstance(q_func_factory, QRQFunctionFactory)
        assert y.shape == (batch_size, q_func_factory.n_quantiles)
