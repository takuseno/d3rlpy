import numpy as np
import pytest
import torch

from d3rlpy.models.torch.q_functions.utility import (
    compute_quantile_huber_loss,
    pick_quantile_value_by_action,
    pick_value_by_action,
)

from ..model_test import ref_quantile_huber_loss


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("keepdims", [True, False])
def test_pick_value_by_action(
    batch_size: int, action_size: int, keepdims: bool
) -> None:
    values = torch.rand(batch_size, action_size)
    action = torch.randint(action_size, size=(batch_size,))

    rets = pick_value_by_action(values, action, keepdims)

    if keepdims:
        assert rets.shape == (batch_size, 1)
    else:
        assert rets.shape == (batch_size,)

    rets = rets.view(batch_size, -1)

    for i in range(batch_size):
        assert (rets[i] == values[i][action[i]]).all()


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("keepdims", [True, False])
def test_pick_quantile_value_by_action(
    batch_size: int,
    action_size: int,
    n_quantiles: int,
    keepdims: bool,
) -> None:
    values = torch.rand(batch_size, action_size, n_quantiles)
    action = torch.randint(action_size, size=(batch_size,))

    rets = pick_quantile_value_by_action(values, action, keepdims)

    if keepdims:
        assert rets.shape == (batch_size, 1, n_quantiles)
    else:
        assert rets.shape == (batch_size, n_quantiles)

    rets = rets.view(batch_size, -1)

    for i in range(batch_size):
        assert (rets[i] == values[i][action[i]]).all()


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n_quantiles", [200])
def test_compute_quantile_huber_loss(batch_size: int, n_quantiles: int) -> None:
    y = np.random.random((batch_size, n_quantiles, 1))
    target = np.random.random((batch_size, 1, n_quantiles))
    taus = np.random.random((1, 1, n_quantiles))

    ref_loss = ref_quantile_huber_loss(y, target, taus, n_quantiles)
    loss = compute_quantile_huber_loss(
        torch.tensor(y), torch.tensor(target), torch.tensor(taus)
    )

    assert np.allclose(loss.cpu().detach().numpy(), ref_loss)
