import numpy as np
import pytest

from d3rlpy.algos import (
    GreedyTransformerActionSampler,
    IdentityTransformerActionSampler,
    SoftmaxTransformerActionSampler,
)


@pytest.mark.parametrize("action_size", [4])
def test_identity_transformer_action_sampler(action_size: int) -> None:
    action_sampler = IdentityTransformerActionSampler()

    x = np.random.random(action_size)
    action = action_sampler(x)

    assert np.all(action == x)


@pytest.mark.parametrize("action_size", [10])
def test_softmax_transformer_action_sampler(action_size: int) -> None:
    action_sampler = SoftmaxTransformerActionSampler()

    logits = np.random.random(action_size)
    action = action_sampler(logits)
    assert isinstance(action, int)

    same_actions = []
    for _ in range(100):
        same_actions.append(action == action_sampler(logits))
    assert not all(same_actions)


@pytest.mark.parametrize("action_size", [10])
def test_greedy_transformer_action_sampler(action_size: int) -> None:
    action_sampler = GreedyTransformerActionSampler()

    logits = np.random.random(action_size)
    action = action_sampler(logits)
    assert isinstance(action, int)
    assert action == np.argmax(logits)
