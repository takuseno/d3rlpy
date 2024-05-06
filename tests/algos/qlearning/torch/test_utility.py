import pytest

from d3rlpy.algos.qlearning.torch.utility import sample_q_values_with_policy
from d3rlpy.models.builders import (
    create_continuous_q_function,
    create_normal_policy,
)
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.types import Shape

from ....models.torch.model_test import DummyEncoderFactory
from ....testing_utils import create_torch_observations


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 84, 84), ((100,), (200,))]
)
@pytest.mark.parametrize("action_size", [4])
@pytest.mark.parametrize("n_action_samples", [10])
@pytest.mark.parametrize("batch_size", [256])
@pytest.mark.parametrize("n_critics", [2])
def test_sample_q_values_with_policy(
    observation_shape: Shape,
    action_size: int,
    n_action_samples: int,
    batch_size: int,
    n_critics: int,
) -> None:
    policy = create_normal_policy(
        observation_shape=observation_shape,
        action_size=action_size,
        encoder_factory=DummyEncoderFactory(),
        device="cpu:0",
    )
    _, q_func_forwarder = create_continuous_q_function(
        observation_shape=observation_shape,
        action_size=action_size,
        encoder_factory=DummyEncoderFactory(),
        q_func_factory=MeanQFunctionFactory(),
        n_ensembles=n_critics,
        device="cpu:0",
    )

    observations = create_torch_observations(observation_shape, batch_size)

    q_values, log_probs = sample_q_values_with_policy(
        policy=policy,
        q_func_forwarder=q_func_forwarder,
        policy_observations=observations,
        value_observations=observations,
        n_action_samples=n_action_samples,
        detach_policy_output=False,
    )
    assert q_values.shape == (n_critics, batch_size, n_action_samples)
    assert log_probs.shape == (1, batch_size, n_action_samples)
