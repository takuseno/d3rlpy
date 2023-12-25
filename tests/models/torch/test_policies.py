import pytest
import torch

from d3rlpy.models.torch.distributions import (
    GaussianDistribution,
    SquashedGaussianDistribution,
)
from d3rlpy.models.torch.policies import (
    ActionOutput,
    CategoricalPolicy,
    DeterministicPolicy,
    DeterministicResidualPolicy,
    NormalPolicy,
    build_gaussian_distribution,
    build_squashed_gaussian_distribution,
)
from d3rlpy.types import Shape

from ...testing_utils import create_torch_observations
from .model_test import (
    DummyEncoder,
    DummyEncoderWithAction,
    check_parameter_updates,
)


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_deterministic_policy(
    observation_shape: Shape, action_size: int, batch_size: int
) -> None:
    encoder = DummyEncoder(observation_shape)
    policy = DeterministicPolicy(
        encoder=encoder,
        hidden_size=encoder.get_feature_size(),
        action_size=action_size,
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    y = policy(x)
    assert y.mu.shape == (batch_size, action_size)
    assert y.squashed_mu.shape == (batch_size, action_size)

    # check best action
    best_action = policy(x).squashed_mu
    assert torch.allclose(best_action, y.squashed_mu)

    # check layer connection
    check_parameter_updates(policy, (x,))


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("scale", [0.05])
@pytest.mark.parametrize("batch_size", [32])
def test_deterministic_residual_policy(
    observation_shape: Shape, action_size: int, scale: float, batch_size: int
) -> None:
    encoder = DummyEncoderWithAction(observation_shape, action_size)
    policy = DeterministicResidualPolicy(
        encoder=encoder,
        hidden_size=encoder.get_feature_size(),
        action_size=action_size,
        scale=scale,
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    action = torch.rand(batch_size, action_size)
    y = policy(x, action)
    assert y.mu.shape == (batch_size, action_size)
    assert y.squashed_mu.shape == (batch_size, action_size)

    # check residual
    assert not (y.squashed_mu == action).any()
    assert ((y.squashed_mu - action).abs() <= scale).all()

    # check best action
    best_action = policy(x, action).squashed_mu
    assert torch.allclose(best_action, y.squashed_mu)

    # check layer connection
    check_parameter_updates(policy, (x, action))


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("min_logstd", [-20.0])
@pytest.mark.parametrize("max_logstd", [2.0])
@pytest.mark.parametrize("use_std_parameter", [True, False])
@pytest.mark.parametrize("n", [10])
def test_normal_policy(
    observation_shape: Shape,
    action_size: int,
    batch_size: int,
    min_logstd: float,
    max_logstd: float,
    use_std_parameter: bool,
    n: int,
) -> None:
    encoder = DummyEncoder(observation_shape)
    policy = NormalPolicy(
        encoder=encoder,
        hidden_size=encoder.get_feature_size(),
        action_size=action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    y = policy(x)
    assert y.mu.shape == (batch_size, action_size)
    assert y.squashed_mu.shape == (batch_size, action_size)

    # check layer connection
    check_parameter_updates(policy, (x,))


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n", [10])
def test_categorical_policy(
    observation_shape: Shape, action_size: int, batch_size: int, n: int
) -> None:
    encoder = DummyEncoder(observation_shape)
    policy = CategoricalPolicy(
        encoder=encoder,
        hidden_size=encoder.get_feature_size(),
        action_size=action_size,
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    dist = policy(x)
    assert dist.probs.shape == (batch_size, action_size)
    assert dist.sample().shape == (batch_size,)


@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_build_gaussian_distribution(action_size: int, batch_size: int) -> None:
    mu = torch.rand(batch_size, action_size)
    squashed_mu = torch.rand(batch_size, action_size)
    logstd = torch.rand(batch_size, action_size)
    action = ActionOutput(mu, squashed_mu, logstd)

    dist = build_gaussian_distribution(action)
    assert isinstance(dist, GaussianDistribution)

    assert torch.all(dist.mean == squashed_mu)
    assert torch.all(dist.std == logstd.exp())


@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_build_squashed_gaussian_distribution(
    action_size: int, batch_size: int
) -> None:
    mu = torch.rand(batch_size, action_size)
    squashed_mu = torch.rand(batch_size, action_size)
    logstd = torch.rand(batch_size, action_size)
    action = ActionOutput(mu, squashed_mu, logstd)

    dist = build_squashed_gaussian_distribution(action)
    assert isinstance(dist, SquashedGaussianDistribution)

    assert torch.all(dist.mean == torch.tanh(mu))
    assert torch.all(dist.std == logstd.exp())
