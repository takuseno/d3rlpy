import pytest
import torch

from d3rlpy.models.torch.imitators import (
    VAEDecoder,
    VAEEncoder,
    compute_deterministic_imitation_loss,
    compute_discrete_imitation_loss,
    compute_stochastic_imitation_loss,
    compute_vae_error,
    forward_vae_sample,
    forward_vae_sample_n,
)
from d3rlpy.models.torch.policies import (
    CategoricalPolicy,
    DeterministicPolicy,
    NormalPolicy,
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
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("batch_size", [32])
def test_vae_encoder(
    observation_shape: Shape,
    action_size: int,
    latent_size: int,
    batch_size: int,
) -> None:
    encoder = DummyEncoderWithAction(observation_shape, action_size)
    vae_encoder = VAEEncoder(
        encoder=encoder,
        hidden_size=encoder.get_feature_size(),
        latent_size=latent_size,
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    action = torch.rand(batch_size, action_size)
    dist = vae_encoder(x, action)
    assert dist.mean.shape == (batch_size, latent_size)


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n", [100])
def test_vae_decoder(
    observation_shape: Shape,
    action_size: int,
    latent_size: int,
    batch_size: int,
    n: int,
) -> None:
    encoder = DummyEncoderWithAction(observation_shape, latent_size)
    vae_decoder = VAEDecoder(
        encoder=encoder,
        hidden_size=encoder.get_feature_size(),
        action_size=action_size,
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    latent = torch.rand(batch_size, latent_size)
    action = vae_decoder(x, latent)
    assert action.shape == (batch_size, action_size)

    # check forward_vae_sample
    y = forward_vae_sample(vae_decoder, x, latent_size)
    assert y.shape == (batch_size, action_size)

    # check forward_vae_sample_n
    y = forward_vae_sample_n(vae_decoder, x, latent_size, n)
    assert y.shape == (batch_size, n, action_size)

    # check layer connections
    check_parameter_updates(vae_decoder, (x, latent))


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("beta", [0.5])
def test_conditional_vae(
    observation_shape: Shape,
    action_size: int,
    latent_size: int,
    batch_size: int,
    beta: float,
) -> None:
    encoder_encoder = DummyEncoderWithAction(observation_shape, action_size)
    decoder_encoder = DummyEncoderWithAction(observation_shape, latent_size)
    vae_encoder = VAEEncoder(
        encoder=encoder_encoder,
        hidden_size=encoder_encoder.get_feature_size(),
        latent_size=latent_size,
    )
    vae_decoder = VAEDecoder(
        encoder=decoder_encoder,
        hidden_size=decoder_encoder.get_feature_size(),
        action_size=action_size,
    )

    x = create_torch_observations(observation_shape, batch_size)
    action = torch.rand(batch_size, action_size)

    # test compute error
    error = compute_vae_error(vae_encoder, vae_decoder, x, action, beta)
    assert error.ndim == 0


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("beta", [0.5])
def test_compute_discrete_imitation_loss(
    observation_shape: Shape, action_size: int, batch_size: int, beta: float
) -> None:
    encoder = DummyEncoder(observation_shape)
    policy = CategoricalPolicy(
        encoder=encoder,
        hidden_size=encoder.get_feature_size(),
        action_size=action_size,
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    action = torch.randint(low=0, high=action_size, size=(batch_size,))
    loss = compute_discrete_imitation_loss(policy, x, action, beta)
    assert loss.ndim == 0


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_compute_deterministic_imitation_loss(
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
    action = torch.rand(batch_size, action_size)
    loss = compute_deterministic_imitation_loss(policy, x, action)
    assert loss.ndim == 0
    assert loss == ((policy(x).squashed_mu - action) ** 2).mean()


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("min_logstd", [-20.0])
@pytest.mark.parametrize("max_logstd", [2.0])
@pytest.mark.parametrize("use_std_parameter", [True, False])
def test_compute_stochastic_imitation_loss(
    observation_shape: Shape,
    action_size: int,
    batch_size: int,
    min_logstd: float,
    max_logstd: float,
    use_std_parameter: bool,
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
    action = torch.rand(batch_size, action_size)
    loss = compute_stochastic_imitation_loss(policy, x, action)
    assert loss.ndim == 0
