import pytest
import torch

from d3rlpy.models.torch.imitators import (
    ConditionalVAE,
    VAEDecoder,
    VAEEncoder,
    compute_deterministic_imitation_loss,
    compute_discrete_imitation_loss,
    compute_stochastic_imitation_loss,
    compute_vae_error,
    forward_vae_decode,
    forward_vae_encode,
    forward_vae_sample,
    forward_vae_sample_n,
)
from d3rlpy.models.torch.policies import (
    CategoricalPolicy,
    DeterministicPolicy,
    NormalPolicy,
)

from .model_test import (
    DummyEncoder,
    DummyEncoderWithAction,
    check_parameter_updates,
)


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("batch_size", [32])
def test_vae_encoder(
    feature_size: int,
    action_size: int,
    latent_size: int,
    batch_size: int,
) -> None:
    encoder = DummyEncoderWithAction(feature_size, action_size)
    vae_encoder = VAEEncoder(
        encoder=encoder,
        hidden_size=feature_size,
        latent_size=latent_size,
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    dist = vae_encoder(x, action)
    assert dist.mean.shape == (batch_size, latent_size)


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("batch_size", [32])
def test_vae_decoder(
    feature_size: int,
    action_size: int,
    latent_size: int,
    batch_size: int,
) -> None:
    encoder = DummyEncoderWithAction(feature_size, latent_size)
    vae_decoder = VAEDecoder(
        encoder=encoder,
        hidden_size=feature_size,
        action_size=action_size,
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    latent = torch.rand(batch_size, latent_size)
    action = vae_decoder(x, latent)
    assert action.shape == (batch_size, action_size)

    # check layer connections
    check_parameter_updates(vae_decoder, (x, latent))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("beta", [0.5])
def test_conditional_vae(
    feature_size: int,
    action_size: int,
    latent_size: int,
    batch_size: int,
    n: int,
    beta: float,
) -> None:
    encoder_encoder = DummyEncoderWithAction(feature_size, action_size)
    decoder_encoder = DummyEncoderWithAction(feature_size, latent_size)
    vae_encoder = VAEEncoder(
        encoder=encoder_encoder,
        hidden_size=feature_size,
        latent_size=latent_size,
    )
    vae_decoder = VAEDecoder(
        encoder=decoder_encoder,
        hidden_size=feature_size,
        action_size=action_size,
    )
    vae = ConditionalVAE(
        encoder=vae_encoder,
        decoder=vae_decoder,
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = vae(x, action)
    assert y.shape == (batch_size, action_size)

    # test encode
    dist = forward_vae_encode(vae, x, action)
    assert dist.mean.shape == (batch_size, latent_size)

    # test decode
    y = forward_vae_decode(vae, x, dist.sample())
    assert y.shape == (batch_size, action_size)

    # test decode sample
    y = forward_vae_sample(vae, x)
    assert y.shape == (batch_size, action_size)

    # test decode sample n
    y = forward_vae_sample_n(vae, x, n)
    assert y.shape == (batch_size, n, action_size)

    # test compute error
    error = compute_vae_error(vae, x, action, beta)
    assert error.ndim == 0

    # check layer connections
    check_parameter_updates(vae, (x, action))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("beta", [0.5])
def test_compute_discrete_imitation_loss(
    feature_size: int, action_size: int, batch_size: int, beta: float
) -> None:
    encoder = DummyEncoder(feature_size)
    policy = CategoricalPolicy(
        encoder=encoder,
        hidden_size=feature_size,
        action_size=action_size,
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.randint(low=0, high=action_size, size=(batch_size,))
    loss = compute_discrete_imitation_loss(policy, x, action, beta)
    assert loss.ndim == 0


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_compute_deterministic_imitation_loss(
    feature_size: int, action_size: int, batch_size: int
) -> None:
    encoder = DummyEncoder(feature_size)
    policy = DeterministicPolicy(
        encoder=encoder,
        hidden_size=feature_size,
        action_size=action_size,
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    loss = compute_deterministic_imitation_loss(policy, x, action)
    assert loss.ndim == 0
    assert loss == ((policy(x).squashed_mu - action) ** 2).mean()


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("min_logstd", [-20.0])
@pytest.mark.parametrize("max_logstd", [2.0])
@pytest.mark.parametrize("use_std_parameter", [True, False])
def test_compute_stochastic_imitation_loss(
    feature_size: int,
    action_size: int,
    batch_size: int,
    min_logstd: float,
    max_logstd: float,
    use_std_parameter: bool,
) -> None:
    encoder = DummyEncoder(feature_size)
    policy = NormalPolicy(
        encoder=encoder,
        hidden_size=feature_size,
        action_size=action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    loss = compute_stochastic_imitation_loss(policy, x, action)
    assert loss.ndim == 0
