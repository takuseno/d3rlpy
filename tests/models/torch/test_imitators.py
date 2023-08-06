import pytest
import torch
import torch.nn.functional as F

from d3rlpy.models.torch.imitators import (
    ConditionalVAE,
    DeterministicRegressor,
    DiscreteImitator,
    ProbablisticRegressor,
    VAEDecoder,
    VAEEncoder,
    compute_vae_error,
    forward_vae_decode,
    forward_vae_encode,
    forward_vae_sample,
    forward_vae_sample_n,
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
@pytest.mark.parametrize("beta", [1e-2])
@pytest.mark.parametrize("batch_size", [32])
def test_discrete_imitator(
    feature_size: int, action_size: int, beta: float, batch_size: int
) -> None:
    encoder = DummyEncoder(feature_size)
    imitator = DiscreteImitator(
        encoder=encoder,
        hidden_size=feature_size,
        action_size=action_size,
        beta=beta,
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = imitator(x)
    assert torch.allclose(y.exp().sum(dim=1), torch.ones(batch_size))
    y, logits = imitator.compute_log_probs_with_logits(x)
    assert torch.allclose(y, F.log_softmax(logits, dim=1))

    action = torch.randint(low=0, high=action_size - 1, size=(batch_size,))
    loss = imitator.compute_error(x, action)
    penalty = (logits**2).mean()
    assert torch.allclose(loss, F.nll_loss(y, action) + beta * penalty)

    # check layer connections
    check_parameter_updates(imitator, (x, action))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_deterministic_regressor(
    feature_size: int, action_size: int, batch_size: int
) -> None:
    encoder = DummyEncoder(feature_size)
    imitator = DeterministicRegressor(
        encoder=encoder,
        hidden_size=feature_size,
        action_size=action_size,
    )

    x = torch.rand(batch_size, feature_size)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)

    action = torch.rand(batch_size, action_size)
    loss = imitator.compute_error(x, action)
    assert torch.allclose(F.mse_loss(y, action), loss)

    # check layer connections
    check_parameter_updates(imitator, (x, action))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n", [10])
def test_probablistic_regressor(
    feature_size: int, action_size: int, batch_size: int, n: int
) -> None:
    encoder = DummyEncoder(feature_size)
    imitator = ProbablisticRegressor(
        encoder=encoder,
        hidden_size=feature_size,
        action_size=action_size,
        min_logstd=-20,
        max_logstd=2,
    )

    x = torch.rand(batch_size, feature_size)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)

    action = torch.rand(batch_size, action_size)
    loss = imitator.compute_error(x, action)
    assert loss.ndim == 0

    y = imitator.sample_n(x, n)
    assert y.shape == (batch_size, n, action_size)

    # check layer connections
    check_parameter_updates(imitator, (x, action))
