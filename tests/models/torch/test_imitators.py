import pytest
import torch
import torch.nn.functional as F

from d3rlpy.models.encoders import DefaultEncoderFactory

from d3rlpy.models.torch.imitators import ConditionalVAE
from d3rlpy.models.torch.imitators import DiscreteImitator
from d3rlpy.models.torch.imitators import DeterministicRegressor
from d3rlpy.models.torch.imitators import ProbablisticRegressor
from .model_test import check_parameter_updates, DummyEncoder


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("batch_size", [32])
def test_conditional_vae(
    feature_size, action_size, latent_size, beta, batch_size
):
    encoder_encoder = DummyEncoder(feature_size, action_size, True)
    decoder_encoder = DummyEncoder(feature_size, latent_size, True)
    vae = ConditionalVAE(encoder_encoder, decoder_encoder, beta)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = vae(x, action)
    assert y.shape == (batch_size, action_size)

    # check encode
    dist = vae.encode(x, action)
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.mean.shape == (batch_size, latent_size)

    # check decode
    latent = torch.rand(batch_size, latent_size)
    y = vae.decode(x, latent)
    assert y.shape == (batch_size, action_size)

    # TODO: test vae.compute_likelihood_loss(x, action)

    # check layer connections
    check_parameter_updates(vae, (x, action))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("beta", [1e-2])
@pytest.mark.parametrize("batch_size", [32])
def test_discrete_imitator(feature_size, action_size, beta, batch_size):
    encoder = DummyEncoder(feature_size)
    imitator = DiscreteImitator(encoder, action_size, beta)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = imitator(x)
    assert torch.allclose(y.exp().sum(dim=1), torch.ones(batch_size))
    y, logits = imitator.compute_log_probs_with_logits(x)
    assert torch.allclose(y, F.log_softmax(logits, dim=1))

    action = torch.randint(low=0, high=action_size - 1, size=(batch_size,))
    loss = imitator.compute_error(x, action)
    penalty = (logits ** 2).mean()
    assert torch.allclose(loss, F.nll_loss(y, action) + beta * penalty)

    # check layer connections
    check_parameter_updates(imitator, (x, action))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_deterministic_regressor(feature_size, action_size, batch_size):
    encoder = DummyEncoder(feature_size)
    imitator = DeterministicRegressor(encoder, action_size)

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
def test_probablistic_regressor(feature_size, action_size, batch_size, n):
    encoder = DummyEncoder(feature_size)
    imitator = ProbablisticRegressor(encoder, action_size)

    x = torch.rand(batch_size, feature_size)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)

    action = torch.rand(batch_size, action_size)
    loss = imitator.compute_error(x, action)

    y = imitator.sample_n(x, n)
    assert y.shape == (batch_size, n, action_size)

    # check layer connections
    check_parameter_updates(imitator, (x, action))
