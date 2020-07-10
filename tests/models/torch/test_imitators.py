import pytest
import torch
import torch.nn.functional as F

from d3rlpy.models.torch.imitators import create_conditional_vae
from d3rlpy.models.torch.imitators import create_discrete_imitator
from d3rlpy.models.torch.imitators import create_deterministic_regressor
from d3rlpy.models.torch.imitators import ConditionalVAE
from d3rlpy.models.torch.imitators import DiscreteImitator
from d3rlpy.models.torch.imitators import DeterministicRegressor
from .model_test import check_parameter_updates, DummyHead


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('latent_size', [32])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_conditional_vae(observation_shape, action_size, latent_size,
                                batch_size, use_batch_norm):
    vae = create_conditional_vae(observation_shape, action_size, latent_size,
                                 use_batch_norm)

    assert isinstance(vae, ConditionalVAE)

    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = vae(x, action)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('beta', [1e-2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_discrete_imitator(observation_shape, action_size, beta,
                                  batch_size, use_batch_norm):
    imitator = create_discrete_imitator(observation_shape, action_size, beta,
                                        use_batch_norm)

    assert isinstance(imitator, DiscreteImitator)

    x = torch.rand((batch_size, ) + observation_shape)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_deterministic_regressor(observation_shape, action_size,
                                        batch_size, use_batch_norm):
    imitator = create_deterministic_regressor(observation_shape, action_size,
                                              use_batch_norm)

    assert isinstance(imitator, DeterministicRegressor)

    x = torch.rand((batch_size, ) + observation_shape)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('latent_size', [32])
@pytest.mark.parametrize('beta', [0.5])
@pytest.mark.parametrize('batch_size', [32])
def test_conditional_vae(feature_size, action_size, latent_size, beta,
                         batch_size):
    encoder_head = DummyHead(feature_size, action_size, True)
    decoder_head = DummyHead(feature_size, latent_size, True)
    vae = ConditionalVAE(encoder_head, decoder_head, beta)

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


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('beta', [1e-2])
@pytest.mark.parametrize('batch_size', [32])
def test_discrete_imitator(feature_size, action_size, beta, batch_size):
    head = DummyHead(feature_size)
    imitator = DiscreteImitator(head, action_size, beta)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = imitator(x)
    assert torch.allclose(y.exp().sum(dim=1), torch.ones(batch_size))
    y, logits = imitator(x, with_logits=True)
    assert torch.allclose(y, F.log_softmax(logits, dim=1))

    action = torch.randint(low=0, high=action_size - 1, size=(batch_size, ))
    loss = imitator.compute_error(x, action)
    penalty = (logits**2).mean()
    assert torch.allclose(loss, F.nll_loss(y, action) + beta * penalty)


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
def test_deterministic_regressor(feature_size, action_size, batch_size):
    head = DummyHead(feature_size)
    imitator = DeterministicRegressor(head, action_size)

    x = torch.rand(batch_size, feature_size)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)

    action = torch.rand(batch_size, action_size)
    loss = imitator.compute_error(x, action)
    assert torch.allclose(F.mse_loss(y, action), loss)
