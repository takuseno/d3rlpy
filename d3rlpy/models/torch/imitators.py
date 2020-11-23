import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from .encoders import create_encoder


def create_conditional_vae(observation_shape,
                           action_size,
                           latent_size,
                           beta,
                           encoder_factory=None):
    if encoder_factory:
        encoder_encoder = encoder_factory.create(observation_shape,
                                                 action_size)
        decoder_encoder = encoder_factory.create(observation_shape,
                                                 latent_size)
    else:
        encoder_encoder = create_encoder(observation_shape, action_size)
        decoder_encoder = create_encoder(observation_shape, latent_size)
    return ConditionalVAE(encoder_encoder, decoder_encoder, beta)


def create_discrete_imitator(observation_shape,
                             action_size,
                             beta,
                             encoder_factory=None):
    if encoder_factory:
        encoder = encoder_factory.create(observation_shape)
    else:
        encoder = create_encoder(observation_shape)
    return DiscreteImitator(encoder, action_size, beta)


def create_deterministic_regressor(observation_shape, action_size,
                                   encoder_factory):
    if encoder_factory:
        encoder = encoder_factory.create(observation_shape)
    else:
        encoder = create_encoder(observation_shape)
    return DeterministicRegressor(encoder, action_size)


def create_probablistic_regressor(observation_shape,
                                  action_size,
                                  encoder_factory=None):
    if encoder_factory:
        encoder = encoder_factory.create(observation_shape)
    else:
        encoder = create_encoder(observation_shape)
    return ProbablisticRegressor(encoder, action_size)


class ConditionalVAE(nn.Module):
    def __init__(self, encoder_encoder, decoder_encoder, beta):
        super().__init__()
        self.encoder_encoder = encoder_encoder
        self.decoder_encoder = decoder_encoder
        self.beta = beta

        self.action_size = encoder_encoder.action_size
        self.latent_size = decoder_encoder.action_size

        # encoder
        self.mu = nn.Linear(encoder_encoder.get_feature_size(),
                            self.latent_size)
        self.logstd = nn.Linear(encoder_encoder.get_feature_size(),
                                self.latent_size)
        # decoder
        self.fc = nn.Linear(decoder_encoder.get_feature_size(),
                            self.action_size)

    def forward(self, x, action):
        dist = self.encode(x, action)
        return self.decode(x, dist.rsample())

    def encode(self, x, action):
        h = self.encoder_encoder(x, action)
        mu = self.mu(h)
        logstd = self.logstd(h)
        clipped_logstd = logstd.clamp(-20.0, 2.0)
        return Normal(mu, clipped_logstd.exp())

    def decode(self, x, latent):
        h = self.decoder_encoder(x, latent)
        return torch.tanh(self.fc(h))

    def compute_error(self, x, action):
        dist = self.encode(x, action)
        kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
        y = self.decode(x, dist.rsample())
        return F.mse_loss(y, action) + self.beta * kl_loss


class DiscreteImitator(nn.Module):
    def __init__(self, encoder, action_size, beta):
        super().__init__()
        self.encoder = encoder
        self.beta = beta
        self.fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x, with_logits=False):
        h = self.encoder(x)
        logits = self.fc(h)
        log_probs = F.log_softmax(logits, dim=1)
        if with_logits:
            return log_probs, logits
        return log_probs

    def compute_error(self, x, action):
        log_probs, logits = self.forward(x, with_logits=True)
        penalty = (logits**2).mean()
        return F.nll_loss(log_probs, action.view(-1)) + self.beta * penalty


class DeterministicRegressor(nn.Module):
    def __init__(self, encoder, action_size):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x):
        h = self.encoder(x)
        h = self.fc(h)
        return torch.tanh(h)

    def compute_error(self, x, action):
        return F.mse_loss(self.forward(x), action)


class ProbablisticRegressor(nn.Module):
    def __init__(self, encoder, action_size):
        super().__init__()
        self.encoder = encoder
        self.mu = nn.Linear(encoder.get_feature_size(), action_size)
        self.logstd = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logstd = self.logstd(h)
        clipped_logstd = logstd.clamp(-20.0, 2.0)
        return Normal(mu, clipped_logstd.exp())

    def forward(self, x):
        dist = self.dist(x)
        return dist.rsample()

    def sample_n(self, x, n):
        dist = self.dist(x)
        actions = dist.rsample((n, ))
        # (n, batch, action) -> (batch, n, action)
        return actions.transpose(0, 1)

    def compute_error(self, x, action):
        return F.mse_loss(self.forward(x), action)
