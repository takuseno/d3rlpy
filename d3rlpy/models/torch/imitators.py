import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from .heads import create_head


def create_conditional_vae(observation_shape,
                           action_size,
                           latent_size,
                           beta,
                           use_batch_norm=False):
    encoder_head = create_head(observation_shape,
                               action_size,
                               use_batch_norm=use_batch_norm)
    decoder_head = create_head(observation_shape,
                               latent_size,
                               use_batch_norm=use_batch_norm)
    return ConditionalVAE(encoder_head, decoder_head, beta)


def create_discrete_imitator(observation_shape,
                             action_size,
                             beta,
                             use_batch_norm=False):
    head = create_head(observation_shape, use_batch_norm=use_batch_norm)
    return DiscreteImitator(head, action_size, beta)


def create_deterministic_regressor(observation_shape,
                                   action_size,
                                   use_batch_norm=False):
    head = create_head(observation_shape, use_batch_norm=use_batch_norm)
    return DeterministicRegressor(head, action_size)


def create_probablistic_regressor(observation_shape,
                                  action_size,
                                  use_batch_norm=False):
    head = create_head(observation_shape, use_batch_norm=use_batch_norm)
    return ProbablisticRegressor(head, action_size)


class ConditionalVAE(nn.Module):
    def __init__(self, encoder_head, decoder_head, beta):
        super().__init__()
        self.encoder_head = encoder_head
        self.decoder_head = decoder_head
        self.beta = beta

        self.action_size = encoder_head.action_size
        self.latent_size = decoder_head.action_size

        # encoder
        self.mu = nn.Linear(encoder_head.feature_size, self.latent_size)
        self.logstd = nn.Linear(encoder_head.feature_size, self.latent_size)
        # decoder
        self.fc = nn.Linear(decoder_head.feature_size, self.action_size)

    def forward(self, x, action):
        dist = self.encode(x, action)
        return self.decode(x, dist.rsample())

    def encode(self, x, action):
        h = self.encoder_head(x, action)
        mu = self.mu(h)
        logstd = self.logstd(h)
        clipped_logstd = logstd.clamp(-20.0, 2.0)
        return Normal(mu, clipped_logstd.exp())

    def decode(self, x, latent):
        h = self.decoder_head(x, latent)
        return torch.tanh(self.fc(h))

    def compute_error(self, x, action):
        dist = self.encode(x, action)
        kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
        y = self.decode(x, dist.rsample())
        return F.mse_loss(y, action) + self.beta * kl_loss


class DiscreteImitator(nn.Module):
    def __init__(self, head, action_size, beta):
        super().__init__()
        self.head = head
        self.beta = beta
        self.fc = nn.Linear(head.feature_size, action_size)

    def forward(self, x, with_logits=False):
        h = self.head(x)
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
    def __init__(self, head, action_size):
        super().__init__()
        self.head = head
        self.fc = nn.Linear(head.feature_size, action_size)

    def forward(self, x):
        h = self.head(x)
        h = self.fc(h)
        return torch.tanh(h)

    def compute_error(self, x, action):
        return F.mse_loss(self.forward(x), action)


class ProbablisticRegressor(nn.Module):
    def __init__(self, head, action_size):
        super().__init__()
        self.head = head
        self.mu = nn.Linear(head.feature_size, action_size)
        self.logstd = nn.Linear(head.feature_size, action_size)

    def dist(self, x):
        h = self.head(x)
        mu = self.mu(h)
        logstd = self.logstd(h)
        clipped_logstd = logstd.clamp(-20.0, 2.0)
        return Normal(mu, clipped_logstd)

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
