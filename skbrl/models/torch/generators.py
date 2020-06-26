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
                           use_batch_norm=True):
    encoder_head = create_head(observation_shape,
                               action_size,
                               use_batch_norm=use_batch_norm)
    decoder_head = create_head(observation_shape,
                               latent_size,
                               use_batch_norm=use_batch_norm)
    return ConditionalVAE(encoder_head, decoder_head, beta)


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

    def compute_likelihood_loss(self, x, action):
        dist = self.encode(x, action)
        kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
        y = self.decode(x, dist.rsample())
        return F.mse_loss(y, action) + self.beta * kl_loss
