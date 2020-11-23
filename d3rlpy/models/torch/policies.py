import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from abc import ABCMeta, abstractmethod
from torch.distributions import Normal, Categorical
from .encoders import create_encoder


def create_deterministic_policy(observation_shape,
                                action_size,
                                encoder_factory=None):
    if encoder_factory:
        encoder = encoder_factory.create(observation_shape)
    else:
        encoder = create_encoder(observation_shape)
    return DeterministicPolicy(encoder, action_size)


def create_deterministic_residual_policy(observation_shape,
                                         action_size,
                                         scale,
                                         encoder_factory=None):
    if encoder_factory:
        encoder = encoder_factory.create(observation_shape, action_size)
    else:
        encoder = create_encoder(observation_shape, action_size)
    return DeterministicResidualPolicy(encoder, scale)


def create_normal_policy(observation_shape,
                         action_size,
                         encoder_factory=None,
                         min_logstd=-20.0,
                         max_logstd=2.0,
                         use_std_parameter=False):
    if encoder_factory:
        encoder = encoder_factory.create(observation_shape)
    else:
        encoder = create_encoder(observation_shape)
    return NormalPolicy(encoder,
                        action_size,
                        min_logstd=min_logstd,
                        max_logstd=max_logstd,
                        use_std_parameter=use_std_parameter)


def create_categorical_policy(observation_shape,
                              action_size,
                              encoder_factory=None):
    if encoder_factory:
        encoder = encoder_factory.create(observation_shape)
    else:
        encoder = create_encoder(observation_shape)
    return CategoricalPolicy(encoder, action_size)


def squash_action(dist, raw_action):
    squashed_action = torch.tanh(raw_action)
    jacob = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
    log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=-1, keepdims=True)
    return squashed_action, log_prob


class Policy(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, x, with_log_prob=False):
        pass

    @abstractmethod
    def sample_n(self, x, n, with_log_prob=False):
        pass

    @abstractmethod
    def best_action(self, x):
        pass


class DeterministicPolicy(Policy, nn.Module):
    def __init__(self, encoder, action_size):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x, with_raw=False):
        h = self.encoder(x)
        raw_action = self.fc(h)
        if with_raw:
            return torch.tanh(raw_action), raw_action
        return torch.tanh(raw_action)

    def sample(self, x, with_log_prob=False):
        raise NotImplementedError(
            'deterministic policy does not support sample')

    def sample_n(self, x, n, with_log_prob=False):
        raise NotImplementedError(
            'deterministic policy does not support sample_n')

    def best_action(self, x):
        return self.forward(x)


class DeterministicResidualPolicy(nn.Module):
    def __init__(self, encoder, scale):
        super().__init__()
        self.scale = scale
        self.encoder = encoder
        self.fc = nn.Linear(encoder.get_feature_size(), encoder.action_size)

    def forward(self, x, action):
        h = self.encoder(x, action)
        residual_action = self.scale * torch.tanh(self.fc(h))
        return (action + residual_action).clamp(-1.0, 1.0)

    def best_action(self, x, action):
        return self.forward(x, action)


class NormalPolicy(Policy, nn.Module):
    def __init__(self, encoder, action_size, min_logstd, max_logstd,
                 use_std_parameter):
        super().__init__()
        self.action_size = action_size
        self.encoder = encoder
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.use_std_parameter = use_std_parameter
        self.mu = nn.Linear(encoder.get_feature_size(), action_size)
        if self.use_std_parameter:
            initial_logstd = torch.zeros(1, action_size, dtype=torch.float32)
            self.logstd = nn.Parameter(initial_logstd)
        else:
            self.logstd = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        if self.use_std_parameter:
            clipped_logstd = self.get_logstd_parameter()
        else:
            logstd = self.logstd(h)
            clipped_logstd = logstd.clamp(self.min_logstd, self.max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def forward(self, x, deterministic=False, with_log_prob=False):
        if deterministic:
            # to avoid errors at ONNX export because broadcast_tensors in
            # Normal distribution is not supported by ONNX
            action = self.mu(self.encoder(x))
        else:
            dist = self.dist(x)
            action = dist.rsample()

        if with_log_prob:
            return squash_action(dist, action)

        return torch.tanh(action)

    def sample(self, x, with_log_prob=False):
        return self.forward(x, with_log_prob=with_log_prob)

    def sample_n(self, x, n, with_log_prob=False):
        dist = self.dist(x)

        action = dist.rsample((n, ))

        squashed_action_T, log_prob_T = squash_action(dist, action)

        # (n, batch, action) -> (batch, n, action)
        squashed_action = squashed_action_T.transpose(0, 1)
        # (n, batch, 1) -> (batch, n, 1)
        log_prob = log_prob_T.transpose(0, 1)

        if with_log_prob:
            return squashed_action, log_prob

        return squashed_action

    def best_action(self, x):
        return self.forward(x, deterministic=True, with_log_prob=False)

    def get_logstd_parameter(self):
        assert self.use_std_parameter
        logstd = torch.sigmoid(self.logstd)
        base_logstd = self.max_logstd - self.min_logstd
        return self.min_logstd + logstd * base_logstd


class CategoricalPolicy(Policy, nn.Module):
    def __init__(self, encoder, action_size):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(self, x):
        h = self.encoder(x)
        h = self.fc(h)
        return Categorical(torch.softmax(h, dim=1))

    def forward(self, x, deterministic=False, with_log_prob=False):
        dist = self.dist(x)

        if deterministic:
            action = dist.probs.argmax(dim=1)
        else:
            action = dist.sample()

        if with_log_prob:
            return action, dist.log_prob(action)

        return action

    def sample(self, x, with_log_prob=False):
        return self.forward(x, with_log_prob=with_log_prob)

    def sample_n(self, x, n, with_log_prob=False):
        dist = self.dist(x)

        action_T = dist.sample((n, ))
        log_prob_T = dist.log_prob(action_T)

        # (n, batch) -> (batch, n)
        action = action_T.transpose(0, 1)
        # (n, batch) -> (batch, n)
        log_prob = log_prob_T.transpose(0, 1)

        if with_log_prob:
            return action, log_prob

        return action

    def best_action(self, x):
        return self.forward(x, deterministic=True)

    def log_probs(self, x):
        dist = self.dist(x)
        return dist.logits
