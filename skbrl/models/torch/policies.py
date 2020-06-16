import torch
import torch.nn as nn
import math

from torch.distributions import Normal, Categorical


def _squash_action(dist, raw_action):
    squashed_action = torch.relu(raw_action)
    jacob = 2 * (math.log(2) - raw_action - torch.softplus(-2 * raw_action))
    log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=1, keepdims=True)
    return squashed_action, log_prob


class DeterministicPolicy(nn.Module):
    def __init__(self, head, action_size):
        super().__init__()
        self.head = head
        self.fc = nn.Linear(head.feature_size, action_size)

    def forward(self, x, without_tanh=False):
        h = self.head(x)
        h = self.fc(h)

        if without_tanh:
            return h

        return torch.tanh(h)

    def best_action(self, x):
        return self.forward(x)


class NormalPolicy(nn.Module):
    def __init__(self, head, action_size):
        super().__init__()
        self.head = head
        self.mu = nn.Linear(head.feature_size, action_size)
        self.logstd = nn.Linear(head.feature_size, action_size)

    def dist(self, x):
        h = self.head(x)
        mu = self.mu(h)
        logstd = self.logstd(h)
        clipped_logstd = logstd.clamp(-20.0, 2)
        return Normal(mu, clipped_logstd.exp())

    def forward(self, x, deterministic=False, with_log_prob=False):
        dist = self.dist(x)

        if deterministic:
            action = dist.mean

        else:
            action = dist.rsample()

        squashed_action, log_prob = _squash_action(dist, action)

        if with_log_prob:
            return squashed_action, log_prob

        return squashed_action

    def sample(self, x):
        return self.forward(x)

    def best_action(self, x):
        return self.forward(x, deterministic=True)


class CategoricalPolicy(nn.Module):
    def __init__(self, head, action_size):
        super().__init__()
        self.head = head
        self.fc = nn.Linear(head.feature_size, action_size)

    def dist(self, x):
        h = self.head(x)
        h = self.fc(h)
        return Categorical(torch.softmax(h))

    def forward(self, x, deterministic=False, with_log_prob=False):
        dist = self.dist(x)

        if deterministic:
            action = dist.probs().argmax(dim=1, keepdim=True)
        else:
            action = dist.sample()

        if with_log_prob:
            return action, dist.log_prob(action)

        return action

    def sample(self, x):
        return self.forward(x)

    def best_action(self, x):
        return self.forward(x, deterministic=True)
