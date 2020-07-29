import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.nn.utils import spectral_norm
from .heads import create_head


def create_probablistic_model(observation_shape,
                              action_size,
                              n_ensembles=5,
                              use_batch_norm=False):
    models = []
    for _ in range(n_ensembles):
        head = create_head(observation_shape,
                           action_size,
                           use_batch_norm=use_batch_norm)
        model = ProbablisticModel(head)
        models.append(model)
    return EnsembleDynamicsModel(models)


def _compute_ensemble_variance(observations, rewards, variances,
                               variance_type):
    if variance_type == 'max':
        return variances.max(dim=1).values
    elif variance_type == 'data':
        data = torch.cat([observations, rewards], dim=2)
        return (data.std(dim=1)**2).sum(dim=1, keepdims=True)
    raise ValueError('invalid variance_type.')


class EnsembleDynamicsModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, action, with_variance=False, variance_type='max'):
        observations = []
        rewards = []
        variances = []

        # predict next observation and reward
        for model in self.models:
            observation, reward, variance = model(x, action, True)
            observations.append(observation.view(1, x.shape[0], -1))
            rewards.append(reward.view(1, x.shape[0], 1))
            variances.append(variance.view(1, x.shape[0], 1))

        observations = torch.cat(observations, dim=0).transpose(0, 1)
        rewards = torch.cat(rewards, dim=0).transpose(0, 1)
        variances = torch.cat(variances, dim=0).transpose(0, 1)

        # uniformly sample from ensemble outputs
        indices = torch.randint(0, len(self.models), size=(x.shape[0], ))
        observations = observations[torch.arange(x.shape[0]), indices]
        rewards = rewards[torch.arange(x.shape[0]), indices]

        if with_variance:
            variances = _compute_ensemble_variance(observations, rewards,
                                                   variances, variance_type)
            return observations, rewards, variances

        return observations, rewards

    def compute_error(self, obs_t, act_t, rew_tp1, obs_tp1):
        loss_sum = 0.0
        for i, model in enumerate(self.models):
            # bootstrapping
            loss = model.compute_error(obs_t, act_t, rew_tp1, obs_tp1)
            assert loss.shape == (obs_t.shape[0], 1)
            mask = torch.randint(0, 2, size=loss.shape, device=obs_t.device)
            loss_sum += (loss * mask).mean()

        return loss_sum


class ProbablisticModel(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

        feature_size = head.feature_size
        observation_size = head.observation_shape[0]
        out_size = observation_size + 1

        # TODO: handle image observation
        self.mu = spectral_norm(nn.Linear(feature_size, out_size))
        self.logstd = nn.Linear(feature_size, out_size)

        # logstd bounds
        init_max = torch.empty(1, out_size, dtype=torch.float32).fill_(2.0)
        init_min = torch.empty(1, out_size, dtype=torch.float32).fill_(-10.0)
        self.max_logstd = nn.Parameter(init_max)
        self.min_logstd = nn.Parameter(init_min)

    def compute_stats(self, x, action):
        h = self.head(x, action)

        mu = self.mu(h)

        # log standard deviation with bounds
        logstd = self.logstd(h)
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)

        return mu, logstd

    def forward(self, x, action, with_variance=False):
        mu, logstd = self.compute_stats(x, action)
        dist = Normal(mu, logstd.exp())
        pred = dist.rsample()
        next_x = pred[:, :-1]
        next_reward = pred[:, -1].view(-1, 1)
        if with_variance:
            return next_x, next_reward, dist.variance.sum(dim=1, keepdims=True)
        return next_x, next_reward

    def compute_error(self, obs_t, act_t, rew_tp1, obs_tp1):
        mu, logstd = self.compute_stats(obs_t, act_t)
        inv_std = torch.exp(-logstd)

        y = torch.cat([obs_tp1, rew_tp1], dim=1)

        likelihood_loss = (((mu - y)**2) * inv_std).sum(dim=1)
        penalty = logstd.sum(dim=1)
        bound_loss = self.max_logstd.sum() - self.min_logstd.sum()
        loss = likelihood_loss + penalty + bound_loss / obs_t.shape[0]

        return loss.view(-1, 1)
