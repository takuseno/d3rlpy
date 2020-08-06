import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.nn.utils import spectral_norm
from .encoders import create_encoder


def create_probablistic_dynamics(observation_shape,
                                 action_size,
                                 n_ensembles=5,
                                 use_batch_norm=False,
                                 discrete_action=False):
    models = []
    for _ in range(n_ensembles):
        encoder = create_encoder(observation_shape,
                                 action_size,
                                 use_batch_norm=use_batch_norm,
                                 discrete_action=discrete_action,
                                 activation_type='swish',
                                 hidden_units=[200, 200, 200, 200])
        model = ProbablisticDynamics(encoder)
        models.append(model)
    return EnsembleDynamics(models)


def _compute_ensemble_variance(observations, rewards, variances,
                               variance_type):
    if variance_type == 'max':
        return variances.max(dim=1).values
    elif variance_type == 'data':
        data = torch.cat([observations, rewards], dim=2)
        return (data.std(dim=1)**2).sum(dim=1, keepdims=True)
    raise ValueError('invalid variance_type.')


class EnsembleDynamics(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, action, with_variance=False, variance_type='data'):
        observations = []
        rewards = []
        variances = []

        # predict next observation and reward
        for model in self.models:
            observation, reward, variance = model(x, action, True)
            observations.append(observation.view(1, x.shape[0], -1))
            rewards.append(reward.view(1, x.shape[0], 1))
            variances.append(variance.view(1, x.shape[0], 1))

        all_observations = torch.cat(observations, dim=0).transpose(0, 1)
        all_rewards = torch.cat(rewards, dim=0).transpose(0, 1)
        variances = torch.cat(variances, dim=0).transpose(0, 1)

        # uniformly sample from ensemble outputs
        indices = torch.randint(0, len(self.models), size=(x.shape[0], ))
        observations = all_observations[torch.arange(x.shape[0]), indices]
        rewards = all_rewards[torch.arange(x.shape[0]), indices]

        if with_variance:
            variances = _compute_ensemble_variance(all_observations,
                                                   all_rewards, variances,
                                                   variance_type)
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


def _apply_spectral_norm_recursively(model):
    for _, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for i in range(len(module)):
                _apply_spectral_norm_recursively(module[i])
        else:
            spectral_norm(module)


def _gaussian_likelihood(x, mu, logstd):
    inv_std = torch.exp(-logstd)
    return (((mu - x)**2) * inv_std).mean(dim=1, keepdims=True)


class ProbablisticDynamics(nn.Module):
    """ Probablistic dynamics model.

    References:
        * `Janner et al., When to Trust Your Model: Model-Based Policy
          Optimization. <https://arxiv.org/abs/1906.08253>`_
        * `Chua et al., Deep Reinforcement Learning in a Handful of Trials
          using Probabilistic Dynamics Models.
          <https://arxiv.org/abs/1805.12114>`_

    """
    def __init__(self, encoder):
        super().__init__()
        # apply spectral normalization except logstd encoder.
        _apply_spectral_norm_recursively(encoder)
        self.encoder = encoder

        feature_size = encoder.feature_size
        observation_size = encoder.observation_shape[0]
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
        h = self.encoder(x, action)

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
        # residual prediction
        next_x = x + pred[:, :-1]
        next_reward = pred[:, -1].view(-1, 1)
        if with_variance:
            return next_x, next_reward, dist.variance.sum(dim=1, keepdims=True)
        return next_x, next_reward

    def compute_error(self, obs_t, act_t, rew_tp1, obs_tp1):
        mu, logstd = self.compute_stats(obs_t, act_t)

        # residual prediction
        mu_x = obs_t + mu[:, :-1]
        mu_reward = mu[:, -1].view(-1, 1)
        logstd_x = logstd[:, :-1]
        logstd_reward = logstd[:, -1].view(-1, 1)

        # gaussian likelihood loss
        likelihood_loss = _gaussian_likelihood(obs_tp1, mu_x, logstd_x)
        likelihood_loss += _gaussian_likelihood(rew_tp1, mu_reward,
                                                logstd_reward)

        # penalty to minimize standard deviation
        penalty = logstd.sum(dim=1, keepdims=True)

        # minimize logstd bounds
        bound_loss = self.max_logstd.sum() - self.min_logstd.sum()

        loss = likelihood_loss + penalty + 1e-2 * bound_loss

        return loss.view(-1, 1)
