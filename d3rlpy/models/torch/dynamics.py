import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, cast
from torch.distributions import Normal
from torch.nn.utils import spectral_norm
from .encoders import EncoderWithAction


def _compute_ensemble_variance(
    observations: torch.Tensor,
    rewards: torch.Tensor,
    variances: torch.Tensor,
    variance_type: str,
) -> torch.Tensor:
    if variance_type == "max":
        return variances.max(dim=1).values
    elif variance_type == "data":
        data = torch.cat([observations, rewards], dim=2)
        return (data.std(dim=1) ** 2).sum(dim=1, keepdim=True)
    raise ValueError("invalid variance_type.")


def _apply_spectral_norm_recursively(model: nn.Module) -> None:
    for _, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for i in range(len(module)):
                _apply_spectral_norm_recursively(module[i])
        else:
            spectral_norm(module)


def _gaussian_likelihood(
    x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor
) -> torch.Tensor:
    inv_std = torch.exp(-logstd)
    return (((mu - x) ** 2) * inv_std).mean(dim=1, keepdim=True)


class ProbablisticDynamics(nn.Module):
    """Probablistic dynamics model.

    References:
        * `Janner et al., When to Trust Your Model: Model-Based Policy
          Optimization. <https://arxiv.org/abs/1906.08253>`_
        * `Chua et al., Deep Reinforcement Learning in a Handful of Trials
          using Probabilistic Dynamics Models.
          <https://arxiv.org/abs/1805.12114>`_

    """

    _encoder: EncoderWithAction
    _mu: nn.Linear
    _logstd: nn.Linear
    _max_logstd: nn.Parameter
    _min_logstd: nn.Parameter

    def __init__(self, encoder: EncoderWithAction):
        super().__init__()
        # apply spectral normalization except logstd encoder.
        _apply_spectral_norm_recursively(cast(nn.Module, encoder))
        self._encoder = encoder

        feature_size = encoder.get_feature_size()
        observation_size = encoder.observation_shape[0]
        out_size = observation_size + 1

        # TODO: handle image observation
        self._mu = spectral_norm(nn.Linear(feature_size, out_size))
        self._logstd = nn.Linear(feature_size, out_size)

        # logstd bounds
        init_max = torch.empty(1, out_size, dtype=torch.float32).fill_(2.0)
        init_min = torch.empty(1, out_size, dtype=torch.float32).fill_(-10.0)
        self._max_logstd = nn.Parameter(init_max)
        self._min_logstd = nn.Parameter(init_min)

    def compute_stats(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encoder(x, action)

        mu = self._mu(h)

        # log standard deviation with bounds
        logstd = self._logstd(h)
        logstd = self._max_logstd - F.softplus(self._max_logstd - logstd)
        logstd = self._min_logstd + F.softplus(logstd - self._min_logstd)

        return mu, logstd

    def forward(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_with_variance(x, action)[:2]

    def predict_with_variance(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd = self.compute_stats(x, action)
        dist = Normal(mu, logstd.exp())
        pred = dist.rsample()
        # residual prediction
        next_x = x + pred[:, :-1]
        next_reward = pred[:, -1].view(-1, 1)
        return next_x, next_reward, dist.variance.sum(dim=1, keepdims=True)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        obs_tp1: torch.Tensor,
    ) -> torch.Tensor:
        mu, logstd = self.compute_stats(obs_t, act_t)

        # residual prediction
        mu_x = obs_t + mu[:, :-1]
        mu_reward = mu[:, -1].view(-1, 1)
        logstd_x = logstd[:, :-1]
        logstd_reward = logstd[:, -1].view(-1, 1)

        # gaussian likelihood loss
        likelihood_loss = _gaussian_likelihood(obs_tp1, mu_x, logstd_x)
        likelihood_loss += _gaussian_likelihood(
            rew_tp1, mu_reward, logstd_reward
        )

        # penalty to minimize standard deviation
        penalty = logstd.sum(dim=1, keepdim=True)

        # minimize logstd bounds
        bound_loss = self._max_logstd.sum() - self._min_logstd.sum()

        loss = likelihood_loss + penalty + 1e-2 * bound_loss

        return loss.view(-1, 1)


class EnsembleDynamics(nn.Module):
    _models: nn.ModuleList

    def __init__(self, models: List[ProbablisticDynamics]):
        super().__init__()
        self._models = nn.ModuleList(models)

    def forward(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_with_variance(x, action)[:2]

    def predict_with_variance(
        self, x: torch.Tensor, action: torch.Tensor, variance_type: str = "data"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations_list: List[torch.Tensor] = []
        rewards_list: List[torch.Tensor] = []
        variances_list: List[torch.Tensor] = []

        # predict next observation and reward
        for model in self._models:
            obs, rew, var = model.predict_with_variance(x, action)
            observations_list.append(obs.view(1, x.shape[0], -1))
            rewards_list.append(rew.view(1, x.shape[0], 1))
            variances_list.append(var.view(1, x.shape[0], 1))

        all_observations = torch.cat(observations_list, dim=0).transpose(0, 1)
        all_rewards = torch.cat(rewards_list, dim=0).transpose(0, 1)
        variances = torch.cat(variances_list, dim=0).transpose(0, 1)

        # uniformly sample from ensemble outputs
        indices = torch.randint(0, len(self._models), size=(x.shape[0],))
        observations = all_observations[torch.arange(x.shape[0]), indices]
        rewards = all_rewards[torch.arange(x.shape[0]), indices]

        variances = _compute_ensemble_variance(
            all_observations, all_rewards, variances, variance_type
        )

        return observations, rewards, variances

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        obs_tp1: torch.Tensor,
    ) -> torch.Tensor:
        loss_sum = torch.tensor(0.0, dtype=torch.float32, device=obs_t.device)
        for i, model in enumerate(self._models):
            # bootstrapping
            loss = model.compute_error(obs_t, act_t, rew_tp1, obs_tp1)
            assert loss.shape == (obs_t.shape[0], 1)
            mask = torch.randint(0, 2, size=loss.shape, device=obs_t.device)
            loss_sum += (loss * mask).mean()

        return loss_sum

    @property
    def models(self) -> nn.ModuleList:
        return self._models
