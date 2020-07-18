import torch
import torch.nn as nn
import copy
import math

from torch.optim import Adam
from d3rlpy.models.torch.imitators import create_probablistic_regressor
from d3rlpy.models.torch.q_functions import compute_max_with_n_actions
from d3rlpy.algos.bear import IBEARImpl
from .utility import torch_api, train_api
from .sac_impl import SACImpl


def _gaussian_kernel(x, y, sigma):
    return (-((x - y)**2) / (2 * sigma**2)).exp()


class BEARImpl(SACImpl, IBEARImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, imitator_learning_rate,
                 temp_learning_rate, alpha_learning_rate, gamma, tau,
                 n_critics, bootstrap, initial_temperature, initial_alpha,
                 alpha_threshold, lam, n_action_samples, mmd_sigma, eps,
                 use_batch_norm, q_func_type, use_gpu, scaler):
        # imitator requires these parameters
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.imitator_learning_rate = imitator_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.initial_alpha = initial_alpha
        self.alpha_threshold = alpha_threshold
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.mmd_sigma = mmd_sigma
        self.use_batch_norm = use_batch_norm

        self._build_imitator()

        super().__init__(observation_shape, action_size, actor_learning_rate,
                         critic_learning_rate, temp_learning_rate, gamma, tau,
                         n_critics, bootstrap, initial_temperature, eps,
                         use_batch_norm, q_func_type, use_gpu, scaler)

        self._build_imitator_optim()
        self._build_alpha()
        self._build_alpha_optim()

    def _build_imitator(self):
        self.imitator = create_probablistic_regressor(self.observation_shape,
                                                      self.action_size,
                                                      self.use_batch_norm)

    def _build_imitator_optim(self):
        self.imitator_optim = Adam(self.imitator.parameters(),
                                   self.imitator_learning_rate,
                                   eps=self.eps)

    def _build_alpha(self):
        initial_val = math.log(self.initial_alpha)
        data = torch.full((1, 1), initial_val, device=self.device)
        self.log_alpha = nn.Parameter(data)

    def _build_alpha_optim(self):
        self.alpha_optim = Adam([self.log_alpha],
                                self.alpha_learning_rate,
                                eps=self.eps)

    @train_api
    @torch_api
    def update_actor(self, obs_t):
        if self.scaler:
            obs_t = self.scaler.transform(obs_t)

        action, log_prob = self.policy(obs_t, with_log_prob=True)
        entropy = self.log_temp.exp() * log_prob
        # mean is empirically better
        q_t = self.q_func(obs_t, action, 'mean')
        td_loss = (entropy - q_t).mean()

        mmd_loss = self._compute_mmd(obs_t)

        loss = td_loss + mmd_loss

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    @train_api
    @torch_api
    def update_imitator(self, obs_t, act_t):
        if self.scaler:
            obs_t = self.scaler.transform(obs_t)

        loss = self.imitator.compute_error(obs_t, act_t)

        self.imitator_optim.zero_grad()
        loss.backward()
        self.imitator_optim.step()

        return loss.cpu().detach().numpy()

    @train_api
    @torch_api
    def update_alpha(self, obs_t):
        if self.scaler:
            obs_t = self.scaler.transform(obs_t)

        loss = -self._compute_mmd(obs_t)

        self.alpha_optim.zero_grad()
        loss.backward()
        self.alpha_optim.step()

        cur_alpha = self.log_alpha.exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_alpha

    def _compute_mmd(self, x):
        with torch.no_grad():
            behavior_actions = self.imitator.sample_n(x, self.n_action_samples)
        policy_actions = self.policy.sample_n(x, self.n_action_samples)

        # (batch, n, action) -> (batch, n, 1, action)
        behavior_actions = behavior_actions.reshape(x.shape[0], -1, 1,
                                                    self.action_size)
        policy_actions = policy_actions.reshape(x.shape[0], -1, 1,
                                                self.action_size)
        # (batch, n, action) -> (batch, 1, n, action)
        behavior_actions_T = behavior_actions.reshape(x.shape[0], 1, -1,
                                                      self.action_size)
        policy_actions_T = policy_actions.reshape(x.shape[0], 1, -1,
                                                  self.action_size)

        # 1 / N^2 \sum k(a_\pi, a_\pi)
        inter_policy = _gaussian_kernel(policy_actions, policy_actions_T,
                                        self.mmd_sigma)
        mmd = inter_policy.sum(dim=1).sum(dim=1) / self.n_action_samples**2

        # 1 / N^2 \sum k(a_\beta, a_\beta)
        inter_data = _gaussian_kernel(behavior_actions, behavior_actions_T,
                                      self.mmd_sigma)
        mmd += inter_data.sum(dim=1).sum(dim=1) / self.n_action_samples**2

        # 2 / N^2 \sum k(a_\pi, a_\beta)
        distance = _gaussian_kernel(policy_actions, behavior_actions_T,
                                    self.mmd_sigma)
        mmd -= 2 * distance.sum(dim=1).sum(dim=1) / self.n_action_samples**2

        clipped_alpha = self.log_alpha.clamp(-5.0, 10.0).exp()

        return (clipped_alpha * (mmd - self.alpha_threshold)).sum(dim=1).mean()

    def compute_target(self, x):
        with torch.no_grad():
            # BCQ-like target computation
            actions, log_probs = self.policy.sample_n(x, self.n_action_samples,
                                                      True)
            values, indices = compute_max_with_n_actions(
                x, actions, self.targ_q_func, self.lam, True)

            # (batch, n, 1) -> (batch, 1)
            max_log_prob = log_probs[torch.arange(x.shape[0]), indices]

            return values - self.log_temp.exp() * max_log_prob
