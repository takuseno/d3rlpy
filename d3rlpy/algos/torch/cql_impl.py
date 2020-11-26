import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.optim import Adam
from .utility import torch_api, train_api
from .sac_impl import SACImpl
from .dqn_impl import DoubleDQNImpl


class CQLImpl(SACImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, temp_learning_rate, alpha_learning_rate,
                 actor_optim_factory, critic_optim_factory, temp_optim_factory,
                 alpha_optim_factory, actor_encoder_factory,
                 critic_encoder_factory, gamma, tau, n_critics, bootstrap,
                 share_encoder, initial_temperature, initial_alpha,
                 alpha_threshold, n_action_samples, q_func_type, use_gpu,
                 scaler, augmentation, n_augmentations):
        super().__init__(observation_shape=observation_shape,
                         action_size=action_size,
                         actor_learning_rate=actor_learning_rate,
                         critic_learning_rate=critic_learning_rate,
                         temp_learning_rate=temp_learning_rate,
                         actor_optim_factory=actor_optim_factory,
                         critic_optim_factory=critic_optim_factory,
                         temp_optim_factory=temp_optim_factory,
                         actor_encoder_factory=actor_encoder_factory,
                         critic_encoder_factory=critic_encoder_factory,
                         gamma=gamma,
                         tau=tau,
                         n_critics=n_critics,
                         bootstrap=bootstrap,
                         share_encoder=share_encoder,
                         initial_temperature=initial_temperature,
                         q_func_type=q_func_type,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation,
                         n_augmentations=n_augmentations)
        self.alpha_learning_rate = alpha_learning_rate
        self.alpha_optim_factory = alpha_optim_factory
        self.initial_alpha = initial_alpha
        self.alpha_threshold = alpha_threshold
        self.n_action_samples = n_action_samples

        # initialized in build
        self.log_alpha = None
        self.alpha_optim = None

    def build(self):
        super().build()
        self._build_alpha()
        self._build_alpha_optim()

    def _build_alpha(self):
        initial_val = math.log(self.initial_alpha)
        data = torch.full((1, 1), initial_val, device=self.device)
        self.log_alpha = nn.Parameter(data)

    def _build_alpha_optim(self):
        self.alpha_optim = self.alpha_optim_factory.create(
            [self.log_alpha], lr=self.alpha_learning_rate)

    def _compute_critic_loss(self, obs_t, act_t, rew_tp1, q_tp1):
        loss = super()._compute_critic_loss(obs_t, act_t, rew_tp1, q_tp1)
        conservative_loss = self._compute_conservative_loss(obs_t, act_t)
        return loss + conservative_loss

    @train_api
    @torch_api(scaler_targets=['obs_t'])
    def update_alpha(self, obs_t, act_t):
        loss = -self._compute_conservative_loss(obs_t, act_t)

        self.alpha_optim.zero_grad()
        loss.backward()
        self.alpha_optim.step()

        cur_alpha = self.log_alpha.exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_alpha

    def _compute_conservative_loss(self, obs_t, act_t):
        with torch.no_grad():
            policy_actions, n_log_probs = self.policy.sample_n(
                obs_t, self.n_action_samples, with_log_prob=True)

        repeated_obs_t = obs_t.expand(self.n_action_samples, *obs_t.shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs_t = repeated_obs_t.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs_t = transposed_obs_t.reshape(-1, *obs_t.shape[1:])
        # (batch, n, action) -> (batch * n, action)
        flat_policy_acts = policy_actions.reshape(-1, self.action_size)

        # estimate action-values for policy actions
        policy_values = self.q_func(flat_obs_t, flat_policy_acts, 'none')
        policy_values = policy_values.view(self.n_critics, obs_t.shape[0],
                                           self.n_action_samples, 1)
        log_probs = n_log_probs.view(1, -1, self.n_action_samples, 1)

        # estimate action-values for actions from uniform distribution
        # uniform distribution between [-1.0, 1.0]
        random_actions = torch.zeros_like(flat_policy_acts).uniform_(-1.0, 1.0)
        random_values = self.q_func(flat_obs_t, random_actions, 'none')
        random_values = random_values.view(self.n_critics, obs_t.shape[0],
                                           self.n_action_samples, 1)

        # get maximum value to avoid overflow
        base = torch.max(policy_values.max(), random_values.max()).detach()

        # compute logsumexp
        policy_meanexp = (policy_values - base - log_probs).exp().mean(dim=2)
        random_meanexp = (random_values - base).exp().mean(dim=2) / 0.5
        # small constant value seems to be necessary to avoid nan
        logsumexp = (0.5 * random_meanexp + 0.5 * policy_meanexp + 1e-10).log()
        logsumexp += base

        # estimate action-values for data actions
        data_values = self.q_func(obs_t, act_t, 'none')

        element_wise_loss = logsumexp - data_values - self.alpha_threshold

        # this clipping seems to stabilize training
        clipped_alpha = self.log_alpha.clamp(-10.0, 2.0).exp()

        return (clipped_alpha * element_wise_loss).sum(dim=0).mean()


class DiscreteCQLImpl(DoubleDQNImpl):
    def _compute_loss(self, obs_t, act_t, rew_tp1, q_tp1):
        loss = super()._compute_loss(obs_t, act_t, rew_tp1, q_tp1)
        conservative_loss = self._compute_conservative_loss(obs_t, act_t)
        return loss + conservative_loss

    def _compute_conservative_loss(self, obs_t, act_t):
        # compute logsumexp
        policy_values = self.q_func(obs_t)
        logsumexp = torch.logsumexp(policy_values, dim=1, keepdims=True)

        # estimate action-values under data distribution
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        data_values = (self.q_func(obs_t) * one_hot).sum(dim=1, keepdims=True)

        return (logsumexp - data_values).mean()
