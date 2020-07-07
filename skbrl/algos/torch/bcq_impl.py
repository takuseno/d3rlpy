import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from torch.optim import Adam
from skbrl.models.torch.heads import PixelHead
from skbrl.models.torch.policies import create_deterministic_residual_policy
from skbrl.models.torch.q_functions import create_continuous_q_function
from skbrl.models.torch.imitators import create_conditional_vae
from skbrl.models.torch.imitators import create_discrete_imitator
from skbrl.models.torch.imitators import DiscreteImitator
from skbrl.algos.torch.utility import torch_api, train_api
from skbrl.algos.bcq import IBCQImpl
from .ddpg_impl import DDPGImpl
from .dqn_impl import DoubleDQNImpl


class BCQImpl(DDPGImpl, IBCQImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, imitator_learning_rate, gamma, tau,
                 n_critics, lam, n_action_samples, action_flexibility,
                 latent_size, beta, eps, use_batch_norm,
                 use_quantile_regression, use_gpu):
        # imitator requires these parameters
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.use_batch_norm = use_batch_norm
        self.eps = eps

        self.imitator_learning_rate = imitator_learning_rate
        self.n_critics = n_critics
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.action_flexibility = action_flexibility
        self.latent_size = latent_size
        self.beta = beta

        self._build_imitator()

        super().__init__(observation_shape, action_size, actor_learning_rate,
                         critic_learning_rate, gamma, tau, 0.0, eps,
                         use_batch_norm, use_quantile_regression, use_gpu)

        # setup optimizer after the parameters move to GPU
        self._build_imitator_optim()

    def _build_critic(self):
        self.q_func = create_continuous_q_function(
            self.observation_shape,
            self.action_size,
            n_ensembles=self.n_critics,
            use_batch_norm=self.use_batch_norm,
            use_quantile_regression=self.use_quantile_regression)

    def _build_actor(self):
        self.policy = create_deterministic_residual_policy(
            self.observation_shape, self.action_size, self.action_flexibility,
            self.use_batch_norm)

    def _build_imitator(self):
        self.imitator = create_conditional_vae(self.observation_shape,
                                               self.action_size,
                                               self.latent_size, self.beta,
                                               self.use_batch_norm)

    def _build_imitator_optim(self):
        self.imitator_optim = Adam(self.imitator.parameters(),
                                   self.imitator_learning_rate,
                                   eps=self.eps)

    @train_api
    @torch_api
    def update_actor(self, obs_t):
        latent = torch.randn(obs_t.shape[0],
                             self.latent_size,
                             device=self.device)
        clipped_latent = latent.clamp(-0.5, 0.5)
        sampled_action = self.imitator.decode(obs_t, clipped_latent)
        action = self.policy(obs_t, sampled_action)
        loss = -self.q_func(obs_t, action, 'min').mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    @train_api
    @torch_api
    def update_imitator(self, obs_t, act_t):
        loss = self.imitator.compute_likelihood_loss(obs_t, act_t)

        self.imitator_optim.zero_grad()
        loss.backward()
        self.imitator_optim.step()

        return loss.cpu().detach().numpy()

    def _repeat_observation(self, x):
        # (batch_size, *obs_shape) -> (batch_size, n, *obs_shape)
        repeat_shape = (x.shape[0], self.n_action_samples, *x.shape[1:])
        repeated_x = x.view(x.shape[0], 1, *x.shape[1:]).expand(repeat_shape)
        return repeated_x

    def _sample_action(self, repeated_x, target=False):
        # TODO: this seems to be slow with image observation
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # sample latent variable
        latent = torch.randn(flattened_x.shape[0],
                             self.latent_size,
                             device=self.device)
        clipped_latent = latent.clamp(-0.5, 0.5)
        # sample action
        sampled_action = self.imitator.decode(flattened_x, clipped_latent)
        # add residual action
        policy = self.targ_policy if target else self.policy
        action = policy(flattened_x, sampled_action)
        return action.view(-1, self.n_action_samples, self.action_size)

    def _predict_value(self, repeated_x, action, target=False):
        # TODO: this seems to be slow with image observation
        # (batch_size, n, *obs_shape) -> (batch_size * n, *obs_shape)
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # (batch_size, n, action_size) -> (batch_size * n, action_size)
        flattend_action = action.view(-1, self.action_size)
        # estimate values
        if target:
            return self.targ_q_func.compute_target(flattened_x,
                                                   flattend_action, 'none')
        return self.q_func(flattened_x, flattend_action, 'none')

    def _predict_best_action(self, x):
        # TODO: this seems to be slow with image observation
        repeated_x = self._repeat_observation(x)
        action = self._sample_action(repeated_x)
        values = self._predict_value(repeated_x, action)[0]
        # pick the best (batch_size * n) -> (batch_size,)
        index = values.view(-1, self.n_action_samples).argmax(dim=1)
        return action[torch.arange(action.shape[0]), index]

    def compute_target(self, x):
        # TODO: this seems to be slow with image observation
        with torch.no_grad():
            repeated_x = self._repeat_observation(x)
            action = self._sample_action(repeated_x, True)
            # estimate values (n_ensembles, batch_size * n, -1)
            # take care of quantile regression
            values = self._predict_value(repeated_x, action, target=True)
            # reshape to (n_ensembles, batch_size, n, -1)
            reshaped_values = values.view(self.n_critics, x.shape[0],
                                          self.n_action_samples, -1)

            # get combination indices
            # (n_ensembles, batch_size, n, -1) -> (batch_size, n_ensembles, n)
            mean_values = reshaped_values.mean(dim=3).transpose(0, 1)
            #(batch_size, n_ensembles, n) -> (batch_size, n)
            max_values, max_indices = mean_values.max(dim=1)
            min_values, min_indices = mean_values.min(dim=1)
            mix_values = (1.0 - self.lam) * max_values + self.lam * min_values
            #(batch_size, n) -> (batch_size,)
            mix_indices = mix_values.argmax(dim=1)

            # fuse maximum values and minimum values
            # (n_ensembles, batch, n, -1) -> (batch, n, n_ensembels, -1)
            transposed_values = reshaped_values.permute(1, 2, 0, 3)
            # (batch, n, n_ensembles, -1) -> (batch * n, n_ensembles, -1)
            bn = x.shape[0] * self.n_action_samples
            flatten_values = transposed_values.view(bn, self.n_critics, -1)
            # (batch * n, n_ensembles, -1) -> (batch * n, -1)
            bn_indices = torch.arange(bn)
            max_values = flatten_values[bn_indices, max_indices.view(-1)]
            min_values = flatten_values[bn_indices, min_indices.view(-1)]
            # (batch * n, -1) -> (batch, n, -1)
            max_values = max_values.view(x.shape[0], self.n_action_samples, -1)
            min_values = min_values.view(x.shape[0], self.n_action_samples, -1)
            mix_values = (1.0 - self.lam) * max_values + self.lam * min_values
            # (batch, n, -1) -> (batch, -1)
            return mix_values[torch.arange(x.shape[0]), mix_indices]


class DiscreteBCQImpl(DoubleDQNImpl):
    def __init__(self, observation_shape, action_size, learning_rate, gamma,
                 action_flexibility, beta, eps, use_batch_norm,
                 use_quantile_regression, use_gpu):

        self.action_flexibility = action_flexibility
        self.beta = beta

        super().__init__(observation_shape, action_size, learning_rate, gamma,
                         eps, use_batch_norm, use_quantile_regression, use_gpu)

    def _build_network(self):
        super()._build_network()
        # share convolutional layers if observation is pixel
        if isinstance(self.q_func.head, PixelHead):
            self.imitator = DiscreteImitator(self.q_func.head,
                                             self.action_size, self.beta)
        else:
            self.imitator = create_discrete_imitator(self.observation_shape,
                                                     self.action_size,
                                                     self.beta,
                                                     self.use_batch_norm)

    def _build_optim(self):
        q_func_params = list(self.q_func.parameters())
        imitator_params = list(self.imitator.parameters())
        # retrieve unique elements
        unique_params = list(set(q_func_params + imitator_params))
        self.optim = Adam(unique_params, lr=self.learning_rate, eps=self.eps)

    @train_api
    @torch_api
    def update(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        # convert float to long
        act_t = act_t.long()

        # loss for Q function
        q_tp1 = self.compute_target(obs_tp1) * (1.0 - ter_tp1)
        value_loss = self.q_func.compute_td(obs_t, act_t, rew_tp1, q_tp1,
                                            self.gamma)

        # loss for imitator
        imitator_loss = self.imitator.compute_likelihood_loss(obs_t, act_t)

        loss = value_loss + imitator_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        value_loss = value_loss.cpu().detach().numpy()
        imitator_loss = imitator_loss.cpu().detach().numpy()

        return value_loss, imitator_loss

    def _predict_best_action(self, x):
        log_probs = self.imitator(x)
        ratio = log_probs - log_probs.max(dim=1, keepdim=True).values
        mask = (ratio > math.log(self.action_flexibility)).float()
        value = self.q_func(x)
        normalized_value = value - value.min(dim=1, keepdim=True).values
        action = (normalized_value * mask).argmax(dim=1)
        return action
