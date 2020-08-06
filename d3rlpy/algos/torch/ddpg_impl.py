import numpy as np
import torch
import copy

from torch.optim import Adam
from d3rlpy.models.torch.q_functions import create_continuous_q_function
from d3rlpy.models.torch.policies import create_deterministic_policy
from .utility import soft_sync, torch_api
from .utility import train_api, eval_api
from .utility import compute_augemtation_mean
from .base import TorchImplBase


class DDPGImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, gamma, tau, n_critics, bootstrap,
                 share_encoder, reguralizing_rate, eps, use_batch_norm,
                 q_func_type, use_gpu, scaler, augmentation, n_augmentations,
                 encoder_params):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.reguralizing_rate = reguralizing_rate
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.scaler = scaler
        self.augmentation = augmentation
        self.n_augmentations = n_augmentations
        self.encoder_params = encoder_params
        self.use_gpu = use_gpu

    def build(self):
        # setup torch models
        self._build_critic()
        self._build_actor()

        # setup target networks
        self.targ_q_func = copy.deepcopy(self.q_func)
        self.targ_policy = copy.deepcopy(self.policy)

        if self.use_gpu:
            self.to_gpu(self.use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()

    def _build_critic(self):
        self.q_func = create_continuous_q_function(
            self.observation_shape,
            self.action_size,
            n_ensembles=self.n_critics,
            use_batch_norm=self.use_batch_norm,
            q_func_type=self.q_func_type,
            bootstrap=self.bootstrap,
            share_encoder=self.share_encoder,
            encoder_params=self.encoder_params)

    def _build_critic_optim(self):
        self.critic_optim = Adam(self.q_func.parameters(),
                                 lr=self.critic_learning_rate,
                                 eps=self.eps)

    def _build_actor(self):
        self.policy = create_deterministic_policy(
            self.observation_shape,
            self.action_size,
            self.use_batch_norm,
            encoder_params=self.encoder_params)

    def _build_actor_optim(self):
        self.actor_optim = Adam(self.policy.parameters(),
                                lr=self.actor_learning_rate,
                                eps=self.eps)

    @train_api
    @torch_api
    def update_critic(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        if self.scaler:
            obs_t = self.scaler.transform(obs_t)
            obs_tp1 = self.scaler.transform(obs_tp1)

        q_tp1 = compute_augemtation_mean(self.augmentation,
                                         self.n_augmentations,
                                         self.compute_target, {'x': obs_tp1},
                                         ['x'])
        q_tp1 *= (1.0 - ter_tp1)

        loss = compute_augemtation_mean(self.augmentation,
                                        self.n_augmentations,
                                        self._compute_critic_loss, {
                                            'obs_t': obs_t,
                                            'act_t': act_t,
                                            'rew_tp1': rew_tp1,
                                            'q_tp1': q_tp1
                                        }, ['obs_t'])

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_critic_loss(self, obs_t, act_t, rew_tp1, q_tp1):
        return self.q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1,
                                         self.gamma)

    @train_api
    @torch_api
    def update_actor(self, obs_t):
        if self.scaler:
            obs_t = self.scaler.transform(obs_t)

        loss = compute_augemtation_mean(self.augmentation,
                                        self.n_augmentations,
                                        self._compute_actor_loss,
                                        {'obs_t': obs_t}, ['obs_t'])

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_actor_loss(self, obs_t):
        action, raw_action = self.policy(obs_t, with_raw=True)
        q_t = self.q_func(obs_t, action, 'min')
        penalty = self.reguralizing_rate * (raw_action**2).mean()
        return -q_t.mean() + penalty

    def compute_target(self, x):
        with torch.no_grad():
            action = self.targ_policy(x)
            return self.targ_q_func.compute_target(x, action.clamp(-1.0, 1.0))

    def _predict_best_action(self, x):
        return self.policy.best_action(x)

    @eval_api
    @torch_api
    def predict_value(self, x, action, with_std):
        assert x.shape[0] == action.shape[0]

        if self.scaler:
            x = self.scaler.transform(x)

        with torch.no_grad():
            values = self.q_func(x, action, 'none').cpu().detach().numpy()
            values = np.transpose(values, [1, 0, 2])

        mean_values = values.mean(axis=1).reshape(-1)
        stds = np.std(values, axis=1).reshape(-1)

        if with_std:
            return mean_values, stds

        return mean_values

    def sample_action(self, x):
        return self.predict_best_action(x)

    def update_critic_target(self):
        soft_sync(self.targ_q_func, self.q_func, self.tau)

    def update_actor_target(self):
        soft_sync(self.targ_policy, self.policy, self.tau)
