import numpy as np
import torch
import copy

from d3rlpy.models.torch.q_functions import create_continuous_q_function
from d3rlpy.models.torch.policies import create_deterministic_policy
from .utility import soft_sync, torch_api
from .utility import train_api, eval_api
from .utility import compute_augmentation_mean
from .base import TorchImplBase


class DDPGImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, actor_optim_factory,
                 critic_optim_factory, actor_encoder_factory,
                 critic_encoder_factory, gamma, tau, n_critics, bootstrap,
                 share_encoder, reguralizing_rate, q_func_type, use_gpu,
                 scaler, augmentation, n_augmentations):
        super().__init__(observation_shape, action_size, scaler)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_optim_factory = actor_optim_factory
        self.critic_optim_factory = critic_optim_factory
        self.actor_encoder_factory = actor_encoder_factory
        self.critic_encoder_factory = critic_encoder_factory
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.reguralizing_rate = reguralizing_rate
        self.q_func_type = q_func_type
        self.augmentation = augmentation
        self.n_augmentations = n_augmentations
        self.use_gpu = use_gpu

        # initialized in build
        self.q_func = None
        self.policy = None
        self.targ_q_func = None
        self.targ_policy = None
        self.actor_optim = None
        self.critic_optim = None

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
            self.critic_encoder_factory,
            n_ensembles=self.n_critics,
            q_func_type=self.q_func_type,
            bootstrap=self.bootstrap,
            share_encoder=self.share_encoder)

    def _build_critic_optim(self):
        self.critic_optim = self.critic_optim_factory.create(
            self.q_func.parameters(), lr=self.critic_learning_rate)

    def _build_actor(self):
        self.policy = create_deterministic_policy(self.observation_shape,
                                                  self.action_size,
                                                  self.actor_encoder_factory)

    def _build_actor_optim(self):
        self.actor_optim = self.actor_optim_factory.create(
            self.policy.parameters(), lr=self.actor_learning_rate)

    @train_api
    @torch_api(scaler_targets=['obs_t', 'obs_tp1'])
    def update_critic(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        q_tp1 = compute_augmentation_mean(augmentation=self.augmentation,
                                          n_augmentations=self.n_augmentations,
                                          func=self.compute_target,
                                          inputs={'x': obs_tp1},
                                          targets=['x'])
        q_tp1 *= (1.0 - ter_tp1)

        loss = compute_augmentation_mean(augmentation=self.augmentation,
                                         n_augmentations=self.n_augmentations,
                                         func=self._compute_critic_loss,
                                         inputs={
                                             'obs_t': obs_t,
                                             'act_t': act_t,
                                             'rew_tp1': rew_tp1,
                                             'q_tp1': q_tp1
                                         },
                                         targets=['obs_t'])

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_critic_loss(self, obs_t, act_t, rew_tp1, q_tp1):
        return self.q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1,
                                         self.gamma)

    @train_api
    @torch_api(scaler_targets=['obs_t'])
    def update_actor(self, obs_t):
        loss = compute_augmentation_mean(augmentation=self.augmentation,
                                         n_augmentations=self.n_augmentations,
                                         func=self._compute_actor_loss,
                                         inputs={'obs_t': obs_t},
                                         targets=['obs_t'])

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
    @torch_api(scaler_targets=['x'])
    def predict_value(self, x, action, with_std):
        assert x.shape[0] == action.shape[0]

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
