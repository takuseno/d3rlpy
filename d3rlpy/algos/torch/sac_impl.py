import torch
import torch.nn as nn
import numpy as np
import math
import copy

from d3rlpy.models.torch.policies import create_normal_policy
from d3rlpy.models.torch.policies import create_categorical_policy
from d3rlpy.models.torch.q_functions import create_discrete_q_function
from .utility import torch_api, train_api, eval_api, hard_sync
from .utility import compute_augmentation_mean
from .ddpg_impl import DDPGImpl
from .base import TorchImplBase


class SACImpl(DDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, temp_learning_rate, actor_optim_factory,
                 critic_optim_factory, temp_optim_factory,
                 actor_encoder_factory, critic_encoder_factory, q_func_factory,
                 gamma, tau, n_critics, bootstrap, share_encoder,
                 initial_temperature, use_gpu, scaler, augmentation,
                 n_augmentations):
        super().__init__(observation_shape=observation_shape,
                         action_size=action_size,
                         actor_learning_rate=actor_learning_rate,
                         critic_learning_rate=critic_learning_rate,
                         actor_optim_factory=actor_optim_factory,
                         critic_optim_factory=critic_optim_factory,
                         actor_encoder_factory=actor_encoder_factory,
                         critic_encoder_factory=critic_encoder_factory,
                         q_func_factory=q_func_factory,
                         gamma=gamma,
                         tau=tau,
                         n_critics=n_critics,
                         bootstrap=bootstrap,
                         share_encoder=share_encoder,
                         reguralizing_rate=0.0,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation,
                         n_augmentations=n_augmentations)
        self.temp_learning_rate = temp_learning_rate
        self.temp_optim_factory = temp_optim_factory
        self.initial_temperature = initial_temperature

        # initialized in build
        self.log_temp = None
        self.temp_optim = None

    def build(self):
        super().build()
        # TODO: save and load temperature parameter
        # setup temeprature after device property is set.
        self._build_temperature()
        self._build_temperature_optim()

    def _build_actor(self):
        self.policy = create_normal_policy(self.observation_shape,
                                           self.action_size,
                                           self.actor_encoder_factory)

    def _build_temperature(self):
        initial_val = math.log(self.initial_temperature)
        data = torch.full((1, 1), initial_val, device=self.device)
        self.log_temp = nn.Parameter(data)

    def _build_temperature_optim(self):
        self.temp_optim = self.temp_optim_factory.create(
            [self.log_temp], lr=self.temp_learning_rate)

    def _compute_actor_loss(self, obs_t):
        action, log_prob = self.policy(obs_t, with_log_prob=True)
        entropy = self.log_temp.exp() * log_prob
        q_t = self.q_func(obs_t, action, 'min')
        return (entropy - q_t).mean()

    @train_api
    @torch_api(scaler_targets=['obs_t'])
    def update_temp(self, obs_t):
        with torch.no_grad():
            _, log_prob = self.policy.sample(obs_t, with_log_prob=True)
            targ_temp = log_prob - self.action_size

        loss = -(self.log_temp.exp() * targ_temp).mean()

        self.temp_optim.zero_grad()
        loss.backward()
        self.temp_optim.step()

        # current temperature value
        cur_temp = self.log_temp.exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_temp

    @eval_api
    @torch_api(scaler_targets=['x'])
    def sample_action(self, x):
        with torch.no_grad():
            return self.policy.sample(x).cpu().detach().numpy()

    def compute_target(self, x):
        with torch.no_grad():
            action, log_prob = self.policy.sample(x, with_log_prob=True)
            entropy = self.log_temp.exp() * log_prob
            return self.targ_q_func.compute_target(x, action) - entropy


class DiscreteSACImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, temp_learning_rate, actor_optim_factory,
                 critic_optim_factory, temp_optim_factory,
                 actor_encoder_factory, critic_encoder_factory, q_func_factory,
                 gamma, n_critics, bootstrap, share_encoder,
                 initial_temperature, use_gpu, scaler, augmentation,
                 n_augmentations):
        super().__init__(observation_shape, action_size, scaler)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.temp_learning_rate = temp_learning_rate
        self.actor_optim_factory = actor_optim_factory
        self.critic_optim_factory = critic_optim_factory
        self.temp_optim_factory = temp_optim_factory
        self.actor_encoder_factory = actor_encoder_factory
        self.critic_encoder_factory = critic_encoder_factory
        self.q_func_factory = q_func_factory
        self.gamma = gamma
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.initial_temperature = initial_temperature
        self.use_gpu = use_gpu
        self.augmentation = augmentation
        self.n_augmentations = n_augmentations

    def build(self):
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

        # TODO: save and load temperature parameter
        # setup temeprature after device property is set.
        self._build_temperature()
        self._build_temperature_optim()

    def _build_critic(self):
        self.q_func = create_discrete_q_function(
            self.observation_shape,
            self.action_size,
            self.critic_encoder_factory,
            self.q_func_factory,
            n_ensembles=self.n_critics,
            bootstrap=self.bootstrap,
            share_encoder=self.share_encoder)

    def _build_critic_optim(self):
        self.critic_optim = self.critic_optim_factory.create(
            self.q_func.parameters(), lr=self.critic_learning_rate)

    def _build_actor(self):
        self.policy = create_categorical_policy(self.observation_shape,
                                                self.action_size,
                                                self.actor_encoder_factory)

    def _build_actor_optim(self):
        self.actor_optim = self.actor_optim_factory.create(
            self.policy.parameters(), lr=self.actor_learning_rate)

    def _build_temperature(self):
        initial_val = math.log(self.initial_temperature)
        data = torch.full((1, 1), initial_val, device=self.device)
        self.log_temp = nn.Parameter(data)

    def _build_temperature_optim(self):
        self.temp_optim = self.temp_optim_factory.create(
            [self.log_temp], lr=self.temp_learning_rate)

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
                                             'act_t': act_t.long(),
                                             'rew_tp1': rew_tp1,
                                             'q_tp1': q_tp1
                                         },
                                         targets=['obs_t'])

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.cpu().detach().numpy()

    def compute_target(self, x):
        with torch.no_grad():
            log_probs = self.policy.log_probs(x)
            probs = log_probs.exp()
            entropy = self.log_temp.exp() * log_probs
            target = self.targ_q_func.compute_target(x)
            keepdims = True
            if target.ndim == 3:
                entropy = entropy.unsqueeze(-1)
                probs = probs.unsqueeze(-1)
                keepdims = False
            return (probs * (target - entropy)).sum(dim=1, keepdims=keepdims)

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
        with torch.no_grad():
            q_t = self.q_func(obs_t, reduction='min')
        log_probs = self.policy.log_probs(obs_t)
        probs = log_probs.exp()
        entropy = self.log_temp.exp() * log_probs
        return (probs * (entropy - q_t)).sum(dim=1).mean()

    @train_api
    @torch_api(scaler_targets=['obs_t'])
    def update_temp(self, obs_t):
        with torch.no_grad():
            log_probs = self.policy.log_probs(obs_t)
            probs = log_probs.exp()
            expct_log_probs = (probs * log_probs).sum(dim=1, keepdims=True)
            entropy_target = 0.98 * (-math.log(1 / self.action_size))
            targ_temp = expct_log_probs + entropy_target

        loss = -(self.log_temp.exp() * targ_temp).mean()

        self.temp_optim.zero_grad()
        loss.backward()
        self.temp_optim.step()

        # current temperature value
        cur_temp = self.log_temp.exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_temp

    @eval_api
    @torch_api(scaler_targets=['x'])
    def predict_value(self, x, action, with_std):
        assert x.shape[0] == action.shape[0]

        action = action.view(-1).long().cpu().detach().numpy()
        with torch.no_grad():
            values = self.q_func(x, reduction='none').cpu().detach().numpy()
            values = np.transpose(values, [1, 0, 2])

        mean_values = values.mean(axis=1)
        stds = np.std(values, axis=1)

        ret_values = []
        ret_stds = []
        for v, std, a in zip(mean_values, stds, action):
            ret_values.append(v[a])
            ret_stds.append(std[a])

        if with_std:
            return np.array(ret_values), np.array(ret_stds)

        return np.array(ret_values)

    def _predict_best_action(self, x):
        return self.policy.best_action(x)

    @eval_api
    @torch_api(scaler_targets=['x'])
    def sample_action(self, x):
        with torch.no_grad():
            return self.policy.sample(x).cpu().detach().numpy()

    def update_target(self):
        hard_sync(self.targ_q_func, self.q_func)
