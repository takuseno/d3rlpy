import numpy as np
import torch
import copy

from torch.optim import Adam
from d3rlpy.models.torch.q_functions import create_continuous_q_function
from d3rlpy.models.torch.q_functions import create_discrete_q_function
from d3rlpy.algos.torch.utility import torch_api, train_api, eval_api
from d3rlpy.algos.torch.utility import soft_sync, hard_sync
from d3rlpy.algos.torch.utility import compute_augemtation_mean
from d3rlpy.algos.torch.base import TorchImplBase


class FQEImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, learning_rate, gamma,
                 discrete_action, n_critics, bootstrap, share_encoder, eps,
                 use_batch_norm, q_func_type, use_gpu, scaler, augmentation,
                 n_augmentations, encoder_params):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.discrete_action = discrete_action
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.use_gpu = use_gpu
        self.scaler = scaler
        self.augmentation = augmentation
        self.n_augmentations = n_augmentations
        self.encoder_params = encoder_params

    def build(self):
        self._build_critic()

        self.targ_q_func = copy.deepcopy(self.q_func)

        if self.use_gpu:
            self.to_gpu(self.use_gpu)
        else:
            self.to_cpu()

        self._build_critic_optim()

    def _build_critic(self):
        if self.discrete_action:
            self.q_func = create_discrete_q_function(
                self.observation_shape,
                self.action_size,
                n_ensembles=self.n_critics,
                use_batch_norm=self.use_batch_norm,
                q_func_type=self.q_func_type,
                bootstrap=self.bootstrap,
                share_encoder=self.share_encoder,
                encoder_params=self.encoder_params)
        else:
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
                                 lr=self.learning_rate,
                                 eps=self.eps)

    @train_api
    @torch_api
    def update(self, obs_t, act_t, rew_tp1, act_tp1, obs_tp1, ter_tp1):
        if self.scaler:
            obs_t = self.scaler.transform(obs_t)
            obs_tp1 = self.scaler.transform(obs_tp1)

        if self.discrete_action:
            act_t = act_t.long()
            act_tp1 = act_tp1.long()

        q_tp1 = compute_augemtation_mean(self.augmentation,
                                         self.n_augmentations,
                                         self.compute_target, {
                                             'x': obs_tp1,
                                             'action': act_tp1
                                         }, ['x'])
        q_tp1 *= (1.0 - ter_tp1)

        loss = compute_augemtation_mean(self.augmentation,
                                        self.n_augmentations,
                                        self._compute_loss, {
                                            'obs_t': obs_t,
                                            'act_t': act_t,
                                            'rew_tp1': rew_tp1,
                                            'q_tp1': q_tp1
                                        }, ['obs_t'])

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_loss(self, obs_t, act_t, rew_tp1, q_tp1):
        return self.q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1,
                                         self.gamma)

    def compute_target(self, x, action):
        with torch.no_grad():
            return self.targ_q_func.compute_target(x, action)

    @eval_api
    @torch_api
    def predict_value(self, x, action, with_std):
        assert x.shape[0] == action.shape[0]

        if self.scaler:
            x = self.scaler.transform(x)

        with torch.no_grad():
            if self.discrete_action:
                values = self.q_func(x, 'none').cpu().detach().numpy()
            else:
                values = self.q_func(x, action, 'none').cpu().detach().numpy()
            values = np.transpose(values, [1, 0, 2])

        mean_values = values.mean(axis=1)
        stds = np.std(values, axis=1)

        if self.discrete_action:
            action = action.view(-1).long().cpu().detach().numpy()
            ret_values = []
            ret_stds = []
            for v, std, a in zip(mean_values, stds, action):
                ret_values.append(v[a])
                ret_stds.append(std[a])
            mean_values = np.array(ret_values)
            stds = np.array(ret_stds)
        else:
            mean_values = mean_values.reshape(-1)
            stds = stds.reshape(-1)

        if with_std:
            return mean_values, stds

        return mean_values

    def sample_action(self, x):
        raise NotImplementedError

    def _predict_best_action(self, x):
        raise NotImplementedError

    def update_target(self):
        hard_sync(self.targ_q_func, self.q_func)

    def save_policy(self):
        raise NotImplementedError
