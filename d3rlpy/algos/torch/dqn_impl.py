import numpy as np
import torch
import copy

from d3rlpy.models.torch.q_functions import create_discrete_q_function
from .utility import hard_sync
from .utility import torch_api, train_api, eval_api
from .base import TorchImplBase


class DQNImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, learning_rate,
                 optim_factory, encoder_factory, q_func_factory, gamma,
                 n_critics, bootstrap, share_encoder, use_gpu, scaler,
                 augmentation):
        super().__init__(observation_shape, action_size, scaler)
        self.learning_rate = learning_rate
        self.optim_factory = optim_factory
        self.encoder_factory = encoder_factory
        self.q_func_factory = q_func_factory
        self.gamma = gamma
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.augmentation = augmentation
        self.use_gpu = use_gpu

        # initialized in build
        self.q_func = None
        self.targ_q_func = None
        self.optim = None

    def build(self):
        # setup torch models
        self._build_network()

        # setup target network
        self.targ_q_func = copy.deepcopy(self.q_func)

        if self.use_gpu:
            self.to_gpu(self.use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()

    def _build_network(self):
        self.q_func = create_discrete_q_function(
            self.observation_shape,
            self.action_size,
            self.encoder_factory,
            self.q_func_factory,
            n_ensembles=self.n_critics,
            bootstrap=self.bootstrap,
            share_encoder=self.share_encoder)

    def _build_optim(self):
        self.optim = self.optim_factory.create(self.q_func.parameters(),
                                               lr=self.learning_rate)

    @train_api
    @torch_api(scaler_targets=['obs_t', 'obs_tpn'])
    def update(self, obs_t, act_t, rew_tpn, obs_tpn, ter_tpn, n_steps):
        q_tpn = self.augmentation.process(func=self.compute_target,
                                          inputs={'x': obs_tpn},
                                          targets=['x'])
        q_tpn *= (1.0 - ter_tpn)

        loss = self.augmentation.process(func=self._compute_loss,
                                         inputs={
                                             'obs_t': obs_t,
                                             'act_t': act_t.long(),
                                             'rew_tpn': rew_tpn,
                                             'q_tpn': q_tpn,
                                             'n_steps': n_steps
                                         },
                                         targets=['obs_t'])

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.cpu().detach().numpy()

    def _compute_loss(self, obs_t, act_t, rew_tpn, q_tpn, n_steps):
        return self.q_func.compute_error(obs_t, act_t, rew_tpn, q_tpn,
                                         self.gamma**n_steps)

    def compute_target(self, x):
        with torch.no_grad():
            max_action = self.targ_q_func(x).argmax(dim=1)
            return self.targ_q_func.compute_target(x, max_action)

    def _predict_best_action(self, x):
        return self.q_func(x).argmax(dim=1)

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

    def sample_action(self, x):
        return self.predict_best_action(x)

    def update_target(self):
        hard_sync(self.targ_q_func, self.q_func)


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, x):
        with torch.no_grad():
            action = self._predict_best_action(x)
            return self.targ_q_func.compute_target(x, action)
