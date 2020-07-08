import numpy as np
import torch.nn.functional as F
import torch
import copy

from torch.optim import Adam
from skbrl.models.torch.q_functions import create_discrete_q_function
from skbrl.algos.torch.base import TorchImplBase
from skbrl.algos.dqn import IDQNImpl
from skbrl.algos.torch.utility import hard_sync
from skbrl.algos.torch.utility import torch_api, train_api, eval_api


class DQNImpl(TorchImplBase, IDQNImpl):
    def __init__(self, observation_shape, action_size, learning_rate, gamma,
                 eps, use_batch_norm, distribution_type, use_gpu):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.distribution_type = distribution_type

        # setup torch models
        self._build_network()

        # setup target network
        self.targ_q_func = copy.deepcopy(self.q_func)

        self.device = 'cpu:0'
        if use_gpu:
            self.to_gpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()

    def _build_network(self):
        self.q_func = create_discrete_q_function(
            self.observation_shape,
            self.action_size,
            n_ensembles=1,
            use_batch_norm=self.use_batch_norm,
            distribution_type=self.distribution_type)

    def _build_optim(self):
        self.optim = Adam(self.q_func.parameters(),
                          lr=self.learning_rate,
                          eps=self.eps)

    @train_api
    @torch_api
    def update(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        q_tp1 = self.compute_target(obs_tp1) * (1.0 - ter_tp1)
        loss = self.q_func.compute_td(obs_t, act_t.long(), rew_tp1, q_tp1,
                                      self.gamma)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.cpu().detach().numpy()

    def compute_target(self, x):
        with torch.no_grad():
            max_action = self.targ_q_func(x).argmax(dim=1)
            return self.targ_q_func.compute_target(x, max_action)

    def _predict_best_action(self, x):
        return self.q_func(x).argmax(dim=1)

    @eval_api
    @torch_api
    def predict_value(self, x, action):
        assert x.shape[0] == action.shape[0]

        self.q_func.eval()
        with torch.no_grad():
            values = self.q_func(x).cpu().detach().numpy()

        rets = []
        for v, a in zip(values, action.view(-1).long().cpu().detach().numpy()):
            rets.append(v[a])

        return np.array(rets)

    def update_target(self):
        hard_sync(self.targ_q_func, self.q_func)


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, x):
        with torch.no_grad():
            action = self._predict_best_action(x)
            return self.targ_q_func.compute_target(x, action)
