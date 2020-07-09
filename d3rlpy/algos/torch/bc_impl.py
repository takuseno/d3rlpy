import torch
import copy

from torch.optim import Adam
from d3rlpy.models.torch.imitators import create_deterministic_regressor
from d3rlpy.models.torch.imitators import create_discrete_imitator
from d3rlpy.algos.bc import IBCImpl
from .base import TorchImplBase
from .utility import torch_api, train_api


class BCImpl(TorchImplBase, IBCImpl):
    def __init__(self, observation_shape, action_size, learning_rate, eps,
                 use_batch_norm, use_gpu):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.eps = eps
        self.use_batch_norm = use_batch_norm

        self._build_network()

        self.device = 'cpu:0'
        if use_gpu:
            self.to_gpu()

        self._build_optim()

    def _build_network(self):
        self.imitator = create_deterministic_regressor(self.observation_shape,
                                                       self.action_size)

    def _build_optim(self):
        self.optim = Adam(self.imitator.parameters(),
                          lr=self.learning_rate,
                          eps=self.eps)

    @train_api
    @torch_api
    def update_imitator(self, obs_t, act_t):
        loss = self.imitator.compute_l2_loss(obs_t, act_t)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.cpu().detach().numpy()

    def _predict_best_action(self, x):
        return self.imitator(x)

    def predict_value(self, x, action):
        raise NotImplementedError('BC does not support value estimation')


class DiscreteBCImpl(BCImpl):
    def __init__(self, observation_shape, action_size, learning_rate, eps,
                 beta, use_batch_norm, use_gpu):
        self.beta = beta
        super().__init__(observation_shape, action_size, learning_rate, eps,
                         use_batch_norm, use_gpu)

    def _build_network(self):
        self.imitator = create_discrete_imitator(self.observation_shape,
                                                 self.action_size, self.beta)

    def _predict_best_action(self, x):
        return self.imitator(x).argmax(dim=1)

    @train_api
    @torch_api
    def update_imitator(self, obs_t, act_t):
        loss = self.imitator.compute_likelihood_loss(obs_t, act_t.long())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.cpu().detach().numpy()
