import numpy as np
import torch.nn.functional as F
import torch
import copy

from torch.optim import Adam
from skbrl.models.torch.q_functions import create_discrete_q_function
from skbrl.algos.base import ImplBase
from skbrl.algos.torch.utility import hard_sync, torch_api
from skbrl.algos.torch.utility import train_api, eval_api
from skbrl.algos.torch.utility import to_cuda, to_cpu
from skbrl.algos.torch.utility import freeze, unfreeze
from skbrl.algos.torch.utility import map_location


class DQNImpl(ImplBase):
    def __init__(self, observation_shape, action_size, learning_rate, gamma,
                 eps, use_batch_norm, use_gpu):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.use_batch_norm = use_batch_norm

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
        self.q_func = create_discrete_q_function(self.observation_shape,
                                                 self.action_size, 1,
                                                 self.use_batch_norm)

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
            return self.targ_q_func(x).max(dim=1, keepdim=True).values

    def _predict_best_action(self, x):
        return self.q_func(x).argmax(dim=1)

    @eval_api
    @torch_api
    def predict_best_action(self, x):
        with torch.no_grad():
            return self._predict_best_action(x).cpu().detach().numpy()

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

    def save_model(self, fname):
        torch.save(
            {
                'q_func': self.q_func.state_dict(),
                'optim': self.optim.state_dict(),
            }, fname)

    def load_model(self, fname):
        chkpt = torch.load(fname, map_location=map_location(self.device))
        self.q_func.load_state_dict(chkpt['q_func'])
        self.optim.load_state_dict(chkpt['optim'])
        self.update_target()

    @eval_api
    def save_policy(self, fname):
        dummy_x = torch.rand(1, *self.observation_shape, device=self.device)

        # workaround until version 1.6
        freeze(self)

        # dummy function to select best actions
        def _func(x):
            return self._predict_best_action(x)

        traced_script = torch.jit.trace(_func, dummy_x)
        traced_script.save(fname)

        # workaround until version 1.6
        unfreeze(self)

    def to_gpu(self):
        to_cuda(self)
        self.device = 'cuda:0'

    def to_cpu(self):
        to_cpu(self)
        self.device = 'cpu:0'


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, x):
        with torch.no_grad():
            action = self._predict_best_action(x)
            one_hot = F.one_hot(action.view(-1), num_classes=self.action_size)
            q_tp1 = (self.targ_q_func(x) * one_hot)
            return q_tp1.max(dim=1, keepdims=True).values
