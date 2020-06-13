import numpy as np
import torch.nn.functional as F
import torch
import copy

from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_
from skbrl.models.torch.heads import PixelHead, VectorHead
from skbrl.models.torch.q_functions import DiscreteQFunction
from skbrl.algos.base import ImplBase


class DQNImpl(ImplBase):
    def __init__(self, observation_shape, action_size, learning_rate, gamma,
                 alpha, eps, grad_clip, use_batch_norm, use_gpu):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.grad_clip = grad_clip

        # parametric functions
        if len(observation_shape) == 1:
            self.head = VectorHead(observation_shape[0], use_batch_norm)
        else:
            self.head = PixelHead(observation_shape[0], action_size,
                                  use_batch_norm)
        self.q_func = DiscreteQFunction(self.head, action_size)
        self.targ_q_func = copy.deepcopy(self.q_func)

        # optimizer
        self.optim = RMSprop(self.q_func.parameters(),
                             lr=learning_rate,
                             alpha=alpha,
                             eps=eps)

        self.device = 'cpu:0'
        if use_gpu:
            self.to_gpu()

    def update(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        self.q_func.train()
        device = self.q_func.device
        obs_t = torch.tensor(obs_t, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_t, dtype=torch.int32, device=device)
        rew_tp1 = torch.tensor(rew_tp1, dtype=torch.float32, device=device)
        obs_tp1 = torch.tensor(obs_tp1, dtype=torch.float32, device=device)
        ter_tp1 = torch.tensor(ter_tp1, dtype=torch.float32, device=device)

        q_tp1 = self.compute_target(obs_tp1) * (1.0 - ter_tp1)
        loss = self.q_func.compute_td(obs_t, act_t, rew_tp1, q_tp1, self.gamma)

        self.optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_func.parameters(), self.grad_clip)
        self.optim.step()

        return loss.cpu().detach().numpy()

    def compute_target(self, x):
        return self.targ_q_func(x).max(dim=1, keepdim=True).detach()

    def predict_best_action(self, x):
        self.q_func.eval()
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.q_func(x).argmax(dim=1).cpu().detach().numpy()

    def predict_value(self, x, action):
        assert x.shape[0] == action.shape[0]

        self.q_func.eval()
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = self.q_func(x).cpu().detach().numpy()

        rets = []
        for v, a in zip(values, action.reshape(-1)):
            rets.append(v[a])
        return np.array(rets)

    def update_target(self):
        with torch.no_grad():
            params = self.q_func.parameters()
            targ_params = self.targ_q_func.parameters()
            for p, p_targ in zip(params, targ_params):
                p_targ.data.copy_(p.data)

    def save_model(self, fname):
        torch.save(
            {
                'q_func': self.q_func.state_dict(),
                'optim': self.optim.state_dict(),
            }, fname)

    def load_model(self, fname):
        chkpt = torch.load(fname)
        self.q_func.load_state_dict(chkpt['q_func'])
        self.optim.load_state_dict(chkpt['optim'])
        self.update_target()

    def save_policy(self, fname):
        dummy_x = torch.rand(1, *self.observation_shape)

        # workaround until version 1.6
        self.q_func.eval()
        for p in self.q_func.parameters():
            p.requires_grad = False

        # dummy function to select best actions
        def _func(x):
            return self.q_func(x).argmax(dim=1)

        traced_script = torch.jit.trace(_func, dummy_x)
        traced_script.save(fname)

        for p in self.q_func.parameters():
            p.requires_grad = True

    def to_gpu(self):
        self.q_func.cuda()
        self.targ_q_func.cuda()
        self.device = 'cuda:0'

    def to_cpu(self):
        self.q_func.cpu()
        self.targ_q_func.cpu()
        self.device = 'cpu:0'


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, x):
        act = self.q_func(x).argmax(dim=1, keepdim=True)
        one_hot = F.one_hot(act.view(-1), num_classes=self.action_size)
        q_tp1 = (self.targ_q_func(x) * one_hot).max(dim=1, keepdims=True)
        return q_tp1.detach()
