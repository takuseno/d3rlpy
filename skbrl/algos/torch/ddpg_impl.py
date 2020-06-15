import numpy as np
import torch.nn.functional as F
import torch
import copy

from torch.optim import Adam
from skbrl.models.torch.heads import PixelHead, VectorHead
from skbrl.models.torch.heads import PixelHeadWithAction, VectorHeadWithAction
from skbrl.models.torch.q_functions import ContinuousQFunction
from skbrl.models.torch.policies import DeterministicPolicy
from skbrl.algos.base import ImplBase


class DDPGImpl(ImplBase):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, gamma, tau, eps, use_batch_norm,
                 use_gpu):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau

        # critic
        if len(observation_shape) == 1:
            self.critic_head = VectorHeadWithAction(observation_shape,
                                                    action_size,
                                                    use_batch_norm)
        else:
            self.critic_head = PixelHeadWithAction(observation_shape,
                                                   action_size, use_batch_norm)
        self.q_func = ContinuousQFunction(self.critic_head)
        self.targ_q_func = copy.deepcopy(self.q_func)

        # actor
        if len(observation_shape) == 1:
            self.actor_head = VectorHead(observation_shape, use_batch_norm)
        else:
            self.actor_head = PixelHead(observation_shape, use_batch_norm)
        self.policy = DeterministicPolicy(self.actor_head, action_size)
        self.targ_policy = copy.deepcopy(self.policy)

        # optimizer
        self.critic_optim = Adam(self.q_func.parameters(),
                                 lr=critic_learning_rate,
                                 eps=eps)
        self.actor_optim = Adam(self.policy.parameters(),
                                lr=actor_learning_rate,
                                eps=eps)

        self.device = 'cpu:0'
        if use_gpu:
            self.to_gpu()

    def update_critic(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        self.q_func.train()
        device = self.device
        obs_t = torch.tensor(obs_t, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_t, dtype=torch.int64, device=device)
        rew_tp1 = torch.tensor(rew_tp1, dtype=torch.float32, device=device)
        obs_tp1 = torch.tensor(obs_tp1, dtype=torch.float32, device=device)
        ter_tp1 = torch.tensor(ter_tp1, dtype=torch.float32, device=device)

        q_tp1 = self.compute_target(obs_tp1) * (1.0 - ter_tp1)
        loss = self.q_func.compute_td(obs_t, act_t, rew_tp1, q_tp1, self.gamma)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.cpu().detach().numpy()

    def update_actor(self, obs_t):
        self.policy.train()
        device = self.device
        obs_t = torch.tensor(obs_t, dtype=torch.float32, device=device)

        q_t = self.q_func(obs_t, self.policy(obs_t))
        loss = -q_t.mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    def compute_target(self, x):
        with torch.no_grad():
            action = self.targ_policy(x)
            return self.targ_q_func(x, action)

    def predict_best_action(self, x):
        self.policy.eval()
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.policy.best_action(x).cpu().detach().numpy()

    def predict_value(self, x, action):
        assert x.shape[0] == action.shape[0]

        self.q_func.eval()
        self.policy.eval()
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.q_func(x, action).view(-1).cpu().detach().numpy()

    def update_critic_target(self):
        with torch.no_grad():
            params = self.q_func.parameters()
            targ_params = self.targ_q_func.parameters()
            for p, p_targ in zip(params, targ_params):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)

    def update_actor_target(self):
        with torch.no_grad():
            params = self.policy.parameters()
            targ_params = self.targ_policy.parameters()
            for p, p_targ in zip(params, targ_params):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)

    def save_model(self, fname):
        torch.save(
            {
                'q_func': self.q_func.state_dict(),
                'policy': self.policy.state_dict(),
                'critic_optim': self.critic_optim.state_dict(),
                'actor_optim': self.actor_optim.state_dict(),
            }, fname)

    def load_model(self, fname):
        chkpt = torch.load(fname)
        self.q_func.load_state_dict(chkpt['q_func'])
        self.policy.load_state_dict(chkpt['policy'])
        self.critic_optim.load_state_dict(chkpt['critic_optim'])
        self.actor_optim.load_state_dict(chkpt['actor_optim'])
        self.targ_q_func = copy.deepcopy(self.q_func)
        self.targ_policy = copy.deepcopy(self.policy)

    def save_policy(self, fname):
        dummy_x = torch.rand(1, *self.observation_shape)

        # workaround until version 1.6
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False

        # dummy function to select best actions
        def _func(x):
            return self.policy.best_action(x)

        traced_script = torch.jit.trace(_func, dummy_x)
        traced_script.save(fname)

        for p in self.policy.parameters():
            p.requires_grad = True

    def to_gpu(self):
        self.q_func.cuda()
        self.targ_q_func.cuda()
        self.policy.cuda()
        self.targ_policy.cuda()
        self.device = 'cuda:0'

    def to_cpu(self):
        self.q_func.cpu()
        self.targ_q_func.cpu()
        self.policy.cpu()
        self.targ_policy.cpu()
        self.device = 'cpu:0'
