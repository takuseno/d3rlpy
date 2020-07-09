import torch
import copy

from torch.optim import Adam
from d3rlpy.models.torch.q_functions import create_continuous_q_function
from d3rlpy.models.torch.policies import create_deterministic_policy
from d3rlpy.algos.ddpg import IDDPGImpl
from .utility import soft_sync, torch_api
from .utility import train_api, eval_api
from .base import TorchImplBase


class DDPGImpl(TorchImplBase, IDDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, gamma, tau, reguralizing_rate, eps,
                 use_batch_norm, q_func_type, use_gpu):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.reguralizing_rate = reguralizing_rate
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type

        # setup torch models
        self._build_critic()
        self._build_actor()

        # setup target networks
        self.targ_q_func = copy.deepcopy(self.q_func)
        self.targ_policy = copy.deepcopy(self.policy)

        self.device = 'cpu:0'
        if use_gpu:
            self.to_gpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()

    def _build_critic(self):
        self.q_func = create_continuous_q_function(
            self.observation_shape,
            self.action_size,
            n_ensembles=1,
            use_batch_norm=self.use_batch_norm,
            q_func_type=self.q_func_type)

    def _build_critic_optim(self):
        self.critic_optim = Adam(self.q_func.parameters(),
                                 lr=self.critic_learning_rate,
                                 eps=self.eps)

    def _build_actor(self):
        self.policy = create_deterministic_policy(self.observation_shape,
                                                  self.action_size,
                                                  self.use_batch_norm)

    def _build_actor_optim(self):
        self.actor_optim = Adam(self.policy.parameters(),
                                lr=self.actor_learning_rate,
                                eps=self.eps)

    @train_api
    @torch_api
    def update_critic(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        q_tp1 = self.compute_target(obs_tp1) * (1.0 - ter_tp1)
        loss = self.q_func.compute_td(obs_t, act_t, rew_tp1, q_tp1, self.gamma)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.cpu().detach().numpy()

    @train_api
    @torch_api
    def update_actor(self, obs_t):
        action, raw_action = self.policy(obs_t, with_raw=True)
        q_t = self.q_func(obs_t, action)
        loss = -q_t.mean() + self.reguralizing_rate * (raw_action**2).mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    def compute_target(self, x):
        with torch.no_grad():
            action = self.targ_policy(x)
            return self.targ_q_func.compute_target(x, action.clamp(-1.0, 1.0))

    def _predict_best_action(self, x):
        return self.policy.best_action(x)

    @eval_api
    @torch_api
    def predict_value(self, x, action):
        assert x.shape[0] == action.shape[0]
        with torch.no_grad():
            return self.q_func(x, action).view(-1).cpu().detach().numpy()

    def update_critic_target(self):
        soft_sync(self.targ_q_func, self.q_func, self.tau)

    def update_actor_target(self):
        soft_sync(self.targ_policy, self.policy, self.tau)
