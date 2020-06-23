import torch
import torch.nn as nn
import copy
import math

from torch.optim import Adam
from skbrl.models.torch.policies import create_normal_policy
from skbrl.models.torch.q_functions import create_continuous_q_function
from .ddpg_impl import DDPGImpl


class SACImpl(DDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, temp_learning_rate, gamma, tau,
                 n_critics, initial_temperature, eps, use_batch_norm, use_gpu):
        self.n_critics = n_critics
        self.temp_learning_rate = temp_learning_rate
        self.initial_temperature = initial_temperature

        # setup temperature parameter
        # TODO: save and load temperature parameter
        self._build_temperature()
        self._build_temperature_optim()

        super().__init__(observation_shape, action_size, actor_learning_rate,
                         critic_learning_rate, gamma, tau, 0.0, eps,
                         use_batch_norm, use_gpu)

    def _build_critic(self):
        self.q_func = create_continuous_q_function(self.observation_shape,
                                                   self.action_size,
                                                   self.n_critics,
                                                   self.use_batch_norm)

    def _build_actor(self):
        self.policy = create_normal_policy(self.observation_shape,
                                           self.action_size,
                                           self.use_batch_norm)

    def _build_temperature(self):
        initial_val = math.log(self.initial_temperature)
        self.log_temp = nn.Parameter(torch.full((1, 1), initial_val))

    def _build_temperature_optim(self):
        self.temp_optim = Adam([self.log_temp], self.temp_learning_rate)

    def update_actor(self, obs_t):
        self.policy.train()
        self.q_func.train()
        device = self.device
        obs_t = torch.tensor(obs_t, dtype=torch.float32, device=device)

        action, log_prob = self.policy(obs_t, with_log_prob=True)
        entropy = self.log_temp.exp() * log_prob
        q_t = self.q_func(obs_t, action)
        loss = (entropy - q_t).mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    def update_temperature(self, obs_t):
        self.policy.train()
        device = self.device
        obs_t = torch.tensor(obs_t, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, log_prob = self.policy.sample(obs_t, with_log_prob=True)
            targ_temp = log_prob - self.action_size

        loss = -(self.log_temp.exp() * targ_temp).mean()

        self.temp_optim.zero_grad()
        loss.backward()
        self.temp_optim.step()

        return loss.cpu().detach().numpy()

    def compute_target(self, x):
        with torch.no_grad():
            action, log_prob = self.policy.sample(x, with_log_prob=True)
            entropy = self.log_temp.exp() * log_prob
            return self.targ_q_func(x, action) - entropy

    def to_gpu(self):
        super().to_gpu()
        self.log_temp.cuda()

    def to_cpu(self):
        super().to_cpu()
        self.log_temp.cpu()
