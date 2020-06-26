import torch
import torch.nn as nn
import copy

from torch.optim import Adam
from skbrl.models.torch.policies import create_deterministic_residual_policy
from skbrl.models.torch.q_functions import create_continuous_q_function
from skbrl.models.torch.generators import create_conditional_vae
from skbrl.algos.torch.utility import torch_api
from .ddpg_impl import DDPGImpl


class BCQImpl(DDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, generator_learning_rate, gamma, tau,
                 n_critics, lam, n_action_samples, action_flexibility,
                 latent_size, eps, use_batch_norm, use_gpu):
        # generator requires these parameters
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.use_batch_norm = use_batch_norm
        self.eps = eps

        self.generator_learning_rate = generator_learning_rate
        self.n_critics = n_critics
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.action_flexibility = action_flexibility
        self.latent_size = latent_size

        self._build_generator()

        super().__init__(observation_shape, action_size, actor_learning_rate,
                         critic_learning_rate, gamma, tau, 0.0, eps,
                         use_batch_norm, use_gpu)

        # setup optimizer after the parameters move to GPU
        self._build_generator_optim()

    def _build_critic(self):
        self.q_func = create_continuous_q_function(self.observation_shape,
                                                   self.action_size,
                                                   self.n_critics,
                                                   self.use_batch_norm)

    def _build_actor(self):
        self.policy = create_deterministic_residual_policy(
            self.observation_shape, self.action_size, self.action_flexibility,
            self.use_batch_norm)

    def _build_generator(self):
        self.generator = create_conditional_vae(self.observation_shape,
                                                self.action_size,
                                                self.latent_size,
                                                self.use_batch_norm)

    def _build_generator_optim(self):
        self.generator_optim = Adam(self.generator.parameters(),
                                    self.generator_learning_rate,
                                    eps=self.eps)

    @torch_api
    def update_actor(self, obs_t):
        self.policy.train()
        self.q_func.train()
        self.generator.train()

        latent = torch.randn(obs_t.shape[0],
                             self.latent_size,
                             device=self.device)
        clipped_latent = latent.clamp(-0.5, 0.5)
        sampled_action = self.generator.decode(obs_t, clipped_latent)
        action = self.policy(obs_t, sampled_action)
        loss = -self.q_func(obs_t, action, 'min').mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    @torch_api
    def update_generator(self, obs_t, act_t):
        self.generator.train()

        loss = self.generator.compute_likelihood_loss(obs_t, act_t)

        self.generator_optim.zero_grad()
        loss.backward()
        self.generator_optim.step()

        return loss.cpu().detach().numpy()

    def _repeat_observation(self, x):
        # (batch_size, *obs_shape) -> (batch_size, n, *obs_shape)
        repeat_shape = (x.shape[0], self.n_action_samples, *x.shape[1:])
        repeated_x = x.view(x.shape[0], 1, *x.shape[1:]).expand(repeat_shape)
        return repeated_x

    def _sample_action(self, repeated_x, target=False):
        # TODO: this seems to be slow with image observation
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # sample latent variable
        latent = torch.randn(flattened_x.shape[0],
                             self.latent_size,
                             device=self.device)
        clipped_latent = latent.clamp(-0.5, 0.5)
        # sample action
        sampled_action = self.generator.decode(flattened_x, clipped_latent)
        # add residual action
        policy = self.targ_policy if target else self.policy
        action = policy(flattened_x, sampled_action)
        return action.view(-1, self.n_action_samples, self.action_size)

    def _predict_value(self, repeated_x, action, reduction, target=False):
        # TODO: this seems to be slow with image observation
        # (batch_size, n, *obs_shape) -> (batch_size * n, *obs_shape)
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # (batch_size, n, action_size) -> (batch_size * n, action_size)
        flattend_action = action.view(-1, self.action_size)
        # estimate values
        q_func = self.targ_q_func if target else self.q_func
        return q_func(flattened_x, flattend_action, reduction)

    def _predict_best_action(self, x):
        # TODO: this seems to be slow with image observation
        repeated_x = self._repeat_observation(x)
        action = self._sample_action(repeated_x)
        values = self._predict_value(repeated_x, action, 'none')[0]
        # pick the best (batch_size * n) -> (batch_size,)
        index = values.view(-1, self.n_action_samples).argmax(dim=1)
        return action[torch.arange(action.shape[0]), index]

    def compute_target(self, x):
        # TODO: this seems to be slow with image observation
        with torch.no_grad():
            repeated_x = self._repeat_observation(x)
            action = self._sample_action(repeated_x, True)
            # estimate values (n_ensembles, batch_size * n, 1)
            values = self._predict_value(repeated_x, action, 'none', True)
            # (n_ensembles, batch_size * n, 1) -> (n_ensembles, batch_size, n)
            values = values.view(self.n_critics, -1, self.n_action_samples)
            #(n_ensembles, batch_size, n) -> (batch_size, n)
            max_values = (1.0 - self.lam) * values.max(dim=0).values
            min_values = self.lam * values.min(dim=0).values
            mix_values = max_values + min_values
            #(batch_size, n) -> (batch_size, 1)
            return mix_values.max(dim=1, keepdim=True).values

    @torch_api
    def predict_best_action(self, x):
        with torch.no_grad():
            action = self._predict_best_action(x)
            return action.cpu().detach().numpy()

    def save_model(self, fname):
        torch.save(
            {
                'q_func': self.q_func.state_dict(),
                'policy': self.policy.state_dict(),
                'generator': self.generator.state_dict(),
                'critic_optim': self.critic_optim.state_dict(),
                'actor_optim': self.actor_optim.state_dict(),
                'generator_optim': self.generator_optim.state_dict(),
            }, fname)

    def load_model(self, fname):
        chkpt = torch.load(fname)
        self.q_func.load_state_dict(chkpt['q_func'])
        self.policy.load_state_dict(chkpt['policy'])
        self.generator.load_state_dict(chkpt['generator'])
        self.critic_optim.load_state_dict(chkpt['critic_optim'])
        self.actor_optim.load_state_dict(chkpt['actor_optim'])
        self.generator_optim.load_state_dict(chkpt['generator_optim'])
        self.targ_q_func = copy.deepcopy(self.q_func)
        self.targ_policy = copy.deepcopy(self.policy)

    def save_policy(self, fname):
        dummy_x = torch.rand(1, *self.observation_shape)

        # workaround until version 1.6
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False
        self.q_func.eval()
        for p in self.q_func.parameters():
            p.requires_grad = False
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False

        # dummy function to select best actions
        def _func(x):
            return self._predict_best_action(x)

        traced_script = torch.jit.trace(_func, dummy_x)
        traced_script.save(fname)

        for p in self.policy.parameters():
            p.requires_grad = True
        for p in self.q_func.parameters():
            p.requires_grad = True
        for p in self.generator.parameters():
            p.requires_grad = True

    def to_gpu(self):
        super().to_gpu()
        self.generator.cuda()

    def to_cpu(self):
        super().to_cpu()
        self.generator.cpu()
