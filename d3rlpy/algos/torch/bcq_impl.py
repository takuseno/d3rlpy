import torch
import math

from torch.optim import Adam
from d3rlpy.models.torch.encoders import PixelEncoder
from d3rlpy.models.torch.policies import create_deterministic_residual_policy
from d3rlpy.models.torch.q_functions import compute_max_with_n_actions
from d3rlpy.models.torch.imitators import create_conditional_vae
from d3rlpy.models.torch.imitators import create_discrete_imitator
from d3rlpy.models.torch.imitators import DiscreteImitator
from .utility import torch_api, train_api
from .ddpg_impl import DDPGImpl
from .dqn_impl import DoubleDQNImpl


class BCQImpl(DDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, imitator_learning_rate,
                 actor_optim_factory, critic_optim_factory,
                 imitator_optim_factory, actor_encoder_factory,
                 critic_encoder_factory, imitator_encoder_factory,
                 q_func_factory, gamma, tau, n_critics, bootstrap,
                 share_encoder, lam, n_action_samples, action_flexibility,
                 latent_size, beta, use_gpu, scaler, augmentation):
        super().__init__(observation_shape=observation_shape,
                         action_size=action_size,
                         actor_learning_rate=actor_learning_rate,
                         critic_learning_rate=critic_learning_rate,
                         actor_optim_factory=actor_optim_factory,
                         critic_optim_factory=critic_optim_factory,
                         actor_encoder_factory=actor_encoder_factory,
                         critic_encoder_factory=critic_encoder_factory,
                         q_func_factory=q_func_factory,
                         gamma=gamma,
                         tau=tau,
                         n_critics=n_critics,
                         bootstrap=bootstrap,
                         share_encoder=share_encoder,
                         reguralizing_rate=0.0,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation)
        self.imitator_learning_rate = imitator_learning_rate
        self.imitator_optim_factory = imitator_optim_factory
        self.imitator_encoder_factory = imitator_encoder_factory
        self.n_critics = n_critics
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.action_flexibility = action_flexibility
        self.latent_size = latent_size
        self.beta = beta

        # initialized in build
        self.imitator = None
        self.imitator_optim = None

    def build(self):
        self._build_imitator()
        super().build()
        # setup optimizer after the parameters move to GPU
        self._build_imitator_optim()

    def _build_actor(self):
        self.policy = create_deterministic_residual_policy(
            self.observation_shape, self.action_size, self.action_flexibility,
            self.actor_encoder_factory)

    def _build_imitator(self):
        self.imitator = create_conditional_vae(self.observation_shape,
                                               self.action_size,
                                               self.latent_size, self.beta,
                                               self.imitator_encoder_factory)

    def _build_imitator_optim(self):
        self.imitator_optim = self.imitator_optim_factory.create(
            self.imitator.parameters(), lr=self.imitator_learning_rate)

    def _compute_actor_loss(self, obs_t):
        latent = torch.randn(obs_t.shape[0],
                             self.latent_size,
                             device=self.device)
        clipped_latent = latent.clamp(-0.5, 0.5)
        sampled_action = self.imitator.decode(obs_t, clipped_latent)
        action = self.policy(obs_t, sampled_action)
        return -self.q_func(obs_t, action, 'none')[0].mean()

    @train_api
    @torch_api(scaler_targets=['obs_t'])
    def update_imitator(self, obs_t, act_t):
        loss = self.augmentation.process(func=self.imitator.compute_error,
                                         inputs={
                                             'x': obs_t,
                                             'action': act_t
                                         },
                                         targets=['x'])

        self.imitator_optim.zero_grad()
        loss.backward()
        self.imitator_optim.step()

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
        sampled_action = self.imitator.decode(flattened_x, clipped_latent)
        # add residual action
        policy = self.targ_policy if target else self.policy
        action = policy(flattened_x, sampled_action)
        return action.view(-1, self.n_action_samples, self.action_size)

    def _predict_value(self, repeated_x, action, target=False):
        # TODO: this seems to be slow with image observation
        # (batch_size, n, *obs_shape) -> (batch_size * n, *obs_shape)
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # (batch_size, n, action_size) -> (batch_size * n, action_size)
        flattend_action = action.view(-1, self.action_size)
        # estimate values
        if target:
            return self.targ_q_func.compute_target(flattened_x,
                                                   flattend_action, 'none')
        return self.q_func(flattened_x, flattend_action, 'none')

    def _predict_best_action(self, x):
        # TODO: this seems to be slow with image observation
        repeated_x = self._repeat_observation(x)
        action = self._sample_action(repeated_x)
        values = self._predict_value(repeated_x, action)[0]
        # pick the best (batch_size * n) -> (batch_size,)
        index = values.view(-1, self.n_action_samples).argmax(dim=1)
        return action[torch.arange(action.shape[0]), index]

    def sample_action(self, x):
        raise NotImplementedError('BCQ does not support sampling action')

    def compute_target(self, x):
        # TODO: this seems to be slow with image observation
        with torch.no_grad():
            repeated_x = self._repeat_observation(x)
            actions = self._sample_action(repeated_x, True)

            values = compute_max_with_n_actions(x, actions, self.targ_q_func,
                                                self.lam)

            return values


class DiscreteBCQImpl(DoubleDQNImpl):
    def __init__(self, observation_shape, action_size, learning_rate,
                 optim_factory, encoder_factory, q_func_factory, gamma,
                 n_critics, bootstrap, share_encoder, action_flexibility, beta,
                 use_gpu, scaler, augmentation):
        super().__init__(observation_shape=observation_shape,
                         action_size=action_size,
                         learning_rate=learning_rate,
                         optim_factory=optim_factory,
                         encoder_factory=encoder_factory,
                         q_func_factory=q_func_factory,
                         gamma=gamma,
                         n_critics=n_critics,
                         bootstrap=bootstrap,
                         share_encoder=share_encoder,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation)
        self.action_flexibility = action_flexibility
        self.beta = beta

        # initialized in build
        self.imitator = None

    def _build_network(self):
        super()._build_network()
        # share convolutional layers if observation is pixel
        if isinstance(self.q_func.q_funcs[0].encoder, PixelEncoder):
            self.imitator = DiscreteImitator(self.q_func.q_funcs[0].encoder,
                                             self.action_size, self.beta)
        else:
            self.imitator = create_discrete_imitator(self.observation_shape,
                                                     self.action_size,
                                                     self.beta,
                                                     self.encoder_factory)

    def _build_optim(self):
        q_func_params = list(self.q_func.parameters())
        imitator_params = list(self.imitator.parameters())
        # retrieve unique elements
        unique_params = list(set(q_func_params + imitator_params))
        self.optim = self.optim_factory.create(unique_params,
                                               lr=self.learning_rate)

    def _compute_loss(self, obs_t, act_t, rew_tpn, q_tpn, n_steps):
        loss = super()._compute_loss(obs_t, act_t, rew_tpn, q_tpn, n_steps)
        imitator_loss = self.imitator.compute_error(obs_t, act_t.long())
        return loss + imitator_loss

    def _predict_best_action(self, x):
        log_probs = self.imitator(x)
        ratio = log_probs - log_probs.max(dim=1, keepdim=True).values
        mask = (ratio > math.log(self.action_flexibility)).float()
        value = self.q_func(x)
        normalized_value = value - value.min(dim=1, keepdim=True).values
        action = (normalized_value * mask).argmax(dim=1)
        return action
