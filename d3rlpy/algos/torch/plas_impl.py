import torch
import copy

from d3rlpy.models.torch.policies import create_deterministic_policy
from d3rlpy.models.torch.policies import create_deterministic_residual_policy
from d3rlpy.models.torch.imitators import create_conditional_vae
from .utility import torch_api, train_api, soft_sync
from .ddpg_impl import DDPGImpl


class PLASImpl(DDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, imitator_learning_rate,
                 actor_optim_factory, critic_optim_factory,
                 imitator_optim_factory, actor_encoder_factory,
                 critic_encoder_factory, imitator_encoder_factory,
                 q_func_factory, gamma, tau, n_critics, bootstrap,
                 share_encoder, lam, beta, use_gpu, scaler, augmentation):
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
        self.policy = create_deterministic_policy(
            observation_shape=self.observation_shape,
            action_size=2 * self.action_size,
            encoder_factory=self.actor_encoder_factory)

    def _build_imitator(self):
        self.imitator = create_conditional_vae(
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            latent_size=2 * self.action_size,
            beta=self.beta,
            encoder_factory=self.imitator_encoder_factory)

    def _build_imitator_optim(self):
        self.imitator_optim = self.imitator_optim_factory.create(
            params=self.imitator.parameters(), lr=self.imitator_learning_rate)

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

    def _compute_actor_loss(self, obs_t):
        action = self.imitator.decode(obs_t, 2.0 * self.policy(obs_t))
        return -self.q_func(obs_t, action, 'none')[0].mean()

    def _predict_best_action(self, x):
        return self.imitator.decode(x, 2.0 * self.policy(x))

    def compute_target(self, x):
        with torch.no_grad():
            action = self.imitator.decode(x, 2.0 * self.targ_policy(x))
            return self.targ_q_func.compute_target(x, action, 'mix', self.lam)


class PLASWithPerturbationImpl(PLASImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, imitator_learning_rate,
                 actor_optim_factory, critic_optim_factory,
                 imitator_optim_factory, actor_encoder_factory,
                 critic_encoder_factory, imitator_encoder_factory,
                 q_func_factory, gamma, tau, n_critics, bootstrap,
                 share_encoder, lam, beta, action_flexibility, use_gpu, scaler,
                 augmentation):
        super().__init__(observation_shape=observation_shape,
                         action_size=action_size,
                         actor_learning_rate=actor_learning_rate,
                         critic_learning_rate=critic_learning_rate,
                         imitator_learning_rate=imitator_learning_rate,
                         actor_optim_factory=actor_optim_factory,
                         critic_optim_factory=critic_optim_factory,
                         imitator_optim_factory=imitator_optim_factory,
                         actor_encoder_factory=actor_encoder_factory,
                         critic_encoder_factory=critic_encoder_factory,
                         imitator_encoder_factory=imitator_encoder_factory,
                         q_func_factory=q_func_factory,
                         gamma=gamma,
                         tau=tau,
                         n_critics=n_critics,
                         bootstrap=bootstrap,
                         share_encoder=share_encoder,
                         lam=lam,
                         beta=beta,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation)
        self.action_flexibility = action_flexibility

        # initialized in build
        self.perturbation = None
        self.targ_perturbation = None

    def build(self):
        super().build()
        self.targ_perturbation = copy.deepcopy(self.perturbation)

    def _build_actor(self):
        super()._build_actor()
        self.perturbation = create_deterministic_residual_policy(
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            scale=self.action_flexibility,
            encoder_factory=self.actor_encoder_factory)

    def _build_actor_optim(self):
        parameters = list(self.policy.parameters())
        parameters += list(self.perturbation.parameters())
        self.actor_optim = self.actor_optim_factory.create(
            params=parameters, lr=self.actor_learning_rate)

    def _compute_actor_loss(self, obs_t):
        action = self.imitator.decode(obs_t, 2.0 * self.policy(obs_t))
        residual_action = self.perturbation(obs_t, action)
        return -self.q_func(obs_t, residual_action, 'none')[0].mean()

    def _predict_best_action(self, x):
        action = self.imitator.decode(x, 2.0 * self.policy(x))
        return self.perturbation(x, action)

    def compute_target(self, x):
        with torch.no_grad():
            action = self.imitator.decode(x, 2.0 * self.targ_policy(x))
            residual_action = self.targ_perturbation(x, action)
            return self.targ_q_func.compute_target(x,
                                                   residual_action,
                                                   reduction='mix',
                                                   lam=self.lam)

    def update_actor_target(self):
        super().update_actor_target()
        soft_sync(self.targ_perturbation, self.perturbation, self.tau)
