import torch

from skbrl.models.torch.q_functions import create_continuous_q_function
from .ddpg_impl import DDPGImpl


class TD3Impl(DDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, gamma, tau, reguralizing_rate,
                 n_critics, target_smoothing_sigma, target_smoothing_clip, eps,
                 use_batch_norm, distribution_type, use_gpu):
        self.n_critics = n_critics
        self.target_smoothing_sigma = target_smoothing_sigma
        self.target_smoothing_clip = target_smoothing_clip

        super().__init__(observation_shape, action_size, actor_learning_rate,
                         critic_learning_rate, gamma, tau, reguralizing_rate,
                         eps, use_batch_norm, distribution_type, use_gpu)

    def _build_critic(self):
        self.q_func = create_continuous_q_function(
            self.observation_shape,
            self.action_size,
            n_ensembles=self.n_critics,
            use_batch_norm=self.use_batch_norm,
            distribution_type=self.distribution_type)

    def compute_target(self, x):
        with torch.no_grad():
            action = self.targ_policy(x)
            # smoothing target
            noise = self.target_smoothing_sigma * torch.randn(action.shape)
            clipped_noise = noise.clamp(-self.target_smoothing_clip,
                                        self.target_smoothing_clip)
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self.targ_q_func.compute_target(x, clipped_action)
