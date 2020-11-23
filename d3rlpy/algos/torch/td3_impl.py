import torch

from .ddpg_impl import DDPGImpl


class TD3Impl(DDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, actor_optim_factory,
                 critic_optim_factory, actor_encoder_factory,
                 critic_encoder_factory, gamma, tau, reguralizing_rate,
                 n_critics, bootstrap, share_encoder, target_smoothing_sigma,
                 target_smoothing_clip, q_func_type, use_gpu, scaler,
                 augmentation, n_augmentations):
        super().__init__(observation_shape=observation_shape,
                         action_size=action_size,
                         actor_learning_rate=actor_learning_rate,
                         critic_learning_rate=critic_learning_rate,
                         actor_optim_factory=actor_optim_factory,
                         critic_optim_factory=critic_optim_factory,
                         actor_encoder_factory=actor_encoder_factory,
                         critic_encoder_factory=critic_encoder_factory,
                         gamma=gamma,
                         tau=tau,
                         n_critics=n_critics,
                         bootstrap=bootstrap,
                         share_encoder=share_encoder,
                         reguralizing_rate=reguralizing_rate,
                         q_func_type=q_func_type,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation,
                         n_augmentations=n_augmentations)
        self.target_smoothing_sigma = target_smoothing_sigma
        self.target_smoothing_clip = target_smoothing_clip

    def compute_target(self, x):
        with torch.no_grad():
            action = self.targ_policy(x)
            # smoothing target
            noise = torch.randn(action.shape, device=x.device)
            scaled_noise = self.target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(-self.target_smoothing_clip,
                                               self.target_smoothing_clip)
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self.targ_q_func.compute_target(x, clipped_action)
