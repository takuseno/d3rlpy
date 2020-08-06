import torch

from .ddpg_impl import DDPGImpl


class TD3Impl(DDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, gamma, tau, reguralizing_rate,
                 n_critics, bootstrap, share_encoder, target_smoothing_sigma,
                 target_smoothing_clip, eps, use_batch_norm, q_func_type,
                 use_gpu, scaler, augmentation, n_augmentations,
                 encoder_params):
        super().__init__(observation_shape, action_size, actor_learning_rate,
                         critic_learning_rate, gamma, tau, n_critics,
                         bootstrap, share_encoder, reguralizing_rate, eps,
                         use_batch_norm, q_func_type, use_gpu, scaler,
                         augmentation, n_augmentations, encoder_params)
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
