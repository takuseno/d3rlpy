import torch

from torch.optim import Adam
from d3rlpy.models.torch.policies import squash_action, create_normal_policy
from .ddpg_impl import DDPGImpl
from .utility import compute_augemtation_mean
from .utility import torch_api, train_api, eval_api


class AWACImpl(DDPGImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, gamma, tau, lam, n_action_samples,
                 actor_weight_decay, n_critics, bootstrap, share_encoder, eps,
                 use_batch_norm, q_func_type, use_gpu, scaler, augmentation,
                 n_augmentations, encoder_params):
        super().__init__(observation_shape, action_size, actor_learning_rate,
                         critic_learning_rate, gamma, tau, n_critics,
                         bootstrap, share_encoder, 0.0, eps, use_batch_norm,
                         q_func_type, use_gpu, scaler, augmentation,
                         n_augmentations, encoder_params)
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.actor_weight_decay = actor_weight_decay

    def _build_actor(self):
        self.policy = create_normal_policy(self.observation_shape,
                                           self.action_size,
                                           self.use_batch_norm,
                                           encoder_params=self.encoder_params)

    def _build_actor_optim(self):
        self.actor_optim = Adam(self.policy.parameters(),
                                lr=self.actor_learning_rate,
                                eps=self.eps,
                                weight_decay=self.actor_weight_decay)

    @train_api
    @torch_api
    def update_actor(self, obs_t,
                     act_t):  # override with additional parameters
        if self.scaler:
            obs_t = self.scaler.transform(obs_t)

        loss = compute_augemtation_mean(self.augmentation,
                                        self.n_augmentations,
                                        self._compute_actor_loss, {
                                            'obs_t': obs_t,
                                            'act_t': act_t
                                        }, ['obs_t'])

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_actor_loss(self, obs_t, act_t):
        dist = self.policy.dist(obs_t)

        # unnormalize action via inverse tanh function
        unnormalized_act_t = torch.atanh(act_t)

        # compute log probability
        _, log_probs = squash_action(dist, unnormalized_act_t)

        # compute weight
        with torch.no_grad():
            batch_size = obs_t.shape[0]

            # compute action-value
            q_values = self.q_func(obs_t, act_t)

            # sample random actions
            # (batch_size * N, action_size)
            random_actions = torch.empty(batch_size * self.n_action_samples,
                                         self.action_size,
                                         dtype=torch.float32,
                                         device=obs_t.device)
            random_actions.uniform_(-1.0, 1.0)

            # repeat observation
            # (batch_size, obs_size) -> (batch_size, 1, obs_size)
            reshaped_obs_t = obs_t.view(batch_size, 1, *obs_t.shape[1:])
            # (batch_sie, 1, obs_size) -> (batch_size, N, obs_size)
            repeated_obs_t = reshaped_obs_t.expand(batch_size,
                                                   self.n_action_samples,
                                                   *obs_t.shape[1:])
            # (batch_size, N, obs_size) -> (batch_size * N, obs_size)
            flat_obs_t = repeated_obs_t.reshape(-1, *obs_t.shape[1:])

            # compute state-value
            flat_v_values = self.q_func(flat_obs_t, random_actions)
            reshaped_v_values = flat_v_values.view(obs_t.shape[0], -1, 1)
            v_values = reshaped_v_values.mean(dim=1)

            # compute weight
            weights = torch.exp((q_values - v_values) / self.lam)

        return -(log_probs * weights).mean()
