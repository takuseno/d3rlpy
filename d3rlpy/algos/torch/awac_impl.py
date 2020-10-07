import torch
import torch.nn.functional as F

from torch.optim import Adam
from d3rlpy.models.torch.policies import squash_action, create_normal_policy
from .sac_impl import SACImpl
from .utility import compute_augemtation_mean
from .utility import torch_api, train_api, eval_api


class AWACImpl(SACImpl):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, gamma, tau, lam, n_action_samples,
                 max_weight, actor_weight_decay, n_critics, bootstrap,
                 share_encoder, eps, use_batch_norm, q_func_type, use_gpu,
                 scaler, augmentation, n_augmentations, encoder_params):
        super().__init__(observation_shape=observation_shape,
                         action_size=action_size,
                         actor_learning_rate=actor_learning_rate,
                         critic_learning_rate=critic_learning_rate,
                         temp_learning_rate=0.0,
                         gamma=gamma,
                         tau=tau,
                         n_critics=n_critics,
                         bootstrap=bootstrap,
                         share_encoder=share_encoder,
                         initial_temperature=1e-20,
                         eps=eps,
                         use_batch_norm=use_batch_norm,
                         q_func_type=q_func_type,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation,
                         n_augmentations=n_augmentations,
                         encoder_params=encoder_params)
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.max_weight = max_weight
        self.actor_weight_decay = actor_weight_decay

    def _build_actor(self):
        self.policy = create_normal_policy(self.observation_shape,
                                           self.action_size,
                                           self.use_batch_norm,
                                           min_logstd=-6.0,
                                           max_logstd=0.0,
                                           use_std_parameter=True,
                                           encoder_params=self.encoder_params)

    def _build_actor_optim(self):
        self.actor_optim = Adam(self.policy.parameters(),
                                lr=self.actor_learning_rate,
                                eps=self.eps,
                                weight_decay=self.actor_weight_decay)

    @train_api
    @torch_api
    def update_actor(self, obs_t, act_t):
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

        # get current standard deviation for policy function for debug
        mean_std = self.policy.get_logstd_parameter().exp().mean()

        return loss.cpu().detach().numpy(), mean_std.cpu().detach().numpy()

    def _compute_actor_loss(self, obs_t, act_t):
        dist = self.policy.dist(obs_t)

        # unnormalize action via inverse tanh function
        unnormalized_act_t = torch.atanh(act_t.clamp(-0.999999, 0.999999))

        # compute log probability
        _, log_probs = squash_action(dist, unnormalized_act_t)

        # compute exponential weight
        weights = self._compute_weights(obs_t, act_t)

        return -(log_probs * weights).sum()

    def _compute_weights(self, obs_t, act_t):
        with torch.no_grad():
            batch_size = obs_t.shape[0]

            # compute action-value
            q_values = self.q_func(obs_t, act_t, 'min')

            # sample actions
            # (batch_size * N, action_size)
            policy_actions = self.policy.sample_n(obs_t, self.n_action_samples)
            flat_actions = policy_actions.reshape(-1, self.action_size)

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
            flat_v_values = self.q_func(flat_obs_t, flat_actions, 'min')
            reshaped_v_values = flat_v_values.view(obs_t.shape[0], -1, 1)
            v_values = reshaped_v_values.mean(dim=1)

            # compute normalized weight
            adv_values = (q_values - v_values).view(-1)
            weights = F.softmax(adv_values / self.lam, dim=0).view(-1, 1)

            # clip like AWR
            clipped_weights = weights.clamp(0.0, self.max_weight)

        return clipped_weights
