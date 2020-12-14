import torch

from d3rlpy.models.torch.v_functions import create_value_function
from d3rlpy.models.torch.policies import squash_action, create_normal_policy
from d3rlpy.models.torch.policies import create_categorical_policy
from .utility import torch_api, train_api, eval_api
from .base import TorchImplBase


class AWRImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, actor_learning_rate,
                 critic_learning_rate, actor_optim_factory,
                 critic_optim_factory, actor_encoder_factory,
                 critic_encoder_factory, use_gpu, scaler, augmentation):
        super().__init__(observation_shape, action_size, scaler)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_optim_factory = actor_optim_factory
        self.critic_optim_factory = critic_optim_factory
        self.actor_encoder_factory = actor_encoder_factory
        self.critic_encoder_factory = critic_encoder_factory
        self.augmentation = augmentation
        self.use_gpu = use_gpu

        # initialized in build
        self.v_func = None
        self.policy = None
        self.critic_optim = None
        self.actor_optim = None

    def build(self):
        # setup torch models
        self._build_critic()
        self._build_actor()

        if self.use_gpu:
            self.to_gpu(self.use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()

    def _build_critic(self):
        self.v_func = create_value_function(self.observation_shape,
                                            self.critic_encoder_factory)

    def _build_critic_optim(self):
        self.critic_optim = self.critic_optim_factory.create(
            self.v_func.parameters(), lr=self.critic_learning_rate)

    def _build_actor(self):
        self.policy = create_normal_policy(self.observation_shape,
                                           self.action_size,
                                           self.actor_encoder_factory)

    def _build_actor_optim(self):
        self.actor_optim = self.actor_optim_factory.create(
            self.policy.parameters(), lr=self.actor_learning_rate)

    @train_api
    @torch_api(scaler_targets=['observation'])
    def update_critic(self, observation, value):
        loss = self.augmentation.process(func=self._compute_critic_loss,
                                         inputs={
                                             'observation': observation,
                                             'value': value
                                         },
                                         targets=['observation'])

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_critic_loss(self, observation, value):
        return self.v_func.compute_error(observation, value)

    @train_api
    @torch_api(scaler_targets=['observation'])
    def update_actor(self, observation, action, weight):
        loss = self.augmentation.process(func=self._compute_actor_loss,
                                         inputs={
                                             'observation': observation,
                                             'action': action,
                                             'weight': weight
                                         },
                                         targets=['observation'])

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_actor_loss(self, observation, action, weight):
        dist = self.policy.dist(observation)

        # unnormalize action via inverse tanh function
        unnormalized_action = torch.atanh(action.clamp(-0.999999, 0.999999))

        # compute log probability
        _, log_probs = squash_action(dist, unnormalized_action)

        return -(weight * log_probs).mean()

    def _predict_best_action(self, x):
        return self.policy.best_action(x)

    @eval_api
    @torch_api(scaler_targets=['x'])
    def predict_value(self, x, *args, **kwargs):
        with torch.no_grad():
            return self.v_func(x).view(-1).cpu().detach().numpy()

    @eval_api
    @torch_api(scaler_targets=['x'])
    def sample_action(self, x):
        with torch.no_grad():
            return self.policy.sample(x).cpu().detach().numpy()


class DiscreteAWRImpl(AWRImpl):
    def _build_actor(self):
        self.policy = create_categorical_policy(self.observation_shape,
                                                self.action_size,
                                                self.actor_encoder_factory)

    def _compute_actor_loss(self, observation, action, weight):
        dist = self.policy.dist(observation)
        log_probs = dist.log_prob(action).view(observation.shape[0], -1)
        return -(weight * log_probs.sum(dim=1, keepdims=True)).mean()
