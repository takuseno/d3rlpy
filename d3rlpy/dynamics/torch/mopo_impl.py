from torch.optim import Adam
from d3rlpy.models.torch.dynamics import create_probablistic_dynamics
from d3rlpy.algos.torch.utility import torch_api, train_api
from .base import TorchImplBase


class MOPOImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, learning_rate,
                 optim_factory, encoder_factory, n_ensembles, lam,
                 discrete_action, scaler, use_gpu):
        super().__init__(observation_shape, action_size, scaler)
        self.learning_rate = learning_rate
        self.optim_factory = optim_factory
        self.encoder_factory = encoder_factory
        self.n_ensembles = n_ensembles
        self.lam = lam
        self.discrete_action = discrete_action
        self.use_gpu = use_gpu

        self._build_dynamics()

        self.to_cpu()
        if self.use_gpu:
            self.to_gpu()

        self._build_optim()

    def _build_dynamics(self):
        self.dynamics = create_probablistic_dynamics(
            self.observation_shape,
            self.action_size,
            self.encoder_factory,
            n_ensembles=self.n_ensembles,
            discrete_action=self.discrete_action)

    def _build_optim(self):
        self.optim = self.optim_factory.create(self.dynamics.parameters(),
                                               lr=self.learning_rate)

    def _predict(self, x, action):
        return self.dynamics(x, action, True, 'max')

    def _generate(self, x, action):
        observations, rewards, variances = self.dynamics(x,
                                                         action,
                                                         with_variance=True,
                                                         variance_type='max')
        return observations, rewards - self.lam * variances

    @train_api
    @torch_api(scaler_targets=['obs_t', 'obs_tp1'])
    def update(self, obs_t, act_t, rew_tp1, obs_tp1):
        loss = self.dynamics.compute_error(obs_t, act_t, rew_tp1, obs_tp1)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.cpu().detach().numpy()
