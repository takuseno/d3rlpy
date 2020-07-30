from torch.optim import Adam
from d3rlpy.models.torch.dynamics import create_probablistic_dynamics
from d3rlpy.algos.torch.utility import torch_api, train_api
from .base import TorchImplBase


class MOPOImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, learning_rate, eps,
                 n_ensembles, lam, use_batch_norm, scaler, use_gpu):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.eps = eps
        self.n_ensembles = n_ensembles
        self.lam = lam
        self.use_batch_norm = use_batch_norm
        self.scaler = scaler
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
            n_ensembles=self.n_ensembles,
            use_batch_norm=self.use_batch_norm)

    def _build_optim(self):
        self.optim = Adam(self.dynamics.parameters(),
                          self.learning_rate,
                          eps=self.eps)

    def _predict(self, x, action):
        return self.dynamics(x, action)

    def _generate(self, x, action):
        observations, rewards, variances = self.dynamics(x, action, True)
        return observations, rewards - self.lam * variances

    @train_api
    @torch_api
    def update(self, obs_t, act_t, rew_tp1, obs_tp1):
        if self.scaler:
            obs_t = self.scaler.transform(obs_t)
            obs_tp1 = self.scaler.transform(obs_tp1)

        loss = self.dynamics.compute_error(obs_t, act_t, rew_tp1, obs_tp1)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.cpu().detach().numpy()
