from torch.optim import Adam
from d3rlpy.models.torch.imitators import create_deterministic_regressor
from d3rlpy.models.torch.imitators import create_discrete_imitator
from .base import TorchImplBase
from .utility import torch_api, train_api
from .utility import compute_augmentation_mean


class BCImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, learning_rate, eps,
                 use_batch_norm, use_gpu, scaler, augmentation,
                 n_augmentations, encoder_params):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.scaler = scaler
        self.augmentation = augmentation
        self.n_augmentations = n_augmentations
        self.encoder_params = encoder_params
        self.use_gpu = use_gpu

    def build(self):
        self._build_network()

        if self.use_gpu:
            self.to_gpu(self.use_gpu)
        else:
            self.to_cpu()

        self._build_optim()

    def _build_network(self):
        self.imitator = create_deterministic_regressor(
            self.observation_shape,
            self.action_size,
            encoder_params=self.encoder_params)

    def _build_optim(self):
        self.optim = Adam(self.imitator.parameters(),
                          lr=self.learning_rate,
                          eps=self.eps)

    @train_api
    @torch_api(scaler_targets=['obs_t'])
    def update_imitator(self, obs_t, act_t):
        loss = compute_augmentation_mean(augmentation=self.augmentation,
                                         n_augmentations=self.n_augmentations,
                                         func=self._compute_loss,
                                         inputs={
                                             'obs_t': obs_t,
                                             'act_t': act_t
                                         },
                                         targets=['obs_t'])

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.cpu().detach().numpy()

    def _compute_loss(self, obs_t, act_t):
        return self.imitator.compute_error(obs_t, act_t)

    def _predict_best_action(self, x):
        return self.imitator(x)

    def predict_value(self, x, action, with_std):
        raise NotImplementedError('BC does not support value estimation')

    def sample_action(self, x):
        raise NotImplementedError('BC does not support sampling action')


class DiscreteBCImpl(BCImpl):
    def __init__(self, observation_shape, action_size, learning_rate, eps,
                 beta, use_batch_norm, use_gpu, scaler, augmentation,
                 n_augmentations, encoder_params):
        super().__init__(observation_shape=observation_shape,
                         action_size=action_size,
                         learning_rate=learning_rate,
                         eps=eps,
                         use_batch_norm=use_batch_norm,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation,
                         n_augmentations=n_augmentations,
                         encoder_params=encoder_params)
        self.beta = beta

    def _build_network(self):
        self.imitator = create_discrete_imitator(
            self.observation_shape,
            self.action_size,
            self.beta,
            encoder_params=self.encoder_params)

    def _predict_best_action(self, x):
        return self.imitator(x).argmax(dim=1)

    def _compute_loss(self, obs_t, act_t):
        return self.imitator.compute_error(obs_t, act_t.long())
