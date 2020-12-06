from torch.optim import Adam
from d3rlpy.models.torch.imitators import create_deterministic_regressor
from d3rlpy.models.torch.imitators import create_discrete_imitator
from .base import TorchImplBase
from .utility import torch_api, train_api


class BCImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, learning_rate,
                 optim_factory, encoder_factory, use_gpu, scaler,
                 augmentation):
        super().__init__(observation_shape, action_size, scaler)
        self.learning_rate = learning_rate
        self.optim_factory = optim_factory
        self.encoder_factory = encoder_factory
        self.augmentation = augmentation
        self.use_gpu = use_gpu

        # initialized in build
        self.imitator = None
        self.optim = None

    def build(self):
        self._build_network()

        if self.use_gpu:
            self.to_gpu(self.use_gpu)
        else:
            self.to_cpu()

        self._build_optim()

    def _build_network(self):
        self.imitator = create_deterministic_regressor(self.observation_shape,
                                                       self.action_size,
                                                       self.encoder_factory)

    def _build_optim(self):
        self.optim = self.optim_factory.create(self.imitator.parameters(),
                                               lr=self.learning_rate)

    @train_api
    @torch_api(scaler_targets=['obs_t'])
    def update_imitator(self, obs_t, act_t):
        loss = self.augmentation.process(func=self._compute_loss,
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
    def __init__(self, observation_shape, action_size, learning_rate,
                 optim_factory, encoder_factory, beta, use_gpu, scaler,
                 augmentation):
        super().__init__(observation_shape=observation_shape,
                         action_size=action_size,
                         learning_rate=learning_rate,
                         optim_factory=optim_factory,
                         encoder_factory=encoder_factory,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation)
        self.beta = beta

    def _build_network(self):
        self.imitator = create_discrete_imitator(self.observation_shape,
                                                 self.action_size, self.beta,
                                                 self.encoder_factory)

    def _predict_best_action(self, x):
        return self.imitator(x).argmax(dim=1)

    def _compute_loss(self, obs_t, act_t):
        return self.imitator.compute_error(obs_t, act_t.long())
