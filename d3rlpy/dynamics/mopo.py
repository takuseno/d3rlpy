from .base import DynamicsBase
from .torch.mopo_impl import MOPOImpl


class MOPO(DynamicsBase):
    def __init__(self,
                 n_epochs=30,
                 batch_size=100,
                 learning_rate=1e-3,
                 eps=1e-8,
                 n_ensembles=5,
                 n_transitions=5,
                 lam=1.0,
                 horizon=5,
                 use_batch_norm=False,
                 scaler=None,
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size, scaler, use_gpu)
        self.learning_rate = learning_rate
        self.eps = eps
        self.n_ensembles = n_ensembles
        self.n_transitions = n_transitions
        self.lam = lam
        self.horizon = horizon
        self.use_batch_norm = use_batch_norm
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = MOPOImpl(observation_shape=observation_shape,
                             action_size=action_size,
                             learning_rate=self.learning_rate,
                             eps=self.eps,
                             n_ensembles=self.n_ensembles,
                             lam=self.lam,
                             use_batch_norm=self.use_batch_norm,
                             scaler=self.scaler,
                             use_gpu=self.use_gpu)

    def update(self, epoch, total_step, batch):
        loss = self.impl.update(batch.observations, batch.actions,
                                batch.next_rewards, batch.next_observations)
        return [loss]

    def _get_loss_labels(self):
        return ['loss']
