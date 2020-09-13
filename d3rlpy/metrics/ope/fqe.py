from d3rlpy.algos.base import AlgoBase
from .torch.fqe_impl import FQEImpl


class FQE(AlgoBase):
    """ Fitted Q Evaluation.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\\thetta (s, a)` with the trained policy :math:`\\pi_\\phi(s)`.

    .. math::

        L(\\theta) = \\mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \\sim D}
            [(Q_\\theta(s_t, a_t) - r_{t+1}
                - \\gamma Q_{\\theta'}(s_{t+1}, \\pi_\\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        discrete_action (bool): flag to learn discrete action-space.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        target_update_interval (int): interval to update the target network.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function. Available options are
            `['mean', 'qr', 'iqn', 'fqf']`.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup. If the
            observation is pixel, you can pass ``filters`` with list of tuples
            consisting with ``(filter_size, kernel_size, stride)`` and
            ``feature_size`` with an integer scaler for the last linear layer
            size. If the observation is vector, you can pass ``hidden_units``
            with list of hidden unit sizes.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    Attributes:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        discrete_action (bool): flag to learn discrete action-space.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        target_update_interval (int): interval to update the target network.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function.
        n_epochs (int): the number of epochs to train.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    """
    def __init__(self,
                 algo=None,
                 learning_rate=3e-4,
                 batch_size=100,
                 n_frames=1,
                 gamma=0.99,
                 discrete_action=False,
                 n_critics=1,
                 bootstrap=False,
                 share_encoder=False,
                 eps=1e-8,
                 target_update_interval=100,
                 use_batch_norm=False,
                 q_func_type='mean',
                 n_epochs=30,
                 use_gpu=False,
                 scaler=None,
                 augmentation=[],
                 n_augmentations=1,
                 encoder_params={},
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size, n_frames, scaler, augmentation,
                         None, use_gpu)
        self.algo = algo
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.discrete_action = discrete_action
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.eps = eps
        self.target_update_interval = target_update_interval
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.encoder_params = encoder_params
        self.n_augmentations = n_augmentations
        self.impl = impl

    def save_policy(self, fname, as_onnx=False):
        self.algo.save_policy(fname, as_onnx)

    def predict(self, x):
        return self.algo.predict(x)

    def sample_action(self, x):
        return self.algo.sample_action(x)

    def create_impl(self, observation_shape, action_size):
        self.impl = FQEImpl(observation_shape=observation_shape,
                            action_size=action_size,
                            learning_rate=self.learning_rate,
                            gamma=self.gamma,
                            discrete_action=self.discrete_action,
                            n_critics=self.n_critics,
                            bootstrap=self.bootstrap,
                            share_encoder=self.share_encoder,
                            eps=self.eps,
                            use_batch_norm=self.use_batch_norm,
                            q_func_type=self.q_func_type,
                            use_gpu=self.use_gpu,
                            augmentation=self.augmentation,
                            n_augmentations=self.n_augmentations,
                            scaler=self.scaler,
                            encoder_params=self.encoder_params)
        self.impl.build()

    def update(self, epoch, total_step, batch):
        next_actions = self.algo.predict(batch.observations)
        loss = self.impl.update(batch.observations, batch.actions,
                                batch.next_rewards, next_actions,
                                batch.next_observations, batch.terminals)
        if total_step % self.target_update_interval == 0:
            self.impl.update_target()
        return (loss, )

    def _get_loss_labels(self):
        return ['value_loss']
