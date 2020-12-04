from ..algos.base import AlgoBase
from ..optimizers import AdamFactory
from ..argument_utils import check_encoder
from ..argument_utils import check_use_gpu
from ..argument_utils import check_q_func
from ..argument_utils import check_augmentation
from .torch.fqe_impl import FQEImpl, DiscreteFQEImpl


class FQE(AlgoBase):
    r""" Fitted Q Evaluation.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \sim D}
            [(Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory or str):
            optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_update_interval (int): interval to update the target network.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    Attributes:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory): encoder factory.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_update_interval (int): interval to update the target network.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    """
    def __init__(self,
                 *,
                 algo=None,
                 learning_rate=1e-4,
                 optim_factory=AdamFactory(),
                 encoder_factory='default',
                 q_func_factory='mean',
                 batch_size=100,
                 n_frames=1,
                 gamma=0.99,
                 n_critics=1,
                 bootstrap=False,
                 share_encoder=False,
                 target_update_interval=100,
                 use_gpu=False,
                 scaler=None,
                 augmentation=[],
                 n_augmentations=1,
                 impl=None,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         n_frames=n_frames,
                         scaler=scaler,
                         dynamics=None)
        self.algo = algo
        self.learning_rate = learning_rate
        self.optim_factory = optim_factory
        self.encoder_factory = check_encoder(encoder_factory)
        self.q_func_factory = check_q_func(q_func_factory)
        self.gamma = gamma
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.target_update_interval = target_update_interval
        self.augmentation = check_augmentation(augmentation)
        self.n_augmentations = n_augmentations
        self.use_gpu = check_use_gpu(use_gpu)
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
                            optim_factory=self.optim_factory,
                            encoder_factory=self.encoder_factory,
                            q_func_factory=self.q_func_factory,
                            gamma=self.gamma,
                            n_critics=self.n_critics,
                            bootstrap=self.bootstrap,
                            share_encoder=self.share_encoder,
                            use_gpu=self.use_gpu,
                            augmentation=self.augmentation,
                            n_augmentations=self.n_augmentations,
                            scaler=self.scaler)
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


class DiscreteFQE(FQE):
    r""" Fitted Q Evaluation for discrete action-space.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta (s, a)` with the trained policy :math:`\pi_\phi(s)`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1} s_{t+1} \sim D}
            [(Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})))^2]

    The trained Q function in FQE will estimate evaluation metrics more
    accurately than learned Q function during training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory or str):
            optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_update_interval (int): interval to update the target network.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    Attributes:
        algo (d3rlpy.algos.base.AlgoBase): algorithm to evaluate.
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory): encoder factory.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_update_interval (int): interval to update the target network.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        impl (d3rlpy.metrics.ope.torch.FQEImpl): algorithm implementation.

    """
    def create_impl(self, observation_shape, action_size):
        self.impl = DiscreteFQEImpl(observation_shape=observation_shape,
                                    action_size=action_size,
                                    learning_rate=self.learning_rate,
                                    optim_factory=self.optim_factory,
                                    encoder_factory=self.encoder_factory,
                                    q_func_factory=self.q_func_factory,
                                    gamma=self.gamma,
                                    n_critics=self.n_critics,
                                    bootstrap=self.bootstrap,
                                    share_encoder=self.share_encoder,
                                    use_gpu=self.use_gpu,
                                    augmentation=self.augmentation,
                                    n_augmentations=self.n_augmentations,
                                    scaler=self.scaler)
        self.impl.build()
