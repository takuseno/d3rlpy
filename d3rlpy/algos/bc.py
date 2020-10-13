from .base import AlgoBase
from .torch.bc_impl import BCImpl, DiscreteBCImpl


class BC(AlgoBase):
    """ Behavior Cloning algorithm.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\\theta) = \\mathbb{E}_{a_t, s_t \\sim D}
            [(a_t - \\pi_\\theta(s_t))^2]

    Args:
        learning_rate (float): learing rate.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
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
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.bc_impl.BCImpl):
            implemenation of the algorithm.

    Attributes:
        n_epochs (int): the number of epochs to train.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        learning_rate (float): learing rate.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.bc_impl.BCImpl):
            implemenation of the algorithm.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self,
                 *,
                 learning_rate=1e-3,
                 batch_size=100,
                 n_frames=1,
                 eps=1e-8,
                 use_batch_norm=False,
                 n_epochs=1000,
                 use_gpu=False,
                 scaler=None,
                 augmentation=[],
                 n_augmentations=1,
                 encoder_params={},
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs=n_epochs,
                         batch_size=batch_size,
                         n_frames=n_frames,
                         scaler=scaler,
                         augmentation=augmentation,
                         dynamics=dynamics,
                         use_gpu=use_gpu)
        self.learning_rate = learning_rate
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.n_augmentations = n_augmentations
        self.encoder_params = encoder_params
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = BCImpl(observation_shape=observation_shape,
                           action_size=action_size,
                           learning_rate=self.learning_rate,
                           eps=self.eps,
                           use_batch_norm=self.use_batch_norm,
                           use_gpu=self.use_gpu,
                           scaler=self.scaler,
                           augmentation=self.augmentation,
                           n_augmentations=self.n_augmentations,
                           encoder_params=self.encoder_params)
        self.impl.build()

    def update(self, epoch, itr, batch):
        loss = self.impl.update_imitator(batch.observations, batch.actions)
        return (loss, )

    def predict_value(self, x, action):
        """ value prediction is not supported by BC algorithms.
        """
        raise NotImplementedError('BC does not support value estimation.')

    def sample_action(self, x):
        """ sampling action is not supported by BC algorithm.
        """
        raise NotImplementedError('BC does not support sampling action.')

    def _get_loss_labels(self):
        return ['loss']


class DiscreteBC(BC):
    """ Behavior Cloning algorithm for discrete control.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\\theta) = \\mathbb{E}_{a_t, s_t \\sim D}
            [-\\sum_a p(a|s_t) \\log \\pi_\\theta(a|s_t)]

    where :math:`p(a|s_t)` is implemented as a one-hot vector.

    Args:
        n_epochs (int): the number of epochs to train.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        learning_rate (float): learing rate.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        beta (float): reguralization factor.
        use_batch_norm (bool): flag to insert batch normalization layers.
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
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.bc_impl.DiscreteBCImpl):
            implemenation of the algorithm.

    Attributes:
        n_epochs (int): the number of epochs to train.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        learning_rate (float): learing rate.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        beta (float): reguralization factor.
        use_batch_norm (bool): flag to insert batch normalization layers.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.bc_impl.DiscreteBCImpl):
            implemenation of the algorithm.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self,
                 *,
                 learning_rate=1e-3,
                 batch_size=100,
                 n_frames=1,
                 eps=1e-8,
                 beta=0.5,
                 use_batch_norm=False,
                 n_epochs=1000,
                 use_gpu=False,
                 scaler=None,
                 augmentation=[],
                 n_augmentations=1,
                 encoder_params={},
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(learning_rate=learning_rate,
                         batch_size=batch_size,
                         n_frames=n_frames,
                         eps=eps,
                         use_batch_norm=use_batch_norm,
                         n_epochs=n_epochs,
                         use_gpu=use_gpu,
                         scaler=scaler,
                         augmentation=augmentation,
                         n_augmentations=n_augmentations,
                         encoder_params=encoder_params,
                         dynamics=dynamics,
                         impl=impl,
                         **kwargs)
        self.beta = beta

    def create_impl(self, observation_shape, action_size):
        self.impl = DiscreteBCImpl(observation_shape=observation_shape,
                                   action_size=action_size,
                                   learning_rate=self.learning_rate,
                                   eps=self.eps,
                                   beta=self.beta,
                                   use_batch_norm=self.use_batch_norm,
                                   use_gpu=self.use_gpu,
                                   scaler=self.scaler,
                                   augmentation=self.augmentation,
                                   n_augmentations=self.n_augmentations,
                                   encoder_params=self.encoder_params)
        self.impl.build()
