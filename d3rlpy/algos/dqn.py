from .base import AlgoBase
from .torch.dqn_impl import DQNImpl, DoubleDQNImpl
from ..optimizers import AdamFactory
from ..argument_utils import check_encoder
from ..argument_utils import check_use_gpu
from ..argument_utils import check_q_func
from ..argument_utils import check_augmentation


class DQN(AlgoBase):
    r""" Deep Q-Network algorithm.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \max_a Q_{\theta'}(s_{t+1}, a) - Q_\theta(s_t, a_t))^2]

    where :math:`\theta'` is the target network parameter. The target network
    parameter is synchronized every `target_update_interval` iterations.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_

    Args:
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
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.dqn_impl.DQNImpl): algorithm implementation.

    Attributes:
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
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.dqn_impl.DQNImpl): algorithm implementation.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self,
                 *,
                 learning_rate=6.25e-5,
                 optim_factory=AdamFactory(),
                 encoder_factory='default',
                 q_func_factory='mean',
                 batch_size=32,
                 n_frames=1,
                 gamma=0.99,
                 n_critics=1,
                 bootstrap=False,
                 share_encoder=False,
                 target_update_interval=8e3,
                 use_gpu=False,
                 scaler=None,
                 augmentation=None,
                 n_augmentations=1,
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         n_frames=n_frames,
                         scaler=scaler,
                         dynamics=dynamics)
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

    def create_impl(self, observation_shape, action_size):
        self.impl = DQNImpl(observation_shape=observation_shape,
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
                            scaler=self.scaler,
                            augmentation=self.augmentation,
                            n_augmentations=self.n_augmentations)
        self.impl.build()

    def update(self, epoch, total_step, batch):
        loss = self.impl.update(batch.observations, batch.actions,
                                batch.next_rewards, batch.next_observations,
                                batch.terminals)
        if total_step % self.target_update_interval == 0:
            self.impl.update_target()
        return (loss, )

    def _get_loss_labels(self):
        return ['value_loss']


class DoubleDQN(DQN):
    r""" Double Deep Q-Network algorithm.

    The difference from DQN is that the action is taken from the current Q
    function instead of the target Q function.
    This modification significantly decreases overestimation bias of TD
    learning.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma Q_{\theta'}(s_{t+1}, \text{argmax}_a
            Q_\theta(s_{t+1}, a)) - Q_\theta(s_t, a_t))^2]

    where :math:`\theta'` is the target network parameter. The target network
    parameter is synchronized every `target_update_interval` iterations.

    References:
        * `Hasselt et al., Deep reinforcement learning with double Q-learning.
          <https://arxiv.org/abs/1509.06461>`_

    Args:
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_update_interval (int): interval to synchronize the target
            network.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.dqn_impl.DoubleDQNImpl):
            algorithm implementation.

    Attributes:
        learning_rate (float): learning rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        encoder_factory (d3rlpy.encoders.EncoderFactory): encoder factory.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_update_interval (int): interval to synchronize the target
            network.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        dynamics (d3rlpy.dynaics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.dqn_impl.DoubleDQNImpl):
            algorithm implementation.

    """
    def create_impl(self, observation_shape, action_size):
        self.impl = DoubleDQNImpl(observation_shape=observation_shape,
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
                                  scaler=self.scaler,
                                  augmentation=self.augmentation,
                                  n_augmentations=self.n_augmentations)
        self.impl.build()
