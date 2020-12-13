from .base import AlgoBase
from .torch.plas_impl import PLASImpl
from ..optimizers import AdamFactory
from ..argument_utils import check_encoder
from ..argument_utils import check_use_gpu
from ..argument_utils import check_augmentation
from ..argument_utils import check_q_func


class PLAS(AlgoBase):
    r""" Policy in Latent Action Space algorithm.

    PLAS is an offline deep reinforcement learning algorithm whose policy
    function is trained in latent space of Conditional VAE.
    Unlike other algorithms, PLAS can achieve good performance by using
    its less constrained policy function.

    .. math::

       a \sim p_\Beta(a|s, z=\pi_\phi(s))

    where \Beta is a parameter of the decoder in Conditional VAE.

    References:
        * `Zhou et al., PLAS: latent action space for offline reinforcement
          learning. <https://arxiv.org/abs/2011.07213>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for Conditional VAE.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        imitator_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the conditional VAE.
        actor_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the critic.
        imitator_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the conditional VAE.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        lam (float): weight factor for critic ensemble.
        rl_start_epoch (int): epoch to start to update policy function and Q
            functions. If this is large, RL training would be more stabilized.
        beta (float): KL reguralization term for Conditional VAE.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.bcq_impl.BCQImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for Conditional VAE.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        imitator_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the conditional VAE.
        actor_encoder_factory (d3rlpy.encoders.EncoderFactory):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.encoders.EncoderFactory):
            encoder factory for the critic.
        imitator_encoder_factory (d3rlpy.encoders.EncoderFactory):
            encoder factory for the conditional VAE.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        lam (float): weight factor for critic ensemble.
        rl_start_epoch (int): epoch to start to update policy function and Q
            functions.
        beta (float): KL reguralization term for Conditional VAE.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.bcq_impl.BCQImpl): algorithm implementation.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self,
                 *,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 imitator_learning_rate=3e-4,
                 actor_optim_factory=AdamFactory(),
                 critic_optim_factory=AdamFactory(),
                 imitator_optim_factory=AdamFactory(),
                 actor_encoder_factory='default',
                 critic_encoder_factory='default',
                 imitator_encoder_factory='default',
                 q_func_factory='mean',
                 batch_size=256,
                 n_frames=1,
                 n_steps=1,
                 gamma=0.99,
                 tau=0.005,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 update_actor_interval=1,
                 lam=0.75,
                 rl_start_epoch=0,
                 beta=0.5,
                 use_gpu=False,
                 scaler=None,
                 augmentation=None,
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         n_frames=n_frames,
                         n_steps=n_steps,
                         gamma=gamma,
                         scaler=scaler,
                         dynamics=dynamics)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.imitator_learning_rate = imitator_learning_rate
        self.actor_optim_factory = actor_optim_factory
        self.critic_optim_factory = critic_optim_factory
        self.imitator_optim_factory = imitator_optim_factory
        self.actor_encoder_factory = check_encoder(actor_encoder_factory)
        self.critic_encoder_factory = check_encoder(critic_encoder_factory)
        self.imitator_encoder_factory = check_encoder(imitator_encoder_factory)
        self.q_func_factory = check_q_func(q_func_factory)
        self.tau = tau
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.update_actor_interval = update_actor_interval
        self.lam = lam
        self.rl_start_epoch = rl_start_epoch
        self.beta = beta
        self.augmentation = check_augmentation(augmentation)
        self.use_gpu = check_use_gpu(use_gpu)
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = PLASImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            imitator_learning_rate=self.imitator_learning_rate,
            actor_optim_factory=self.actor_optim_factory,
            critic_optim_factory=self.critic_optim_factory,
            imitator_optim_factory=self.imitator_optim_factory,
            actor_encoder_factory=self.actor_encoder_factory,
            critic_encoder_factory=self.critic_encoder_factory,
            imitator_encoder_factory=self.imitator_encoder_factory,
            q_func_factory=self.q_func_factory,
            gamma=self.gamma,
            tau=self.tau,
            n_critics=self.n_critics,
            bootstrap=self.bootstrap,
            share_encoder=self.share_encoder,
            lam=self.lam,
            beta=self.beta,
            use_gpu=self.use_gpu,
            scaler=self.scaler,
            augmentation=self.augmentation)
        self.impl.build()

    def update(self, epoch, total_step, batch):
        if epoch < self.rl_start_epoch:
            imitator_loss = self.impl.update_imitator(batch.observations,
                                                      batch.actions)
            critic_loss, actor_loss = None, None
        else:
            critic_loss = self.impl.update_critic(
                batch.observations, batch.actions, batch.next_rewards,
                batch.next_observations, batch.terminals, batch.n_steps)
            if total_step % self.update_actor_interval == 0:
                actor_loss = self.impl.update_actor(batch.observations)
                self.impl.update_actor_target()
                self.impl.update_critic_target()
            else:
                actor_loss = None
            imitator_loss = None
        return critic_loss, actor_loss, imitator_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss', 'imitator_loss']
