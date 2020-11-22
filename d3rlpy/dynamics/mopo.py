from .base import DynamicsBase
from ..optimizers import AdamFactory


class MOPO(DynamicsBase):
    """ Model-based Offline Policy Optimization.

    MOPO is a model-based RL approach for offline policy optimization.
    MOPO leverages the probablistic ensemble dynamics model to generate
    new dynamics data with uncertainty penalties.

    The ensemble dynamics model consists of :math:`N` probablistic models
    :math:`\\{T_{\\theta_i}\\}_{i=1}^N`.
    At each epoch, new transitions are generated via randomly picked dynamics
    model :math:`T_\\theta`.

    .. math::

        s_{t+1}, r_{t+1} \\sim T_\\theta(s_t, a_t)

    where :math:`s_t \\sim D` for the first step, otherwise :math:`s_t` is the
    previous generated observation, and :math:`a_t \\sim \\pi(\\cdot|s_t)`.
    The generated :math:`r_{t+1}` would be far from the ground truth if the
    actions sampled from the policy function is out-of-distribution.
    Thus, the uncertainty penalty reguralizes this bias.

    .. math::

        \\tilde{r_{t+1}} = r_{t+1} - \\lambda \\max_{i=1}^N
            || \\Sigma_i (s_t, a_t) ||

    where :math:`\\Sigma(s_t, a_t)` is the estimated variance.

    Finally, the generated transitions
    :math:`(s_t, a_t, \\tilde{r_{t+1}}, s_{t+1})` are appended to dataset
    :math:`D`.

    This generation process starts with randomly sampled `n_transitions`
    transitions till `horizon` steps.

    Note:
        Currently, MOPO only supports vector observations.

    References:
        * `Yu et al., MOPO: Model-based Offline Policy Optimization.
          <https://arxiv.org/abs/2005.13239>`_

    Args:
        learning_rate (float): learning rate for dynamics model.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_ensembles (int): the number of dynamics model for ensemble.
        n_transitions (int): the number of parallel trajectories to generate.
        horizon (int): the number of steps to generate.
        lam (float): :math:`\\lambda` for uncertainty penalties.
        use_batch_norm (bool): flag to insert batch normalization layers.
        discrete_action (bool): flag to take discrete actions.
        scaler (d3rlpy.preprocessing.scalers.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        use_gpu (bool or d3rlpy.gpu.Device): flag to use GPU or device.
        impl (d3rlpy.dynamics.base.DynamicsImplBase): dynamics implementation.

    Attributes:
        learning_rate (float): learning rate for dynamics model.
        optim_factory (d3rlpy.optimizers.OptimizerFactory): optimizer factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_ensembles (int): the number of dynamics model for ensemble.
        n_transitions (int): the number of parallel trajectories to generate.
        horizon (int): the number of steps to generate.
        lam (float): :math:`\\lambda` for uncertainty penalties.
        use_batch_norm (bool): flag to insert batch normalization layers.
        discrete_action (bool): flag to take discrete actions.
        scaler (d3rlpy.preprocessing.scalers.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        use_gpu (d3rlpy.gpu.Device): flag to use GPU or device.
        impl (d3rlpy.dynamics.base.DynamicsImplBase): dynamics implementation.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self,
                 *,
                 learning_rate=1e-3,
                 optim_factory=AdamFactory(weight_decay=1e-4),
                 batch_size=100,
                 n_frames=1,
                 n_ensembles=5,
                 n_transitions=400,
                 horizon=5,
                 lam=1.0,
                 use_batch_norm=False,
                 discrete_action=False,
                 scaler=None,
                 augmentation=[],
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         n_frames=n_frames,
                         n_transitions=n_transitions,
                         horizon=horizon,
                         scaler=scaler,
                         augmentation=augmentation,
                         use_gpu=use_gpu)
        self.learning_rate = learning_rate
        self.optim_factory = optim_factory
        self.n_ensembles = n_ensembles
        self.lam = lam
        self.use_batch_norm = use_batch_norm
        self.discrete_action = discrete_action
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from .torch.mopo_impl import MOPOImpl
        self.impl = MOPOImpl(observation_shape=observation_shape,
                             action_size=action_size,
                             learning_rate=self.learning_rate,
                             optim_factory=self.optim_factory,
                             n_ensembles=self.n_ensembles,
                             lam=self.lam,
                             use_batch_norm=self.use_batch_norm,
                             discrete_action=self.discrete_action,
                             scaler=self.scaler,
                             use_gpu=self.use_gpu)

    def update(self, epoch, total_step, batch):
        loss = self.impl.update(batch.observations, batch.actions,
                                batch.next_rewards, batch.next_observations)
        return [loss]

    def _get_loss_labels(self):
        return ['loss']
