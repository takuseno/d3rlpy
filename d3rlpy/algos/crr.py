from typing import Any, List, Optional, Sequence
from .base import AlgoBase, DataGenerator
from .torch.crr_impl import CRRImpl
from ..augmentation import AugmentationPipeline
from ..dataset import TransitionMiniBatch
from ..models.encoders import EncoderFactory
from ..models.q_functions import QFunctionFactory
from ..models.optimizers import OptimizerFactory, AdamFactory
from ..gpu import Device
from ..argument_utility import check_encoder, EncoderArg
from ..argument_utility import check_use_gpu, UseGPUArg
from ..argument_utility import check_augmentation, AugmentationArg
from ..argument_utility import check_q_func, QFuncArg
from ..argument_utility import ScalerArg, ActionScalerArg
from ..constants import IMPL_NOT_INITIALIZED_ERROR


class CRR(AlgoBase):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _bootstrap: bool
    _beta: float
    _n_action_samples: int
    _advantage_type: str
    _weight_type: str
    _max_weight: float
    _n_critics: int
    _share_encoder: bool
    _target_update_interval: int
    _target_reduction_type: str
    _update_actor_interval: int
    _augmentation: AugmentationPipeline
    _use_gpu: Optional[Device]
    _impl: Optional[CRRImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        beta: float = 1.0,
        n_action_samples: int = 4,
        advantage_type: str = "mean",
        weight_type: str = "exp",
        max_weight: float = 20.0,
        n_critics: int = 1,
        bootstrap: bool = False,
        share_encoder: bool = False,
        target_update_interval: int = 100,
        target_reduction_type: str = "min",
        update_actor_interval: int = 1,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        augmentation: AugmentationArg = None,
        generator: Optional[DataGenerator] = None,
        impl: Optional[CRRImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            generator=generator,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._bootstrap = bootstrap
        self._beta = beta
        self._n_action_samples = n_action_samples
        self._advantage_type = advantage_type
        self._weight_type = weight_type
        self._max_weight = max_weight
        self._n_critics = n_critics
        self._share_encoder = share_encoder
        self._target_update_interval = target_update_interval
        self._target_reduction_type = target_reduction_type
        self._update_actor_interval = update_actor_interval
        self._augmentation = check_augmentation(augmentation)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = CRRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            beta=self._beta,
            n_action_samples=self._n_action_samples,
            advantage_type=self._advantage_type,
            weight_type=self._weight_type,
            max_weight=self._max_weight,
            n_critics=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
            target_reduction_type=self._target_reduction_type,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()

    def update(
        self, epoch: int, total_step: int, batch: TransitionMiniBatch
    ) -> List[Optional[float]]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        critic_loss = self._impl.update_critic(
            batch.observations,
            batch.actions,
            batch.next_rewards,
            batch.next_observations,
            batch.terminals,
            batch.n_steps,
            batch.masks,
        )

        actor_loss = self._impl.update_actor(batch.observations, batch.actions)

        if total_step % self._target_update_interval == 0:
            self._impl.update_critic_target()
            self._impl.update_actor_target()

        return [critic_loss, actor_loss]

    def get_loss_labels(self) -> List[str]:
        return ["critic_loss", "actor_loss"]
