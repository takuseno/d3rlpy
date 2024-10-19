import dataclasses

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler

from ..serializable_config import (
    DynamicConfig,
    generate_optional_config_generation,
)

__all__ = [
    "LRSchedulerFactory",
    "WarmupSchedulerFactory",
    "CosineAnnealingLRFactory",
    "make_lr_scheduler_field",
]


@dataclasses.dataclass()
class LRSchedulerFactory(DynamicConfig):
    """A factory class that creates a learning rate scheduler a lazy way."""

    def create(self, optim: Optimizer) -> LRScheduler:
        """Returns a learning rate scheduler object.

        Args:
            optim: PyTorch optimizer.

        Returns:
            Learning rate scheduler.
        """
        raise NotImplementedError


@dataclasses.dataclass()
class WarmupSchedulerFactory(LRSchedulerFactory):
    r"""A warmup learning rate scheduler.

    .. math::

        lr = \max((t + 1) / warmup\_steps, 1)

    Args:
        warmup_steps: Warmup steps.
    """

    warmup_steps: int

    def create(self, optim: Optimizer) -> LRScheduler:
        return LambdaLR(
            optim,
            lambda steps: min((steps + 1) / self.warmup_steps, 1),
        )

    @staticmethod
    def get_type() -> str:
        return "warmup"


@dataclasses.dataclass()
class CosineAnnealingLRFactory(LRSchedulerFactory):
    """A cosine annealing learning rate scheduler.

    Args:
        T_max: Maximum time step.
        eta_min: Minimum learning rate.
        last_epoch: Last epoch.
    """

    T_max: int
    eta_min: float = 0.0
    last_epoch: int = -1

    def create(self, optim: Optimizer) -> LRScheduler:
        return CosineAnnealingLR(
            optim,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
        )

    @staticmethod
    def get_type() -> str:
        return "cosine_annealing"


register_lr_scheduler_factory, make_lr_scheduler_field = (
    generate_optional_config_generation(
        LRSchedulerFactory,
    )
)

register_lr_scheduler_factory(WarmupSchedulerFactory)
register_lr_scheduler_factory(CosineAnnealingLRFactory)
