import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from d3rlpy.models.lr_schedulers import (
    CosineAnnealingLRFactory,
    WarmupSchedulerFactory,
)


@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_warmup_scheduler_factory(module: torch.nn.Module) -> None:
    factory = WarmupSchedulerFactory(warmup_steps=1000)

    lr_scheduler = factory.create(SGD(module.parameters(), 1e-4))

    assert isinstance(lr_scheduler, LambdaLR)

    # check serialization and deserialization
    WarmupSchedulerFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_cosine_annealing_lr_factory(module: torch.nn.Module) -> None:
    factory = CosineAnnealingLRFactory(T_max=1000)

    lr_scheduler = factory.create(SGD(module.parameters(), 1e-4))

    assert isinstance(lr_scheduler, CosineAnnealingLR)

    # check serialization and deserialization
    CosineAnnealingLRFactory.deserialize(factory.serialize())
