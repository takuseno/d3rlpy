import numpy as np
import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from d3rlpy.optimizers.lr_schedulers import (
    CosineAnnealingLRFactory,
    WarmupSchedulerFactory,
)


@pytest.mark.parametrize("warmup_steps", [100])
@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_warmup_scheduler_factory(
    warmup_steps: int, lr: float, module: torch.nn.Module
) -> None:
    factory = WarmupSchedulerFactory(warmup_steps)

    lr_scheduler = factory.create(SGD(module.parameters(), lr=lr))

    assert np.allclose(lr_scheduler.get_lr()[0], lr / warmup_steps)
    for _ in range(warmup_steps):
        lr_scheduler.step()
    assert lr_scheduler.get_lr()[0] == lr

    assert isinstance(lr_scheduler, LambdaLR)

    # check serialization and deserialization
    WarmupSchedulerFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("T_max", [100])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_cosine_annealing_factory(T_max: int, module: torch.nn.Module) -> None:
    factory = CosineAnnealingLRFactory(T_max=T_max)

    lr_scheduler = factory.create(SGD(module.parameters()))

    assert isinstance(lr_scheduler, CosineAnnealingLR)

    # check serialization and deserialization
    CosineAnnealingLRFactory.deserialize(factory.serialize())
