import pytest
import torch
from torch.optim import SGD, Adam, AdamW, RMSprop

from d3rlpy.models.optimizers import (
    AdamFactory,
    AdamWFactory,
    RMSpropFactory,
    SGDFactory,
)


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_sgd_factory(lr: float, module: torch.nn.Module) -> None:
    factory = SGDFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, SGD)
    assert optim.defaults["lr"] == lr

    # check serialization and deserialization
    SGDFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_adam_factory(lr: float, module: torch.nn.Module) -> None:
    factory = AdamFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, Adam)
    assert optim.defaults["lr"] == lr

    # check serialization and deserialization
    AdamFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_adam_w_factory(lr: float, module: torch.nn.Module) -> None:
    factory = AdamWFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, AdamW)
    assert optim.defaults["lr"] == lr

    # check serialization and deserialization
    AdamWFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_rmsprop_factory(lr: float, module: torch.nn.Module) -> None:
    factory = RMSpropFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, RMSprop)
    assert optim.defaults["lr"] == lr

    # check serialization and deserialization
    RMSpropFactory.deserialize(factory.serialize())
