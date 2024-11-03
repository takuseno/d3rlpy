import pytest
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, RMSprop

from d3rlpy.optimizers.optimizers import (
    AdamFactory,
    AdamWFactory,
    GPTAdamWFactory,
    RMSpropFactory,
    SGDFactory,
)


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_sgd_factory(lr: float, module: torch.nn.Module) -> None:
    factory = SGDFactory()

    optim = factory.create(module.named_modules(), lr, False)

    assert isinstance(optim.optim, SGD)
    assert optim.optim.defaults["lr"] == lr

    # check serialization and deserialization
    SGDFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_adam_factory(lr: float, module: torch.nn.Module) -> None:
    factory = AdamFactory()

    optim = factory.create(module.named_modules(), lr, False)

    assert isinstance(optim.optim, Adam)
    assert optim.optim.defaults["lr"] == lr

    # check serialization and deserialization
    AdamFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_adam_w_factory(lr: float, module: torch.nn.Module) -> None:
    factory = AdamWFactory()

    optim = factory.create(module.named_modules(), lr, False)

    assert isinstance(optim.optim, AdamW)
    assert optim.optim.defaults["lr"] == lr

    # check serialization and deserialization
    AdamWFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_rmsprop_factory(lr: float, module: torch.nn.Module) -> None:
    factory = RMSpropFactory()

    optim = factory.create(module.named_modules(), lr, False)

    assert isinstance(optim.optim, RMSprop)
    assert optim.optim.defaults["lr"] == lr

    # check serialization and deserialization
    RMSpropFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("weight_decay", [0.1])
def test_gpt_adam_w_factory(lr: float, weight_decay: float) -> None:
    factory = GPTAdamWFactory(weight_decay=weight_decay)

    class M(nn.Module):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self.decay_module = nn.Linear(20, 30)
            self.non_decay_module = nn.Embedding(30, 30)

    module = M()

    optim = factory.create(module.named_modules(), lr, False)

    assert isinstance(optim.optim, AdamW)
    assert optim.optim.defaults["lr"] == lr
    assert len(optim.optim.param_groups) == 2
    assert optim.optim.param_groups[0]["weight_decay"] == weight_decay
    assert optim.optim.param_groups[1]["weight_decay"] == 0.0

    # check serialization and deserialization
    GPTAdamWFactory.deserialize(factory.serialize())
