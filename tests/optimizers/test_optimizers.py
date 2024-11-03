from typing import Optional

import pytest
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, RMSprop

from d3rlpy.optimizers.lr_schedulers import (
    CosineAnnealingLRFactory,
    LRSchedulerFactory,
)
from d3rlpy.optimizers.optimizers import (
    AdamFactory,
    AdamWFactory,
    GPTAdamWFactory,
    OptimizerWrapper,
    RMSpropFactory,
    SGDFactory,
)


@pytest.mark.parametrize(
    "lr_scheduler_factory", [None, CosineAnnealingLRFactory(100)]
)
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("clip_grad_norm", [None, 1e-4])
def test_optimizer_wrapper(
    lr_scheduler_factory: Optional[LRSchedulerFactory],
    compiled: bool,
    clip_grad_norm: Optional[float],
) -> None:
    model = nn.Linear(100, 200)
    optim = SGD(model.parameters(), lr=1)
    lr_scheduler = (
        lr_scheduler_factory.create(optim) if lr_scheduler_factory else None
    )
    wrapper = OptimizerWrapper(
        params=list(model.parameters()),
        optim=optim,
        compiled=compiled,
        clip_grad_norm=clip_grad_norm,
        lr_scheduler=lr_scheduler,
    )

    loss = model(torch.rand(1, 100)).mean()
    loss.backward()

    # check zero grad
    wrapper.zero_grad()
    if compiled:
        assert model.weight.grad is None
        assert model.bias.grad is None
    else:
        assert torch.all(model.weight.grad == 0)
        assert torch.all(model.bias.grad == 0)

    # check step
    before_weight = torch.zeros_like(model.weight)
    before_weight.copy_(model.weight)
    before_bias = torch.zeros_like(model.bias)
    before_bias.copy_(model.bias)
    loss = model(torch.rand(1, 100)).mean()
    loss.backward()
    model.weight.grad.add_(1)
    model.weight.grad.mul_(10000)
    model.bias.grad.add_(1)
    model.bias.grad.mul_(10000)

    wrapper.step()
    assert torch.all(model.weight != before_weight)
    assert torch.all(model.bias != before_bias)

    # check clip_grad_norm
    if clip_grad_norm:
        assert torch.norm(model.weight.grad) < 1e-4
    else:
        assert torch.norm(model.weight.grad) > 1e-4


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
