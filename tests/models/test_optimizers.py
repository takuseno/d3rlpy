import pytest
import torch
from torch.optim import SGD, Adam, RMSprop

from d3rlpy.models.optimizers import AdamFactory, RMSpropFactory, SGDFactory


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_sgd_factory(lr, module):
    factory = SGDFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, SGD)
    assert optim.defaults["lr"] == lr


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_adam_factory(lr, module):
    factory = AdamFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, Adam)
    assert optim.defaults["lr"] == lr


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_rmsprop_factory(lr, module):
    factory = RMSpropFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, RMSprop)
    assert optim.defaults["lr"] == lr
