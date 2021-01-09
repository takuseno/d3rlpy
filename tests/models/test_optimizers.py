import pytest
import torch

from torch.optim import SGD, Adam, RMSprop
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.optimizers import SGDFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.optimizers import RMSpropFactory


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_optimizer_factory(lr, module):
    factory = OptimizerFactory(SGD)

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, SGD)
    assert optim.defaults["lr"] == lr

    params = factory.get_params()
    parameters = module.parameters()
    assert isinstance(OptimizerFactory(**params).create(parameters, lr), SGD)


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_sgd_factory(lr, module):
    factory = SGDFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, SGD)
    assert optim.defaults["lr"] == lr

    params = factory.get_params()
    parameters = module.parameters()
    assert isinstance(SGDFactory(**params).create(parameters, lr), SGD)


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_adam_factory(lr, module):
    factory = AdamFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, Adam)
    assert optim.defaults["lr"] == lr

    params = factory.get_params()
    parameters = module.parameters()
    assert isinstance(AdamFactory(**params).create(parameters, lr), Adam)


@pytest.mark.parametrize("lr", [1e-4])
@pytest.mark.parametrize("module", [torch.nn.Linear(2, 3)])
def test_rmsprop_factory(lr, module):
    factory = RMSpropFactory()

    optim = factory.create(module.parameters(), lr)

    assert isinstance(optim, RMSprop)
    assert optim.defaults["lr"] == lr

    params = factory.get_params()
    parameters = module.parameters()
    assert isinstance(RMSpropFactory(**params).create(parameters, lr), RMSprop)
