import numpy as np
import torch
import pytest
import copy

from skbrl.algos.torch.utility import soft_sync, hard_sync
from skbrl.algos.torch.utility import set_eval_mode, set_train_mode
from skbrl.algos.torch.utility import freeze, unfreeze
from skbrl.algos.torch.utility import torch_api, train_api, eval_api


@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('input_size', [32])
@pytest.mark.parametrize('output_size', [32])
def test_soft_sync(tau, input_size, output_size):
    module = torch.nn.Linear(input_size, output_size)
    targ_module = torch.nn.Linear(input_size, output_size)
    original = copy.deepcopy(targ_module)

    soft_sync(targ_module, module, tau)

    module_params = module.parameters()
    targ_params = targ_module.parameters()
    original_params = original.parameters()
    for p, targ_p, orig_p in zip(module_params, targ_params, original_params):
        assert torch.allclose(p * tau + orig_p * (1.0 - tau), targ_p)


@pytest.mark.parametrize('input_size', [32])
@pytest.mark.parametrize('output_size', [32])
def test_hard_sync(input_size, output_size):
    module = torch.nn.Linear(input_size, output_size)
    targ_module = torch.nn.Linear(input_size, output_size)

    hard_sync(targ_module, module)

    for p, targ_p in zip(module.parameters(), targ_module.parameters()):
        assert torch.allclose(targ_p, p)


class DummyImpl:
    def __init__(self):
        self.fc1 = torch.nn.Linear(100, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.device = 'cpu:0'

    @torch_api
    def torch_api_func(self, x):
        assert isinstance(x, torch.Tensor)

    @train_api
    def train_api_func(self):
        assert self.fc1.training
        assert self.fc2.training

    @eval_api
    def eval_api_func(self):
        assert not self.fc1.training
        assert not self.fc2.training


def test_eval_mode():
    impl = DummyImpl()
    impl.fc1.train()
    impl.fc2.train()

    set_eval_mode(impl)

    assert not impl.fc1.training
    assert not impl.fc2.training


def test_train_mode():
    impl = DummyImpl()
    impl.fc1.eval()
    impl.fc2.eval()

    set_train_mode(impl)

    assert impl.fc1.training
    assert impl.fc2.training


@pytest.mark.skip(reason='no way to test this')
def test_to_cuda():
    pass


@pytest.mark.skip(reason='no way to test this')
def test_to_cpu():
    pass


def test_freeze():
    impl = DummyImpl()

    freeze(impl)

    for p in impl.fc1.parameters():
        assert not p.requires_grad
    for p in impl.fc2.parameters():
        assert not p.requires_grad


def test_unfreeze():
    impl = DummyImpl()

    freeze(impl)
    unfreeze(impl)

    for p in impl.fc1.parameters():
        assert p.requires_grad
    for p in impl.fc2.parameters():
        assert p.requires_grad


def test_torch_api():
    impl = DummyImpl()

    x = np.random.random((100, 100))
    impl.torch_api_func(x)


def test_train_api():
    impl = DummyImpl()
    impl.fc1.eval()
    impl.fc2.eval()

    impl.train_api_func()


def test_eval_api():
    impl = DummyImpl()
    impl.fc1.train()
    impl.fc2.train()

    impl.eval_api_func()
