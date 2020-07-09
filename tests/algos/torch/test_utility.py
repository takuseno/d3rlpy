import numpy as np
import torch
import pytest
import copy

from unittest.mock import Mock
from d3rlpy.algos.torch.utility import soft_sync, hard_sync
from d3rlpy.algos.torch.utility import set_eval_mode, set_train_mode
from d3rlpy.algos.torch.utility import freeze, unfreeze
from d3rlpy.algos.torch.utility import torch_api, train_api, eval_api
from d3rlpy.algos.torch.utility import map_location
from d3rlpy.algos.torch.utility import get_state_dict, set_state_dict


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


def test_map_location_with_cpu():
    assert map_location('cpu:0') == 'cpu'


def test_map_location_with_cuda():
    fn = map_location('cuda:0')
    dummy = Mock()
    dummy.cuda = Mock()

    fn(dummy, '')

    dummy.cuda.assert_called_with('cuda:0')


class DummyImpl:
    def __init__(self):
        self.fc1 = torch.nn.Linear(100, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.optim = torch.optim.Adam(self.fc1.parameters())
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


def check_if_same_dict(a, b):
    for k, v in a.items():
        if isinstance(v, torch.Tensor):
            assert (b[k] == v).all()
        else:
            assert b[k] == v


def test_get_state_dict():
    impl = DummyImpl()

    state_dict = get_state_dict(impl)

    check_if_same_dict(state_dict['fc1'], impl.fc1.state_dict())
    check_if_same_dict(state_dict['fc2'], impl.fc2.state_dict())
    check_if_same_dict(state_dict['optim'], impl.optim.state_dict())


def test_set_state_dict():
    impl1 = DummyImpl()
    impl2 = DummyImpl()

    impl1.optim.step()

    assert not (impl1.fc1.weight == impl2.fc1.weight).all()
    assert not (impl1.fc1.bias == impl2.fc1.bias).all()
    assert not (impl1.fc2.weight == impl2.fc2.weight).all()
    assert not (impl1.fc2.bias == impl2.fc2.bias).all()

    chkpt = get_state_dict(impl1)

    set_state_dict(impl2, chkpt)

    assert (impl1.fc1.weight == impl2.fc1.weight).all()
    assert (impl1.fc1.bias == impl2.fc1.bias).all()
    assert (impl1.fc2.weight == impl2.fc2.weight).all()
    assert (impl1.fc2.bias == impl2.fc2.bias).all()


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
