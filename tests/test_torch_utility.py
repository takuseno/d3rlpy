import numpy as np
import torch
import pytest
import copy

from unittest.mock import Mock
from d3rlpy.torch_utility import soft_sync, hard_sync
from d3rlpy.torch_utility import set_eval_mode, set_train_mode
from d3rlpy.torch_utility import freeze, unfreeze
from d3rlpy.torch_utility import torch_api, train_api, eval_api
from d3rlpy.torch_utility import augmentation_api
from d3rlpy.torch_utility import map_location
from d3rlpy.torch_utility import get_state_dict, set_state_dict
from d3rlpy.torch_utility import _query_cache


@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("input_size", [32])
@pytest.mark.parametrize("output_size", [32])
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


@pytest.mark.parametrize("input_size", [32])
@pytest.mark.parametrize("output_size", [32])
def test_hard_sync(input_size, output_size):
    module = torch.nn.Linear(input_size, output_size)
    targ_module = torch.nn.Linear(input_size, output_size)

    hard_sync(targ_module, module)

    for p, targ_p in zip(module.parameters(), targ_module.parameters()):
        assert torch.allclose(targ_p, p)


def test_map_location_with_cpu():
    assert map_location("cpu:0") == "cpu"


def test_map_location_with_cuda():
    fn = map_location("cuda:0")
    dummy = Mock()
    dummy.cuda = Mock()

    fn(dummy, "")

    dummy.cuda.assert_called_with("cuda:0")


class DummyImpl:
    def __init__(self):
        self._fc1 = torch.nn.Linear(100, 100)
        self._fc2 = torch.nn.Linear(100, 100)
        self._optim = torch.optim.Adam(self._fc1.parameters())
        self._device = "cpu:0"
        self._scaler = None
        self._action_scaler = None
        self._augmentation = None

    @torch_api()
    def torch_api_func(self, x):
        assert isinstance(x, torch.Tensor)

    @torch_api(scaler_targets=["x"])
    def torch_api_func_with_scaler(self, x, y, ref_x, ref_y):
        assert isinstance(x, torch.Tensor)
        assert torch.allclose(x, torch.tensor(ref_x, dtype=torch.float32))
        assert torch.allclose(y, torch.tensor(ref_y, dtype=torch.float32))

    @torch_api(action_scaler_targets=["x"])
    def torch_api_func_with_action_scaler(self, x, y, ref_x, ref_y):
        assert isinstance(x, torch.Tensor)
        assert torch.allclose(x, torch.tensor(ref_x, dtype=torch.float32))
        assert torch.allclose(y, torch.tensor(ref_y, dtype=torch.float32))

    @train_api
    def train_api_func(self):
        assert self._fc1.training
        assert self._fc2.training

    @eval_api
    def eval_api_func(self):
        assert not self._fc1.training
        assert not self._fc2.training

    @augmentation_api(targets=["x"])
    def augmentation_api_func(self, x, y):
        return x + y

    @property
    def device(self):
        return self._device

    @property
    def scaler(self):
        return self._scaler

    @property
    def action_scaler(self):
        return self._action_scaler

    @property
    def augmentation(self):
        return self._augmentation


def check_if_same_dict(a, b):
    for k, v in a.items():
        if isinstance(v, torch.Tensor):
            assert (b[k] == v).all()
        else:
            assert b[k] == v


def test_get_state_dict():
    impl = DummyImpl()

    state_dict = get_state_dict(impl)

    check_if_same_dict(state_dict["_fc1"], impl._fc1.state_dict())
    check_if_same_dict(state_dict["_fc2"], impl._fc2.state_dict())
    check_if_same_dict(state_dict["_optim"], impl._optim.state_dict())


def test_set_state_dict():
    impl1 = DummyImpl()
    impl2 = DummyImpl()

    impl1._optim.step()

    assert not (impl1._fc1.weight == impl2._fc1.weight).all()
    assert not (impl1._fc1.bias == impl2._fc1.bias).all()
    assert not (impl1._fc2.weight == impl2._fc2.weight).all()
    assert not (impl1._fc2.bias == impl2._fc2.bias).all()

    chkpt = get_state_dict(impl1)

    set_state_dict(impl2, chkpt)

    assert (impl1._fc1.weight == impl2._fc1.weight).all()
    assert (impl1._fc1.bias == impl2._fc1.bias).all()
    assert (impl1._fc2.weight == impl2._fc2.weight).all()
    assert (impl1._fc2.bias == impl2._fc2.bias).all()


def test_eval_mode():
    impl = DummyImpl()
    impl._fc1.train()
    impl._fc2.train()

    set_eval_mode(impl)

    assert not impl._fc1.training
    assert not impl._fc2.training


def test_train_mode():
    impl = DummyImpl()
    impl._fc1.eval()
    impl._fc2.eval()

    set_train_mode(impl)

    assert impl._fc1.training
    assert impl._fc2.training


@pytest.mark.skip(reason="no way to test this")
def test_to_cuda():
    pass


@pytest.mark.skip(reason="no way to test this")
def test_to_cpu():
    pass


def test_freeze():
    impl = DummyImpl()

    freeze(impl)

    for p in impl._fc1.parameters():
        assert not p.requires_grad
    for p in impl._fc2.parameters():
        assert not p.requires_grad


def test_unfreeze():
    impl = DummyImpl()

    freeze(impl)
    unfreeze(impl)

    for p in impl._fc1.parameters():
        assert p.requires_grad
    for p in impl._fc2.parameters():
        assert p.requires_grad


def test_torch_api():
    impl = DummyImpl()
    impl._scaler = None

    x = np.random.random((100, 100))
    impl.torch_api_func(x)


def test_torch_api_with_scaler():
    impl = DummyImpl()

    class DummyScaler:
        def transform(self, x):
            return x + 0.1

    scaler = DummyScaler()
    impl._scaler = scaler

    x = np.random.random((100, 100))
    y = np.random.random((100, 100))
    impl.torch_api_func_with_scaler(x, y, ref_x=x + 0.1, ref_y=y)


def test_torch_api_with_action_scaler():
    impl = DummyImpl()

    class DummyActionScaler:
        def transform(self, action):
            return action + 0.1

    scaler = DummyActionScaler()
    impl._action_scaler = scaler

    x = np.random.random((100, 100))
    y = np.random.random((100, 100))
    impl.torch_api_func_with_action_scaler(x, y, ref_x=x + 0.1, ref_y=y)


def test_train_api():
    impl = DummyImpl()
    impl._fc1.eval()
    impl._fc2.eval()

    impl.train_api_func()


def test_eval_api():
    impl = DummyImpl()
    impl._fc1.train()
    impl._fc2.train()

    impl.eval_api_func()


def test_augmentation_api():
    impl = DummyImpl()

    class DummyAugmentationPipeline:
        def process(self, f, inputs, targets):
            for k, v in inputs.items():
                if k in targets:
                    inputs[k] = v + 1.0
            return f(**inputs)

    impl._augmentation = DummyAugmentationPipeline()

    x = torch.tensor(1.0)
    y = torch.tensor(2.0)
    assert impl.augmentation_api_func(x, y).numpy() == 4.0


def test_query_cache():
    for _ in range(20):
        x1 = np.random.random((100, 100))
        y1 = _query_cache(x1, "cpu:0")
        y2 = _query_cache(x1, "cpu:0")
        assert isinstance(y1, torch.Tensor)
        assert isinstance(y2, torch.Tensor)
        assert y1 is y2

        x2 = np.random.random((100, 100))
        y3 = _query_cache(x2, "cpu:0")
        y4 = _query_cache(x2, "cpu:0")
        assert isinstance(y3, torch.Tensor)
        assert isinstance(y4, torch.Tensor)
        assert y3 is y4
        assert y1 is not y3
