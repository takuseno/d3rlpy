import copy
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from d3rlpy.dataset import Transition, TransitionMiniBatch
from d3rlpy.torch_utility import (
    Swish,
    TorchMiniBatch,
    View,
    eval_api,
    freeze,
    get_state_dict,
    hard_sync,
    map_location,
    set_eval_mode,
    set_state_dict,
    set_train_mode,
    soft_sync,
    torch_api,
    train_api,
    unfreeze,
)


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
        self._reward_scaler = None

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

    @torch_api()
    def torch_api_func_with_batch(self, batch):
        assert isinstance(batch, TorchMiniBatch)
        return batch

    @train_api
    def train_api_func(self):
        assert self._fc1.training
        assert self._fc2.training

    @eval_api
    def eval_api_func(self):
        assert not self._fc1.training
        assert not self._fc2.training

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
    def reward_scaler(self):
        return self._reward_scaler


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


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("use_scaler", [False, True])
@pytest.mark.parametrize("use_action_scaler", [False, True])
@pytest.mark.parametrize("use_reward_scaler", [False, True])
def test_torch_mini_batch(
    batch_size,
    observation_shape,
    action_size,
    use_scaler,
    use_action_scaler,
    use_reward_scaler,
):
    obs_shape = (batch_size,) + observation_shape
    transitions = []
    for _ in range(batch_size):
        transition = Transition(
            observation_shape=observation_shape,
            action_size=action_size,
            observation=np.random.random(obs_shape),
            action=np.random.random(action_size),
            reward=np.random.random(),
            next_observation=np.random.random(obs_shape),
            next_action=np.random.random(action_size),
            next_reward=np.random.random(),
            terminal=0.0,
        )
        transitions.append(transition)

    if use_scaler:

        class DummyScaler:
            def transform(self, x):
                return x + 0.1

        scaler = DummyScaler()
    else:
        scaler = None

    if use_action_scaler:

        class DummyActionScaler:
            def transform(self, x):
                return x + 0.2

        action_scaler = DummyActionScaler()
    else:
        action_scaler = None

    if use_reward_scaler:

        class DummyRewardScaler:
            def transform(self, x):
                return x + 0.2

        reward_scaler = DummyRewardScaler()
    else:
        reward_scaler = None

    batch = TransitionMiniBatch(transitions)

    torch_batch = TorchMiniBatch(
        batch=batch,
        device="cpu:0",
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )

    if use_scaler:
        assert np.all(
            torch_batch.observations.numpy() == batch.observations + 0.1
        )
        assert np.all(
            torch_batch.next_observations.numpy()
            == batch.next_observations + 0.1
        )
    else:
        assert np.all(torch_batch.observations.numpy() == batch.observations)
        assert np.all(
            torch_batch.next_observations.numpy() == batch.next_observations
        )

    if use_action_scaler:
        assert np.all(torch_batch.actions.numpy() == batch.actions + 0.2)
        assert np.all(
            torch_batch.next_actions.numpy() == batch.next_actions + 0.2
        )
    else:
        assert np.all(torch_batch.actions.numpy() == batch.actions)
        assert np.all(torch_batch.next_actions.numpy() == batch.next_actions)

    if use_reward_scaler:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards + 0.2)
        assert np.all(
            torch_batch.next_rewards.numpy() == batch.next_rewards + 0.2
        )
    else:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards)
        assert np.all(torch_batch.next_rewards.numpy() == batch.next_rewards)

    assert np.all(torch_batch.terminals.numpy() == batch.terminals)
    assert np.all(torch_batch.n_steps.numpy() == batch.n_steps)


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


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("use_scaler", [False, True])
@pytest.mark.parametrize("use_action_scaler", [False, True])
@pytest.mark.parametrize("use_reward_scaler", [False, True])
def test_torch_api_with_batch(
    batch_size,
    observation_shape,
    action_size,
    use_scaler,
    use_action_scaler,
    use_reward_scaler,
):
    obs_shape = (batch_size,) + observation_shape
    transitions = []
    for _ in range(batch_size):
        transition = Transition(
            observation_shape=observation_shape,
            action_size=action_size,
            observation=np.random.random(obs_shape),
            action=np.random.random(action_size),
            reward=np.random.random(),
            next_observation=np.random.random(obs_shape),
            next_action=np.random.random(action_size),
            next_reward=np.random.random(),
            terminal=0.0,
        )
        transitions.append(transition)

    if use_scaler:

        class DummyScaler:
            def transform(self, x):
                return x + 0.1

        scaler = DummyScaler()
    else:
        scaler = None

    if use_action_scaler:

        class DummyActionScaler:
            def transform(self, x):
                return x + 0.2

        action_scaler = DummyActionScaler()
    else:
        action_scaler = None

    if use_reward_scaler:

        class DummyRewardScaler:
            def transform(self, x):
                return x + 0.2

        reward_scaler = DummyRewardScaler()
    else:
        reward_scaler = None

    batch = TransitionMiniBatch(transitions)

    impl = DummyImpl()
    impl._scaler = scaler
    impl._action_scaler = action_scaler
    impl._reward_scaler = reward_scaler

    torch_batch = impl.torch_api_func_with_batch(batch)

    if use_scaler:
        assert np.all(
            torch_batch.observations.numpy() == batch.observations + 0.1
        )
        assert np.all(
            torch_batch.next_observations.numpy()
            == batch.next_observations + 0.1
        )
    else:
        assert np.all(torch_batch.observations.numpy() == batch.observations)
        assert np.all(
            torch_batch.next_observations.numpy() == batch.next_observations
        )

    if use_action_scaler:
        assert np.all(torch_batch.actions.numpy() == batch.actions + 0.2)
        assert np.all(
            torch_batch.next_actions.numpy() == batch.next_actions + 0.2
        )
    else:
        assert np.all(torch_batch.actions.numpy() == batch.actions)
        assert np.all(torch_batch.next_actions.numpy() == batch.next_actions)

    if use_reward_scaler:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards + 0.2)
        assert np.all(
            torch_batch.next_rewards.numpy() == batch.next_rewards + 0.2
        )
    else:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards)
        assert np.all(torch_batch.next_rewards.numpy() == batch.next_rewards)

    assert np.all(torch_batch.terminals.numpy() == batch.terminals)
    assert np.all(torch_batch.n_steps.numpy() == batch.n_steps)


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


@pytest.mark.parametrize("in_shape", [(1, 2, 3)])
@pytest.mark.parametrize("out_shape", [(1, 6)])
def test_view(in_shape, out_shape):
    x = torch.rand(in_shape)
    view = View(out_shape)
    assert view(x).shape == out_shape


@pytest.mark.parametrize("in_shape", [(1, 2, 3)])
def test_swish(in_shape):
    x = torch.rand(in_shape)
    swish = Swish()
    y = swish(x)
    assert y.shape == in_shape
    assert torch.allclose(y, x * torch.sigmoid(x))
