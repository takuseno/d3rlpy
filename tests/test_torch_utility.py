import copy
from typing import Any, Dict, Sequence
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from d3rlpy.dataset import TrajectoryMiniBatch, Transition, TransitionMiniBatch
from d3rlpy.preprocessing import ActionScaler, ObservationScaler, RewardScaler
from d3rlpy.torch_utility import (
    Swish,
    TorchMiniBatch,
    TorchTrajectoryMiniBatch,
    View,
    eval_api,
    freeze,
    get_state_dict,
    hard_sync,
    map_location,
    reset_optimizer_states,
    set_eval_mode,
    set_state_dict,
    set_train_mode,
    soft_sync,
    sync_optimizer_state,
    train_api,
    unfreeze,
)

from .testing_utils import create_partial_trajectory


@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("input_size", [32])
@pytest.mark.parametrize("output_size", [32])
def test_soft_sync(tau: float, input_size: int, output_size: int) -> None:
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
def test_hard_sync(input_size: int, output_size: int) -> None:
    module = torch.nn.Linear(input_size, output_size)
    targ_module = torch.nn.Linear(input_size, output_size)

    hard_sync(targ_module, module)

    for p, targ_p in zip(module.parameters(), targ_module.parameters()):
        assert torch.allclose(targ_p, p)


@pytest.mark.parametrize("input_size", [32])
@pytest.mark.parametrize("output_size", [1])
def test_sync_optimizer_state(input_size: int, output_size: int) -> None:
    module = torch.nn.Linear(input_size, output_size)

    # lr=1e-3
    optim = torch.optim.Adam(module.parameters(), lr=1e-3)

    # instantiate state values
    y = module(torch.rand(input_size))
    y.backward()
    optim.step()

    # lr=1e-4
    targ_optim = torch.optim.Adam(module.parameters(), lr=1e-4)

    sync_optimizer_state(targ_optim, optim)

    # check if lr is not synced
    assert targ_optim.param_groups[0]["lr"] != optim.param_groups[0]["lr"]

    # check if state is synced
    targ_state = targ_optim.state_dict()["state"]
    state = optim.state_dict()["state"]
    for i, l in targ_state.items():
        for k, v in l.items():
            if isinstance(v, int):
                assert v == state[i][k]
            else:
                assert torch.allclose(v, state[i][k])


def test_map_location_with_cpu() -> None:
    assert map_location("cpu:0") == "cpu"


def test_map_location_with_cuda() -> None:
    fn = map_location("cuda:0")
    dummy = Mock()
    dummy.cuda = Mock()

    fn(dummy, "")

    dummy.cuda.assert_called_with("cuda:0")


class DummyImpl:
    def __init__(self) -> None:
        self._fc1 = torch.nn.Linear(100, 100)
        self._fc2 = torch.nn.Linear(100, 100)
        self._optim = torch.optim.Adam(self._fc1.parameters())
        self._device = "cpu:0"

    @train_api
    def train_api_func(self) -> None:
        assert self._fc1.training
        assert self._fc2.training

    @eval_api
    def eval_api_func(self) -> None:
        assert not self._fc1.training
        assert not self._fc2.training

    @property
    def device(self) -> str:
        return self._device


def check_if_same_dict(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    for k, v in a.items():
        if isinstance(v, torch.Tensor):
            assert (b[k] == v).all()
        else:
            assert b[k] == v


def test_get_state_dict() -> None:
    impl = DummyImpl()

    state_dict = get_state_dict(impl)

    check_if_same_dict(state_dict["_fc1"], impl._fc1.state_dict())
    check_if_same_dict(state_dict["_fc2"], impl._fc2.state_dict())
    check_if_same_dict(state_dict["_optim"], impl._optim.state_dict())


def test_set_state_dict() -> None:
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


def test_reset_optimizer_states() -> None:
    impl = DummyImpl()

    # instantiate optimizer state
    y = impl._fc1(torch.rand(100)).sum()
    y.backward()
    impl._optim.step()

    # check if state is not empty
    state = copy.deepcopy(impl._optim.state)
    assert state

    reset_optimizer_states(impl)

    # check if state is empty
    reset_state = impl._optim.state
    assert not reset_state


def test_eval_mode() -> None:
    impl = DummyImpl()
    impl._fc1.train()
    impl._fc2.train()

    set_eval_mode(impl)

    assert not impl._fc1.training
    assert not impl._fc2.training


def test_train_mode() -> None:
    impl = DummyImpl()
    impl._fc1.eval()
    impl._fc2.eval()

    set_train_mode(impl)

    assert impl._fc1.training
    assert impl._fc2.training


@pytest.mark.skip(reason="no way to test this")
def test_to_cuda() -> None:
    pass


@pytest.mark.skip(reason="no way to test this")
def test_to_cpu() -> None:
    pass


def test_freeze() -> None:
    impl = DummyImpl()

    freeze(impl)

    for p in impl._fc1.parameters():
        assert not p.requires_grad
    for p in impl._fc2.parameters():
        assert not p.requires_grad


def test_unfreeze() -> None:
    impl = DummyImpl()

    freeze(impl)
    unfreeze(impl)

    for p in impl._fc1.parameters():
        assert p.requires_grad
    for p in impl._fc2.parameters():
        assert p.requires_grad


class DummyObservationScaler(ObservationScaler):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1


class DummyActionScaler(ActionScaler):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2


class DummyRewardScaler(RewardScaler):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("use_observation_scaler", [False, True])
@pytest.mark.parametrize("use_action_scaler", [False, True])
@pytest.mark.parametrize("use_reward_scaler", [False, True])
def test_torch_mini_batch(
    batch_size: int,
    observation_shape: Sequence[int],
    action_size: int,
    use_observation_scaler: bool,
    use_action_scaler: bool,
    use_reward_scaler: bool,
) -> None:
    obs_shape = (batch_size, *observation_shape)
    transitions = []
    for _ in range(batch_size):
        transition = Transition(
            observation=np.random.random(obs_shape),
            action=np.random.random(action_size),
            reward=np.random.random((1,)),
            next_observation=np.random.random(obs_shape),
            terminal=0.0,
            interval=1,
        )
        transitions.append(transition)

    if use_observation_scaler:
        observation_scaler = DummyObservationScaler()
    else:
        observation_scaler = None

    if use_action_scaler:
        action_scaler = DummyActionScaler()
    else:
        action_scaler = None

    if use_reward_scaler:
        reward_scaler = DummyRewardScaler()
    else:
        reward_scaler = None

    batch = TransitionMiniBatch.from_transitions(transitions)

    torch_batch = TorchMiniBatch.from_batch(
        batch=batch,
        device="cpu:0",
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )

    assert isinstance(batch.observations, np.ndarray)
    assert isinstance(batch.next_observations, np.ndarray)
    if use_observation_scaler:
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
    else:
        assert np.all(torch_batch.actions.numpy() == batch.actions)

    if use_reward_scaler:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards + 0.2)
    else:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards)

    assert np.all(torch_batch.terminals.numpy() == batch.terminals)
    assert np.all(torch_batch.intervals.numpy() == batch.intervals)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("length", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("use_observation_scaler", [False, True])
@pytest.mark.parametrize("use_action_scaler", [False, True])
@pytest.mark.parametrize("use_reward_scaler", [False, True])
def test_torch_trajectory_mini_batch(
    batch_size: int,
    length: int,
    observation_shape: Sequence[int],
    action_size: int,
    use_observation_scaler: bool,
    use_action_scaler: bool,
    use_reward_scaler: bool,
) -> None:
    trajectories = []
    for _ in range(batch_size):
        trajectory = create_partial_trajectory(
            observation_shape, action_size, length
        )
        trajectories.append(trajectory)

    if use_observation_scaler:
        observation_scaler = DummyObservationScaler()
    else:
        observation_scaler = None

    if use_action_scaler:
        action_scaler = DummyActionScaler()
    else:
        action_scaler = None

    if use_reward_scaler:
        reward_scaler = DummyRewardScaler()
    else:
        reward_scaler = None

    batch = TrajectoryMiniBatch.from_partial_trajectories(trajectories)

    torch_batch = TorchTrajectoryMiniBatch.from_batch(
        batch=batch,
        device="cpu:0",
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )

    assert isinstance(batch.observations, np.ndarray)
    if use_observation_scaler:
        assert np.all(
            torch_batch.observations.numpy() == batch.observations + 0.1
        )
    else:
        assert np.all(torch_batch.observations.numpy() == batch.observations)

    if use_action_scaler:
        assert np.all(torch_batch.actions.numpy() == batch.actions + 0.2)
    else:
        assert np.all(torch_batch.actions.numpy() == batch.actions)

    if use_reward_scaler:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards + 0.2)
        assert np.all(
            torch_batch.returns_to_go.numpy() == batch.returns_to_go + 0.2
        )
    else:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards)
        assert np.all(torch_batch.returns_to_go.numpy() == batch.returns_to_go)

    assert np.all(torch_batch.terminals.numpy() == batch.terminals)


def test_train_api() -> None:
    impl = DummyImpl()
    impl._fc1.eval()
    impl._fc2.eval()

    impl.train_api_func()


def test_eval_api() -> None:
    impl = DummyImpl()
    impl._fc1.train()
    impl._fc2.train()

    impl.eval_api_func()


@pytest.mark.parametrize("in_shape", [(1, 2, 3)])
@pytest.mark.parametrize("out_shape", [(1, 6)])
def test_view(in_shape: Sequence[int], out_shape: Sequence[int]) -> None:
    x = torch.rand(in_shape)
    view = View(out_shape)
    assert view(x).shape == out_shape


@pytest.mark.parametrize("in_shape", [(1, 2, 3)])
def test_swish(in_shape: Sequence[int]) -> None:
    x = torch.rand(in_shape)
    swish = Swish()
    y = swish(x)
    assert y.shape == in_shape
    assert torch.allclose(y, x * torch.sigmoid(x))
