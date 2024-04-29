import copy
import dataclasses
from io import BytesIO
from typing import Any, Dict, Sequence
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from d3rlpy.dataset import TrajectoryMiniBatch, Transition, TransitionMiniBatch
from d3rlpy.torch_utility import (
    GEGLU,
    Checkpointer,
    Modules,
    Swish,
    TorchMiniBatch,
    TorchTrajectoryMiniBatch,
    View,
    eval_api,
    get_batch_size,
    get_device,
    hard_sync,
    map_location,
    soft_sync,
    sync_optimizer_state,
    train_api,
)

from .dummy_scalers import (
    DummyActionScaler,
    DummyObservationScaler,
    DummyRewardScaler,
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
        self.fc1 = torch.nn.Linear(100, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.optim = torch.optim.Adam(self.fc1.parameters())
        self.modules = DummyModules(self.fc1, self.optim)
        self.device = "cpu:0"

    @train_api
    def train_api_func(self) -> None:
        assert self.fc1.training
        assert not self.fc2.training

    @eval_api
    def eval_api_func(self) -> None:
        assert not self.fc1.training
        assert self.fc2.training


def check_if_same_dict(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    for k, v in a.items():
        if isinstance(v, torch.Tensor):
            assert (b[k] == v).all()
        else:
            assert b[k] == v


def test_reset_optimizer_states() -> None:
    impl = DummyImpl()

    # instantiate optimizer state
    y = impl.fc1(torch.rand(100)).sum()
    y.backward()
    impl.optim.step()

    # check if state is not empty
    state = copy.deepcopy(impl.optim.state)
    assert state

    impl.modules.reset_optimizer_states()

    # check if state is empty
    reset_state = impl.optim.state
    assert not reset_state


@pytest.mark.skip(reason="no way to test this")
def test_to_cuda() -> None:
    pass


@pytest.mark.skip(reason="no way to test this")
def test_to_cpu() -> None:
    pass


def test_get_device() -> None:
    x = torch.rand(10)
    assert get_device(x) == "cpu"
    x = [torch.rand(10), torch.rand(10)]
    assert get_device(x) == "cpu"


def test_get_batch_size() -> None:
    x = torch.rand(32, 10)
    assert get_batch_size(x) == 32
    x = [torch.rand(32, 10), torch.rand(32, 10)]
    assert get_batch_size(x) == 32


@dataclasses.dataclass(frozen=True)
class DummyModules(Modules):
    fc: torch.nn.Linear
    optim: torch.optim.Adam


def test_modules() -> None:
    fc = torch.nn.Linear(100, 200)
    optim = torch.optim.Adam(fc.parameters())
    modules = DummyModules(fc, optim)

    # check checkpointer
    checkpointer = modules.create_checkpointer("cpu:0")
    assert "fc" in checkpointer.modules
    assert "optim" in checkpointer.modules
    assert checkpointer.modules["fc"] is fc
    assert checkpointer.modules["optim"] is optim

    # check freeze
    modules.freeze()
    for p in fc.parameters():
        assert not p.requires_grad

    # check unfreeze
    modules.unfreeze()
    for p in fc.parameters():
        assert p.requires_grad


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("use_observation_scaler", [False, True])
@pytest.mark.parametrize("use_action_scaler", [False, True])
@pytest.mark.parametrize("use_reward_scaler", [False, True])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("compute_returns_to_go", [False, True])
def test_torch_mini_batch(
    batch_size: int,
    observation_shape: Sequence[int],
    action_size: int,
    use_observation_scaler: bool,
    use_action_scaler: bool,
    use_reward_scaler: bool,
    gamma: float,
    compute_returns_to_go: bool,
) -> None:
    obs_shape = (batch_size, *observation_shape)
    transitions = []
    for _ in range(batch_size):
        transition = Transition(
            observation=np.random.random(obs_shape),
            action=np.random.random(action_size),
            reward=np.random.random((1,)).astype(np.float32),
            next_observation=np.random.random(obs_shape),
            rewards_to_go=np.random.random((10, 1)).astype(np.float32),
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
        gamma=gamma,
        compute_returns_to_go=compute_returns_to_go,
        device="cpu:0",
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )

    ref_returns_to_go = []
    if compute_returns_to_go:
        for transition in transitions:
            rewards_to_go = transition.rewards_to_go
            if reward_scaler:
                rewards_to_go = reward_scaler.transform(rewards_to_go)
            R = 0.0
            for i, r in enumerate(np.reshape(rewards_to_go, [-1])):
                R += r * (gamma**i)
            ref_returns_to_go.append([R])
    else:
        ref_returns_to_go.extend([[0.0]] * batch_size)

    assert isinstance(batch.observations, np.ndarray)
    assert isinstance(batch.next_observations, np.ndarray)
    assert isinstance(torch_batch.observations, torch.Tensor)
    assert isinstance(torch_batch.next_observations, torch.Tensor)
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
        assert np.all(torch_batch.rewards.numpy() == batch.rewards + 0.3)
    else:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards)

    assert np.allclose(
        torch_batch.returns_to_go.numpy(), np.array(ref_returns_to_go)
    )
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
    assert isinstance(torch_batch.observations, torch.Tensor)
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
        assert np.all(torch_batch.rewards.numpy() == batch.rewards + 0.3)
        assert np.all(
            torch_batch.returns_to_go.numpy() == batch.returns_to_go + 0.3
        )
    else:
        assert np.all(torch_batch.rewards.numpy() == batch.rewards)
        assert np.all(torch_batch.returns_to_go.numpy() == batch.returns_to_go)

    assert np.all(torch_batch.terminals.numpy() == batch.terminals)


def test_checkpointer() -> None:
    fc1 = torch.nn.Linear(100, 100)
    fc2 = torch.nn.Linear(100, 100)
    optim = torch.optim.Adam(fc1.parameters())
    checkpointer = Checkpointer(
        modules={"fc1": fc1, "fc2": fc2, "optim": optim}, device="cpu:0"
    )

    # prepare reference bytes
    ref_bytes = BytesIO()
    states = {
        "fc1": fc1.state_dict(),
        "fc2": fc2.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(states, ref_bytes)

    # check saved bytes
    saved_bytes = BytesIO()
    checkpointer.save(saved_bytes)
    assert ref_bytes.getvalue() == saved_bytes.getvalue()

    fc1_2 = torch.nn.Linear(100, 100)
    fc2_2 = torch.nn.Linear(100, 100)
    optim_2 = torch.optim.Adam(fc1_2.parameters())
    checkpointer = Checkpointer(
        modules={"fc1": fc1_2, "fc2": fc2_2, "optim": optim_2}, device="cpu:0"
    )

    # check load
    checkpointer.load(BytesIO(saved_bytes.getvalue()))

    # check output
    x = torch.rand(32, 100)
    y1_ref = fc1(x)
    y2_ref = fc2(x)
    y1 = fc1_2(x)
    y2 = fc2_2(x)
    assert torch.all(y1_ref == y1)
    assert torch.all(y2_ref == y2)


def test_train_api() -> None:
    impl = DummyImpl()
    impl.fc1.eval()
    impl.fc2.eval()

    impl.train_api_func()


def test_eval_api() -> None:
    impl = DummyImpl()
    impl.fc1.train()
    impl.fc2.train()

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


@pytest.mark.parametrize("in_shape", [(1, 2, 4)])
def test_geglu(in_shape: Sequence[int]) -> None:
    x = torch.rand(in_shape)
    geglu = GEGLU()
    y = geglu(x)
    ref_shape = list(in_shape)
    ref_shape[-1] = ref_shape[-1] // 2
    assert y.shape == tuple(ref_shape)
