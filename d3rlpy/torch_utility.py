import collections
import dataclasses
from typing import (
    Any,
    BinaryIO,
    Dict,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from typing_extensions import Self

from .dataclass_utils import asdict_without_copy
from .dataset import TrajectoryMiniBatch, TransitionMiniBatch
from .preprocessing import ActionScaler, ObservationScaler, RewardScaler
from .types import NDArray

__all__ = [
    "soft_sync",
    "hard_sync",
    "sync_optimizer_state",
    "map_location",
    "TorchMiniBatch",
    "TorchTrajectoryMiniBatch",
    "Checkpointer",
    "Modules",
    "convert_to_torch",
    "convert_to_torch_recursively",
    "eval_api",
    "train_api",
    "View",
]


def soft_sync(targ_model: nn.Module, model: nn.Module, tau: float) -> None:
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)


def hard_sync(targ_model: nn.Module, model: nn.Module) -> None:
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.copy_(p.data)


def sync_optimizer_state(targ_optim: Optimizer, optim: Optimizer) -> None:
    # source optimizer state
    state = optim.state_dict()["state"]
    # destination optimizer param_groups
    param_groups = targ_optim.state_dict()["param_groups"]
    # update only state
    targ_optim.load_state_dict({"state": state, "param_groups": param_groups})


def map_location(device: str) -> Any:
    if "cuda" in device:
        return lambda storage, loc: storage.cuda(device)
    if "cpu" in device:
        return "cpu"
    raise ValueError(f"invalid device={device}")


def convert_to_torch(array: NDArray, device: str) -> torch.Tensor:
    dtype = torch.uint8 if array.dtype == np.uint8 else torch.float32
    tensor = torch.tensor(data=array, dtype=dtype, device=device)
    return tensor.float()


@overload
def convert_to_torch_recursively(array: NDArray, device: str) -> torch.Tensor:
    ...


@overload
def convert_to_torch_recursively(
    array: Sequence[NDArray], device: str
) -> Sequence[torch.Tensor]:
    ...


def convert_to_torch_recursively(
    array: Union[NDArray, Sequence[NDArray]], device: str
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    if isinstance(array, (list, tuple)):
        return [convert_to_torch(data, device) for data in array]
    elif isinstance(array, np.ndarray):
        return convert_to_torch(array, device)
    else:
        raise ValueError(f"invalid array type: {type(array)}")


@dataclasses.dataclass(frozen=True)
class TorchMiniBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    returns_to_go: torch.Tensor
    terminals: torch.Tensor
    intervals: torch.Tensor
    device: str
    numpy_batch: Optional[TransitionMiniBatch] = None

    @classmethod
    def from_batch(
        cls,
        batch: TransitionMiniBatch,
        device: str,
        observation_scaler: Optional[ObservationScaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ) -> "TorchMiniBatch":
        # convert numpy array to torch tensor
        observations = convert_to_torch_recursively(batch.observations, device)
        actions = convert_to_torch(batch.actions, device)
        rewards = convert_to_torch(batch.rewards, device)
        next_observations = convert_to_torch_recursively(
            batch.next_observations, device
        )
        returns_to_go = convert_to_torch(batch.returns_to_go, device)
        terminals = convert_to_torch(batch.terminals, device)
        intervals = convert_to_torch(batch.intervals, device)

        # TODO: support tuple observation
        assert isinstance(observations, torch.Tensor)
        assert isinstance(next_observations, torch.Tensor)

        # apply scaler
        if observation_scaler:
            observations = observation_scaler.transform(observations)
            next_observations = observation_scaler.transform(next_observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)
            # NOTE: some operations might be incompatible with returns
            returns_to_go = reward_scaler.transform(returns_to_go)

        return TorchMiniBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            returns_to_go=returns_to_go,
            terminals=terminals,
            intervals=intervals,
            device=device,
            numpy_batch=batch,
        )


@dataclasses.dataclass(frozen=True)
class TorchTrajectoryMiniBatch:
    observations: torch.Tensor  # (B, L, ...)
    actions: torch.Tensor  # (B, L, ...)
    rewards: torch.Tensor  # (B, L, 1)
    returns_to_go: torch.Tensor  # (B, L, 1)
    terminals: torch.Tensor  # (B, L, 1)
    timesteps: torch.Tensor  # (B, L, 1)
    masks: torch.Tensor  # (B, L)
    device: str
    numpy_batch: Optional[TrajectoryMiniBatch] = None

    @classmethod
    def from_batch(
        cls,
        batch: TrajectoryMiniBatch,
        device: str,
        observation_scaler: Optional[ObservationScaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ) -> "TorchTrajectoryMiniBatch":
        # convert numpy array to torch tensor
        observations = convert_to_torch_recursively(batch.observations, device)
        actions = convert_to_torch(batch.actions, device)
        rewards = convert_to_torch(batch.rewards, device)
        returns_to_go = convert_to_torch(batch.returns_to_go, device)
        terminals = convert_to_torch(batch.terminals, device)
        timesteps = convert_to_torch(batch.timesteps, device).long()
        masks = convert_to_torch(batch.masks, device)

        # TODO: support tuple observation
        assert isinstance(observations, torch.Tensor)

        # apply scaler
        if observation_scaler:
            observations = observation_scaler.transform(observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)
            # NOTE: some operations might be incompatible with returns
            returns_to_go = reward_scaler.transform(returns_to_go)

        return TorchTrajectoryMiniBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            terminals=terminals,
            timesteps=timesteps,
            masks=masks,
            device=device,
            numpy_batch=batch,
        )


class Checkpointer:
    _modules: Dict[str, Union[nn.Module, Optimizer]]
    _device: str

    def __init__(
        self, modules: Dict[str, Union[nn.Module, Optimizer]], device: str
    ):
        self._modules = modules
        self._device = device

    def save(self, f: BinaryIO) -> None:
        states = {k: v.state_dict() for k, v in self._modules.items()}
        torch.save(states, f)

    def load(self, f: BinaryIO) -> None:
        chkpt = torch.load(f, map_location=map_location(self._device))
        for k, v in self._modules.items():
            v.load_state_dict(chkpt[k])

    @property
    def modules(self) -> Dict[str, Union[nn.Module, Optimizer]]:
        return self._modules


@dataclasses.dataclass(frozen=True)
class Modules:
    def create_checkpointer(self, device: str) -> Checkpointer:
        modules = {
            k: v
            for k, v in asdict_without_copy(self).items()
            if isinstance(v, (nn.Module, torch.optim.Optimizer))
        }
        return Checkpointer(modules=modules, device=device)

    def freeze(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, nn.Module):
                for p in v.parameters():
                    p.requires_grad = False

    def unfreeze(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, nn.Module):
                for p in v.parameters():
                    p.requires_grad = True

    def set_eval(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, nn.Module):
                v.eval()

    def set_train(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, nn.Module):
                v.train()

    def reset_optimizer_states(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, torch.optim.Optimizer):
                v.state = collections.defaultdict(dict)

    def wrap_models_by_ddp(self) -> Self:
        dict_values = asdict_without_copy(self)
        for k, v in dict_values.items():
            if isinstance(v, nn.Module):
                device_id = next(v.parameters()).device.index
                dict_values[k] = DDP(
                    v, device_ids=[device_id] if device_id else None
                )
        return self.__class__(**dict_values)

    def unwrap_models_by_ddp(self) -> Self:
        dict_values = asdict_without_copy(self)
        for k, v in dict_values.items():
            if isinstance(v, DDP):
                dict_values[k] = v.module
        return self.__class__(**dict_values)


TCallable = TypeVar("TCallable")


def eval_api(f: TCallable) -> TCallable:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        assert hasattr(self, "modules")
        assert isinstance(self.modules, Modules)
        self.modules.set_eval()
        return f(self, *args, **kwargs)  # type: ignore

    return wrapper  # type: ignore


def train_api(f: TCallable) -> TCallable:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        assert hasattr(self, "modules")
        assert isinstance(self.modules, Modules)
        self.modules.set_train()
        return f(self, *args, **kwargs)  # type: ignore

    return wrapper  # type: ignore


class View(nn.Module):  # type: ignore
    _shape: Sequence[int]

    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self._shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self._shape)


class Swish(nn.Module):  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
