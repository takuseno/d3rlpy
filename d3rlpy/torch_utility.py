import collections
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from .dataset import TransitionMiniBatch
from .preprocessing import ActionScaler, ObservationScaler, RewardScaler

__all__ = [
    "soft_sync",
    "hard_sync",
    "sync_optimizer_state",
    "set_eval_mode",
    "set_train_mode",
    "to_cuda",
    "to_cpu",
    "to_device",
    "freeze",
    "unfreeze",
    "get_state_dict",
    "set_state_dict",
    "reset_optimizer_states",
    "map_location",
    "TorchMiniBatch",
    "convert_to_torch",
    "convert_to_torch_recursively",
    "eval_api",
    "train_api",
    "View",
]


BLACK_LIST = [
    "policy",
    "q_function",
    "policy_optim",
    "q_function_optim",
]  # special properties


def _get_attributes(obj: Any) -> List[str]:
    return [key for key in dir(obj) if key not in BLACK_LIST]


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


def set_eval_mode(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            module.eval()


def set_train_mode(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            module.train()


def to_cuda(impl: Any, device: str) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.cuda(device)


def to_cpu(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.cpu()


def to_device(impl: Any, device: str) -> None:
    if device.startswith("cuda"):
        to_cuda(impl, device)
    else:
        to_cpu(impl)


def freeze(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad = False


def unfreeze(impl: Any) -> None:
    for key in _get_attributes(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad = True


def get_state_dict(impl: Any) -> Dict[str, Any]:
    rets = {}
    for key in _get_attributes(impl):
        obj = getattr(impl, key)
        if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
            rets[key] = obj.state_dict()
    return rets


def set_state_dict(impl: Any, chkpt: Dict[str, Any]) -> None:
    for key in _get_attributes(impl):
        obj = getattr(impl, key)
        if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
            obj.load_state_dict(chkpt[key])


def reset_optimizer_states(impl: Any) -> None:
    for key in _get_attributes(impl):
        obj = getattr(impl, key)
        if isinstance(obj, torch.optim.Optimizer):
            obj.state = collections.defaultdict(dict)


def map_location(device: str) -> Any:
    if "cuda" in device:
        return lambda storage, loc: storage.cuda(device)
    if "cpu" in device:
        return "cpu"
    raise ValueError(f"invalid device={device}")


def convert_to_torch(array: np.ndarray, device: str) -> torch.Tensor:
    dtype = torch.uint8 if array.dtype == np.uint8 else torch.float32
    tensor = torch.tensor(data=array, dtype=dtype, device=device)
    return tensor.float()


def convert_to_torch_recursively(
    array: Union[np.ndarray, Sequence[np.ndarray]], device: str
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

        return TorchMiniBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            intervals=intervals,
            device=device,
            numpy_batch=batch,
        )


TCallable = TypeVar("TCallable")


def eval_api(f: TCallable) -> TCallable:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        set_eval_mode(self)
        return f(self, *args, **kwargs)  # type: ignore

    return wrapper  # type: ignore


def train_api(f: TCallable) -> TCallable:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        set_train_mode(self)
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
