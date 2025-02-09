import collections
import dataclasses
from typing import (
    Any,
    BinaryIO,
    Generic,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda import CUDAGraph
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from typing_extensions import Self

from .dataclass_utils import asdict_without_copy
from .dataset import TrajectoryMiniBatch, TransitionMiniBatch
from .preprocessing import ActionScaler, ObservationScaler, RewardScaler
from .types import (
    Float32NDArray,
    NDArray,
    OptimizerWrapperProto,
    TorchObservation,
)

__all__ = [
    "soft_sync",
    "hard_sync",
    "sync_optimizer_state",
    "map_location",
    "TorchMiniBatch",
    "TorchTrajectoryMiniBatch",
    "wrap_model_by_ddp",
    "unwrap_ddp_model",
    "Checkpointer",
    "Modules",
    "convert_to_torch",
    "convert_to_torch_recursively",
    "convert_to_numpy_recursively",
    "get_device",
    "get_batch_size",
    "expand_and_repeat_recursively",
    "flatten_left_recursively",
    "eval_api",
    "train_api",
    "View",
    "CudaGraphWrapper",
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
        _, index = device.split(":")
        return lambda storage, loc: storage.cuda(int(index))
    if "cpu" in device:
        return "cpu"
    raise ValueError(f"invalid device={device}")


def convert_to_torch(array: NDArray, device: str) -> torch.Tensor:
    dtype = torch.uint8 if array.dtype == np.uint8 else torch.float32
    tensor = torch.tensor(data=array, dtype=dtype, device=device)
    return tensor.float()


@overload
def convert_to_torch_recursively(
    array: NDArray, device: str
) -> torch.Tensor: ...


@overload
def convert_to_torch_recursively(
    array: Sequence[NDArray], device: str
) -> Sequence[torch.Tensor]: ...


def convert_to_torch_recursively(
    array: Union[NDArray, Sequence[NDArray]], device: str
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    if isinstance(array, (list, tuple)):
        return [convert_to_torch(data, device) for data in array]
    elif isinstance(array, np.ndarray):
        return convert_to_torch(array, device)
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def convert_to_numpy_recursively(
    array: Union[torch.Tensor, Sequence[torch.Tensor]],
) -> Union[NDArray, Sequence[NDArray]]:
    if isinstance(array, (list, tuple)):
        return [data.numpy() for data in array]
    elif isinstance(array, torch.Tensor):
        return array.numpy()  # type: ignore
    else:
        raise ValueError(f"invalid array type: {type(array)}")


_T = TypeVar("_T", bound=Union[torch.Tensor, Sequence[torch.Tensor]])


def copy_recursively(src: _T, dst: _T) -> None:
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        dst.copy_(src)
    elif isinstance(src, (list, tuple)) and isinstance(dst, (list, tuple)):
        for s, d in zip(src, dst):
            d.copy_(s)
    else:
        raise ValueError(
            f"invalid inpu types: src={type(src)}, dst={type(dst)}"
        )


def get_device(x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> str:
    if isinstance(x, torch.Tensor):
        return str(x.device)
    else:
        return str(x[0].device)


def get_batch_size(x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.shape[0])
    else:
        return int(x[0].shape[0])


def flatten_left_recursively(
    x: Union[torch.Tensor, Sequence[torch.Tensor]], dim: int
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    if isinstance(x, torch.Tensor):
        return x.reshape([-1, *x.shape[dim + 1 :]])
    else:
        return [flatten_left_recursively(_x, dim) for _x in x]


def expand_and_repeat_recursively(
    x: Union[torch.Tensor, Sequence[torch.Tensor]], n: int
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    if isinstance(x, torch.Tensor):
        # repeat observation
        # (batch_size, M) -> (batch_size, 1, M)
        reshaped_x = x.view(x.shape[0], 1, *x.shape[1:])
        # (batch_sie, 1, M) -> (batch_size, N, M)
        return reshaped_x.expand(x.shape[0], n, *x.shape[1:])
    else:
        return [expand_and_repeat_recursively(_x, n) for _x in x]


def _compute_return_to_go(
    gamma: float,
    rewards_to_go: Float32NDArray,
    reward_scaler: Optional[RewardScaler],
) -> Float32NDArray:
    rewards = (
        reward_scaler.transform_numpy(rewards_to_go)
        if reward_scaler
        else rewards_to_go
    )
    cum_gammas: Float32NDArray = np.array(
        np.expand_dims(gamma ** np.arange(rewards.shape[0]), axis=1),
        dtype=np.float32,
    )
    return np.sum(cum_gammas * rewards, axis=0)  # type: ignore


@dataclasses.dataclass(frozen=True)
class TorchMiniBatch:
    observations: TorchObservation
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: TorchObservation
    next_actions: torch.Tensor
    returns_to_go: torch.Tensor
    terminals: torch.Tensor
    intervals: torch.Tensor
    device: str
    numpy_batch: Optional[TransitionMiniBatch] = None

    @classmethod
    def from_batch(
        cls,
        batch: TransitionMiniBatch,
        gamma: float,
        compute_returns_to_go: bool,
        device: str,
        observation_scaler: Optional[ObservationScaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ) -> "TorchMiniBatch":
        # convert numpy array to torch tensor
        observations = convert_to_torch_recursively(batch.observations, device)
        actions = convert_to_torch(batch.actions, device)
        next_actions = convert_to_torch(batch.next_actions, device)
        rewards = convert_to_torch(batch.rewards, device)
        next_observations = convert_to_torch_recursively(
            batch.next_observations, device
        )
        terminals = convert_to_torch(batch.terminals, device)
        intervals = convert_to_torch(batch.intervals, device)

        if compute_returns_to_go:
            returns_to_go = convert_to_torch(
                np.array(
                    [
                        _compute_return_to_go(
                            gamma=gamma,
                            rewards_to_go=transition.rewards_to_go,
                            reward_scaler=reward_scaler,
                        )
                        for transition in batch.transitions
                    ]
                ),
                device,
            )
        else:
            returns_to_go = torch.zeros_like(rewards)

        # apply scaler
        if observation_scaler:
            observations = observation_scaler.transform(observations)
            next_observations = observation_scaler.transform(next_observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
            next_actions = action_scaler.transform(next_actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)

        return TorchMiniBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_actions=next_actions,
            returns_to_go=returns_to_go,
            terminals=terminals,
            intervals=intervals,
            device=device,
            numpy_batch=batch,
        )

    def copy_(self, src: Self) -> None:
        assert self.device == src.device, "incompatible device"
        copy_recursively(src.observations, self.observations)
        self.actions.copy_(src.actions)
        self.rewards.copy_(src.rewards)
        copy_recursively(src.next_observations, self.next_observations)
        self.next_actions.copy_(src.next_actions)
        self.returns_to_go.copy_(src.returns_to_go)
        self.terminals.copy_(src.terminals)
        self.intervals.copy_(src.intervals)


@dataclasses.dataclass(frozen=True)
class TorchTrajectoryMiniBatch:
    observations: TorchObservation  # (B, L, ...)
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

    def copy_(self, src: Self) -> None:
        assert self.device == src.device, "incompatible device"
        copy_recursively(src.observations, self.observations)
        self.actions.copy_(src.actions)
        self.rewards.copy_(src.rewards)
        self.returns_to_go.copy_(src.returns_to_go)
        self.terminals.copy_(src.terminals)
        self.timesteps.copy_(src.timesteps)
        self.masks.copy_(src.masks)


_TModule = TypeVar("_TModule", bound=nn.Module)


def wrap_model_by_ddp(model: _TModule) -> _TModule:
    device_id = next(model.parameters()).device.index
    return DDP(model, device_ids=[device_id] if device_id else None)  # type: ignore


def unwrap_ddp_model(model: _TModule) -> _TModule:
    if isinstance(model, DDP):
        model = model.module
    if isinstance(model, nn.ModuleList):
        module_list = nn.ModuleList()
        for v in model:
            module_list.append(unwrap_ddp_model(v))
        model = module_list
    return model


class Checkpointer:
    _modules: dict[str, Union[nn.Module, OptimizerWrapperProto]]
    _device: str

    def __init__(
        self,
        modules: dict[str, Union[nn.Module, OptimizerWrapperProto]],
        device: str,
    ):
        self._modules = modules
        self._device = device

    def save(self, f: BinaryIO) -> None:
        # unwrap DDP
        modules = {
            k: unwrap_ddp_model(v) if isinstance(v, nn.Module) else v
            for k, v in self._modules.items()
        }
        states = {k: v.state_dict() for k, v in modules.items()}
        torch.save(states, f)

    def load(self, f: BinaryIO) -> None:
        chkpt = torch.load(f, map_location=map_location(self._device))
        for k, v in self._modules.items():
            v.load_state_dict(chkpt[k])

    @property
    def modules(self) -> dict[str, Union[nn.Module, OptimizerWrapperProto]]:
        return self._modules


@dataclasses.dataclass(frozen=True)
class Modules:
    def create_checkpointer(self, device: str) -> Checkpointer:
        modules = {
            k: v
            for k, v in asdict_without_copy(self).items()
            if isinstance(v, (nn.Module, OptimizerWrapperProto))
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
            if isinstance(v, nn.Module) and v.training:
                v.eval()

    def set_train(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, nn.Module) and not v.training:
                v.train()

    def reset_optimizer_states(self) -> None:
        for v in asdict_without_copy(self).values():
            if isinstance(v, OptimizerWrapperProto):
                v.optim.state = collections.defaultdict(dict)

    def get_torch_modules(self) -> dict[str, nn.Module]:
        torch_modules: dict[str, nn.Module] = {}
        for k, v in asdict_without_copy(self).items():
            if isinstance(v, nn.Module):
                torch_modules[k] = v
        return torch_modules

    def get_gradients(self) -> Iterator[tuple[str, Float32NDArray]]:
        for module_name, module in self.get_torch_modules().items():
            for name, parameter in module.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    yield (
                        f"{module_name}.{name}",
                        parameter.grad.cpu().detach().numpy(),
                    )


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


class GEGLU(nn.Module):  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


BatchT_contra = TypeVar(
    "BatchT_contra",
    bound=Union[TorchMiniBatch, TorchTrajectoryMiniBatch],
    contravariant=True,
)
RetT_co = TypeVar("RetT_co", covariant=True)


class CudaGraphFunc(Generic[BatchT_contra, RetT_co], Protocol):
    def __call__(self, batch: BatchT_contra) -> RetT_co: ...


class CudaGraphWrapper(Generic[BatchT_contra, RetT_co]):
    _func: CudaGraphFunc[BatchT_contra, RetT_co]
    _input: TorchTrajectoryMiniBatch
    _graph: Optional[CUDAGraph]
    _inpt: Optional[BatchT_contra]
    _out: Optional[RetT_co]

    def __init__(
        self,
        func: CudaGraphFunc[BatchT_contra, RetT_co],
        warmup_steps: int = 3,
        compile_func: bool = True,
    ):
        self._func = torch.compile(func) if compile_func else func
        self._step = 0
        self._graph = None
        self._inpt = None
        self._out = None
        self._warmup_steps = warmup_steps
        self._warmup_stream = torch.cuda.Stream()

    def __call__(self, batch: BatchT_contra) -> RetT_co:
        if self._step < self._warmup_steps:  # warmup
            self._warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._warmup_stream):
                out = self._func(batch)
            torch.cuda.current_stream().wait_stream(self._warmup_stream)
        if self._step == self._warmup_steps - 1:  # build graph
            self._graph = torch.cuda.CUDAGraph()
            self._inpt = batch
            with torch.cuda.graph(self._graph):
                self._out = self._func(self._inpt)
        if self._step >= self._warmup_steps:  # reuse cuda graph
            assert self._inpt
            assert self._out
            assert self._graph
            with torch.no_grad():
                self._inpt.copy_(batch)  # type: ignore
            self._graph.replay()
            out = self._out
        self._step += 1
        return out
