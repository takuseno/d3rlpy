from typing import Any, Callable, Dict, List, Optional, cast
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
from typing_extensions import Protocol

from .preprocessing import Scaler, ActionScaler
from .augmentation import AugmentationPipeline


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


def set_eval_mode(impl: Any) -> None:
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            module.eval()


def set_train_mode(impl: Any) -> None:
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            module.train()


def to_cuda(impl: Any, device: str) -> None:
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.cuda(device)


def to_cpu(impl: Any) -> None:
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.cpu()


def freeze(impl: Any) -> None:
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad = False


def unfreeze(impl: Any) -> None:
    for key in dir(impl):
        module = getattr(impl, key)
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                p.requires_grad = True


def get_state_dict(impl: Any) -> Dict[str, Any]:
    rets = {}
    for key in dir(impl):
        obj = getattr(impl, key)
        if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
            rets[key] = obj.state_dict()
    return rets


def set_state_dict(impl: Any, chkpt: Dict[str, Any]) -> None:
    for key in dir(impl):
        obj = getattr(impl, key)
        if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
            obj.load_state_dict(chkpt[key])


def map_location(device: str) -> Any:
    if "cuda" in device:
        return lambda storage, loc: storage.cuda(device)
    if "cpu" in device:
        return "cpu"
    raise ValueError("invalid device={}".format(device))


class _WithDeviceAndScalerProtocol(Protocol):
    @property
    def device(self) -> str:
        ...

    @property
    def scaler(self) -> Optional[Scaler]:
        ...

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        ...


NUMPY_ARRAY_QUEUE = [0 for _ in range(6)]
TORCH_TENSOR_QUEUE = [None for _ in range(6)]


def _query_cache(array: np.ndarray, device: str) -> torch.Tensor:
    # cache hit
    if id(array) in NUMPY_ARRAY_QUEUE:
        tensor = TORCH_TENSOR_QUEUE[NUMPY_ARRAY_QUEUE.index(id(array))]
        tensor = cast(torch.Tensor, tensor)
        if array.shape == tensor.shape:
            return tensor

    NUMPY_ARRAY_QUEUE.pop(0)
    NUMPY_ARRAY_QUEUE.append(id(array))

    if array.dtype == np.uint8:
        dtype = torch.uint8
    else:
        dtype = torch.float32
    tensor = torch.tensor(data=array, dtype=dtype, device=device)

    TORCH_TENSOR_QUEUE.pop(0)
    TORCH_TENSOR_QUEUE.append(tensor)

    return tensor


def torch_api(
    scaler_targets: Optional[List[str]] = None,
    action_scaler_targets: Optional[List[str]] = None,
) -> Callable[..., np.ndarray]:
    def _torch_api(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        # get argument names
        sig = signature(f)
        arg_keys = list(sig.parameters.keys())[1:]

        def wrapper(
            self: _WithDeviceAndScalerProtocol, *args: np.ndarray, **kwargs: Any
        ) -> np.ndarray:
            # convert all args to torch.Tensor
            tensors = []
            for i, val in enumerate(args):
                if isinstance(val, torch.Tensor):
                    tensor = val
                elif isinstance(val, list):
                    tensor = default_collate(val)
                    tensor = tensor.to(self.device)
                elif isinstance(val, np.ndarray):
                    tensor = _query_cache(val, self.device)
                else:
                    tensor = torch.tensor(
                        data=val,
                        dtype=torch.float32,
                        device=self.device,
                    )

                # preprocess
                if self.scaler and scaler_targets:
                    if arg_keys[i] in scaler_targets:
                        tensor = self.scaler.transform(tensor)

                # preprocess action
                if self.action_scaler and action_scaler_targets:
                    if arg_keys[i] in action_scaler_targets:
                        tensor = self.action_scaler.transform(tensor)

                # make sure if the tensor is float32 type
                if tensor.dtype != torch.float32:
                    tensor = tensor.float()

                tensors.append(tensor)
            return f(self, *tensors, **kwargs)

        return wrapper

    return _torch_api


class _WithAugmentationProtocol(Protocol):
    @property
    def augmentation(self) -> AugmentationPipeline:
        ...


def augmentation_api(targets: List[str]) -> Callable[..., torch.Tensor]:
    def _augmentation_api(
        f: Callable[..., torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        sig = signature(f)
        arg_keys = list(sig.parameters.keys())[1:]

        def wrapper(
            self: _WithAugmentationProtocol, *args: torch.Tensor
        ) -> torch.Tensor:
            inputs: Dict[str, torch.Tensor] = {}
            for key, val in zip(arg_keys, args):
                inputs[key] = val
            inputs["self"] = self
            return self.augmentation.process(f, inputs, targets)

        return wrapper

    return _augmentation_api


def eval_api(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        set_eval_mode(self)
        return f(self, *args, **kwargs)

    return wrapper


def train_api(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        set_train_mode(self)
        return f(self, *args, **kwargs)

    return wrapper
