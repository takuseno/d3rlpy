from typing import Optional, Sequence

import numpy as np
import torch

from ...gpu import Device
from ...preprocessing import ActionScaler, Scaler
from ...torch_utility import (
    eval_api,
    freeze,
    get_state_dict,
    map_location,
    set_state_dict,
    to_cpu,
    to_cuda,
    torch_api,
    unfreeze,
)
from ..base import AlgoImplBase


class TorchImplBase(AlgoImplBase):

    _observation_shape: Sequence[int]
    _action_size: int
    _scaler: Optional[Scaler]
    _action_scaler: Optional[ActionScaler]
    _device: str

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
    ):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._scaler = scaler
        self._action_scaler = action_scaler
        self._device = "cpu:0"

    @eval_api
    @torch_api(scaler_targets=["x"])
    def predict_best_action(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action = self._predict_best_action(x)

            # transform action back to the original range
            if self._action_scaler:
                action = self._action_scaler.reverse_transform(action)

            return action.cpu().detach().numpy()

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @eval_api
    @torch_api(scaler_targets=["x"])
    def sample_action(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action = self._sample_action(x)

            # transform action back to the original range
            if self._action_scaler:
                action = self._action_scaler.reverse_transform(action)

            return action.cpu().detach().numpy()

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @eval_api
    def save_policy(self, fname: str, as_onnx: bool) -> None:
        dummy_x = torch.rand(1, *self.observation_shape, device=self._device)

        # workaround until version 1.6
        freeze(self)

        # dummy function to select best actions
        def _func(x: torch.Tensor) -> torch.Tensor:
            if self._scaler:
                x = self._scaler.transform(x)

            action = self._predict_best_action(x)

            if self._action_scaler:
                action = self._action_scaler.reverse_transform(action)

            return action

        traced_script = torch.jit.trace(_func, dummy_x, check_trace=False)

        if as_onnx:
            # currently, PyTorch cannot directly export function as ONNX.
            torch.onnx.export(
                traced_script,
                dummy_x,
                fname,
                export_params=True,
                opset_version=11,
                input_names=["input_0"],
                output_names=["output_0"],
                example_outputs=traced_script(dummy_x),
            )
        else:
            traced_script.save(fname)

        # workaround until version 1.6
        unfreeze(self)

    def to_gpu(self, device: Device = Device()) -> None:
        self._device = "cuda:%d" % device.get_id()
        to_cuda(self, self._device)

    def to_cpu(self) -> None:
        self._device = "cpu:0"
        to_cpu(self)

    def save_model(self, fname: str) -> None:
        torch.save(get_state_dict(self), fname)

    def load_model(self, fname: str) -> None:
        chkpt = torch.load(fname, map_location=map_location(self._device))
        set_state_dict(self, chkpt)

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def scaler(self) -> Optional[Scaler]:
        return self._scaler

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        return self._action_scaler
