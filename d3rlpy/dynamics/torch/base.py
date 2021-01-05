import numpy as np
import torch

from typing import Optional, Sequence, Tuple
from abc import abstractmethod
from ..base import DynamicsImplBase
from ...gpu import Device
from ...preprocessing import Scaler
from ...torch_utility import to_cuda, to_cpu
from ...torch_utility import torch_api, eval_api
from ...torch_utility import map_location
from ...torch_utility import get_state_dict, set_state_dict


class TorchImplBase(DynamicsImplBase):

    _observation_shape: Sequence[int]
    _action_size: int
    _scaler: Optional[Scaler]
    _device: str

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        scaler: Optional[Scaler],
    ):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._scaler = scaler
        self._device = "cpu:0"

    @eval_api
    @torch_api(scaler_targets=["x"])
    def predict(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            observation, reward, variance = self._predict(x, action)

            if self._scaler:
                observation = self._scaler.reverse_transform(observation)

        observation = observation.cpu().detach().numpy()
        reward = reward.cpu().detach().numpy()
        variance = variance.cpu().detach().numpy()

        return observation, reward, variance

    @abstractmethod
    def _predict(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @eval_api
    @torch_api(scaler_targets=["x"])
    def generate(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            observation, reward = self._generate(x, action)

            if self._scaler:
                observation = self._scaler.reverse_transform(observation)

        observation = observation.cpu().detach().numpy()
        reward = reward.cpu().detach().numpy()
        return observation, reward

    @abstractmethod
    def _generate(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

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
