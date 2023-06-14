import os
from typing import Any, Dict

import numpy as np
from tensorboardX import SummaryWriter

from .logger import LoggerAdapter, LoggerAdapterFactory, SaveProtocol

__all__ = ["TensorboardAdapter", "TensorboardAdapterFactory"]


class TensorboardAdapter(LoggerAdapter):
    _experiment_name: str
    _writer: SummaryWriter
    _params: Dict[str, Any]
    _metrics: Dict[str, float]

    def __init__(self, root_dir: str, experiment_name: str):
        self._experiment_name = experiment_name
        logdir = os.path.join(root_dir, "runs", experiment_name)
        print(logdir)
        self._writer = SummaryWriter(logdir=logdir)
        self._metrics = {}

    def write_params(self, params: Dict[str, Any]) -> None:
        # remove non-scaler values for HParams
        self._params = {k: v for k, v in params.items() if np.isscalar(v)}

    def before_write_metric(self, epoch: int, step: int) -> None:
        self._metrics = {}

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        # self._writer.add_scalar(f"metrics/{name}", value, epoch)
        self._metrics[name] = value

    def after_write_metric(self, epoch: int, step: int) -> None:
        self._writer.add_hparams(
            self._params,
            self._metrics,
            name=self._experiment_name,
            global_step=epoch,
        )

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        pass

    def close(self) -> None:
        self._writer.close()


class TensorboardAdapterFactory(LoggerAdapterFactory):
    _root_dir: str

    def __init__(self, root_dir: str = "tensorboard_logs"):
        self._root_dir = root_dir

    def create(self, experiment_name: str) -> TensorboardAdapter:
        return TensorboardAdapter(self._root_dir, experiment_name)
