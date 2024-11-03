# pylint: disable=protected-access
import random

import gymnasium
import numpy as np
import torch

from . import (
    algos,
    dataset,
    datasets,
    distributed,
    envs,
    logging,
    metrics,
    models,
    notebook_utils,
    ope,
    optimizers,
    preprocessing,
    tokenizers,
    types,
)
from ._version import __version__
from .base import load_learnable
from .constants import ActionSpace, LoggingStrategy, PositionEncodingType
from .healthcheck import run_healthcheck
from .torch_utility import Modules, TorchMiniBatch

__all__ = [
    "algos",
    "dataset",
    "datasets",
    "distributed",
    "envs",
    "logging",
    "metrics",
    "models",
    "optimizers",
    "notebook_utils",
    "ope",
    "preprocessing",
    "tokenizers",
    "types",
    "__version__",
    "load_learnable",
    "ActionSpace",
    "LoggingStrategy",
    "PositionEncodingType",
    "Modules",
    "TorchMiniBatch",
    "seed",
]


def seed(n: int) -> None:
    """Sets random seed value.

    Args:
        n (int): seed value.
    """
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.backends.cudnn.deterministic = True


# run healthcheck
run_healthcheck()

if torch.cuda.is_available():
    # enable autograd compilation
    torch._dynamo.config.compiled_autograd = True
    torch.set_float32_matmul_precision("high")

# register Shimmy if available
try:
    import shimmy

    gymnasium.register_envs(shimmy)
    logging.LOG.info("Register Shimmy environments.")
except ImportError:
    pass
