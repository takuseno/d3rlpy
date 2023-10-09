import random

import numpy as np
import torch

from . import (
    algos,
    dataset,
    datasets,
    envs,
    logging,
    metrics,
    models,
    notebook_utils,
    ope,
    preprocessing,
)
from ._version import __version__
from .base import load_learnable
from .constants import ActionSpace, PositionEncodingType
from .healthcheck import run_healthcheck

__all__ = [
    "algos",
    "dataset",
    "datasets",
    "envs",
    "logging",
    "metrics",
    "models",
    "notebook_utils",
    "ope",
    "preprocessing",
    "__version__",
    "load_learnable",
    "ActionSpace",
    "PositionEncodingType",
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
