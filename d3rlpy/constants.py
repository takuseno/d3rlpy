from enum import Enum

from ._version import __version__

__all__ = [
    "IMPL_NOT_INITIALIZED_ERROR",
    "ALGO_NOT_GIVEN_ERROR",
    "DISCRETE_ACTION_SPACE_MISMATCH_ERROR",
    "CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR",
    "ActionSpace",
    "PositionEncodingType",
]

IMPL_NOT_INITIALIZED_ERROR = (
    "The neural network parameters are not "
    "initialized. Pleaes call build_with_dataset, "
    "build_with_env, or directly call fit or "
    "fit_online method."
)

ALGO_NOT_GIVEN_ERROR = (
    "The algorithm to evaluate is not given. Please give the trained algorithm"
    " to the argument."
)

DISCRETE_ACTION_SPACE_MISMATCH_ERROR = (
    "The action-space of the given dataset is not compatible with the"
    " algorithm. Please use discrete action-space algorithms. The algorithms"
    " list is available below.\n"
    f"https://d3rlpy.readthedocs.io/en/v{__version__}/references/algos.html"
)

CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR = (
    "The action-space of the given dataset is not compatible with the"
    " algorithm. Please use continuous action-space algorithms. The algorithm"
    " list is available below.\n"
    f"https://d3rlpy.readthedocs.io/en/v{__version__}/references/algos.html"
)


class ActionSpace(Enum):
    CONTINUOUS = 1
    DISCRETE = 2
    BOTH = 3


class PositionEncodingType(Enum):
    SIMPLE = "simple"
    GLOBAL = "global"


class LoggingStrategy(Enum):
    STEPS = "steps"
    EPOCH = "epoch"
