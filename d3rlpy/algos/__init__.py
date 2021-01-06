from typing import Any, Dict, Type
from .base import AlgoBase, DataGenerator
from .awac import AWAC
from .awr import AWR, DiscreteAWR
from .bc import BC, DiscreteBC
from .bcq import BCQ, DiscreteBCQ
from .bear import BEAR
from .cql import CQL, DiscreteCQL
from .ddpg import DDPG
from .dqn import DQN, DoubleDQN
from .plas import PLAS, PLASWithPerturbation
from .sac import SAC, DiscreteSAC
from .td3 import TD3


__all__ = [
    "AlgoBase",
    "AWAC",
    "AWR",
    "DiscreteAWR",
    "BC",
    "DiscreteBC",
    "BCQ",
    "DiscreteBCQ",
    "BEAR",
    "CQL",
    "DiscreteCQL",
    "DDPG",
    "DQN",
    "DoubleDQN",
    "PLAS",
    "PLASWithPerturbation",
    "SAC",
    "DiscreteSAC",
    "TD3",
    "DataGenerator",
    "get_algo",
    "create_algo",
]


DISCRETE_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "awr": DiscreteAWR,
    "bc": DiscreteBC,
    "bcq": DiscreteBCQ,
    "cql": DiscreteCQL,
    "dqn": DQN,
    "double_dqn": DoubleDQN,
    "sac": DiscreteSAC,
}

CONTINUOUS_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "awac": AWAC,
    "awr": AWR,
    "bc": BC,
    "bcq": BCQ,
    "bear": BEAR,
    "cql": CQL,
    "ddpg": DDPG,
    "sac": SAC,
    "plas": PLASWithPerturbation,
    "td3": TD3,
}


def get_algo(name: str, discrete: bool) -> Type[AlgoBase]:
    """Returns algorithm class from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.

    Returns:
        type: algorithm class.

    """
    if discrete:
        if name in DISCRETE_ALGORITHMS:
            return DISCRETE_ALGORITHMS[name]
        raise ValueError("%s does not support discrete action-space." % name)
    if name in CONTINUOUS_ALGORITHMS:
        return CONTINUOUS_ALGORITHMS[name]
    raise ValueError("%s does not support continuous action-space." % name)


def create_algo(name: str, discrete: bool, **params: Any) -> AlgoBase:
    """Returns algorithm object from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.
        params (any): arguments for algorithm.

    Returns:
        d3rlpy.algos.base.AlgoBase: algorithm.

    """
    return get_algo(name, discrete)(**params)
