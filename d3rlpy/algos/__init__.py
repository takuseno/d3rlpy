from .base import AlgoBase
from .awac import AWAC
from .awr import AWR, DiscreteAWR
from .bc import BC, DiscreteBC
from .bcq import BCQ, DiscreteBCQ
from .bear import BEAR
from .cql import CQL, DiscreteCQL
from .ddpg import DDPG
from .dqn import DQN, DoubleDQN
from .plas import PLAS
from .sac import SAC, DiscreteSAC
from .td3 import TD3


def create_algo(name, discrete, **params):
    """ Returns algorithm from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.
        params (any): arguments for algorithm.

    Returns:
        d3rlpy.algos.base.AlgoBase: algorithm.

    """
    if name == 'awr':
        if discrete:
            return DiscreteAWR(**params)
        else:
            return AWR(**params)
    elif name == 'bc':
        if discrete:
            return DiscreteBC(**params)
        else:
            return BC(**params)
    elif name == 'bcq':
        if discrete:
            return DiscreteBCQ(**params)
        else:
            return BCQ(**params)
    elif name == 'bear':
        if discrete:
            raise ValueError('BEAR does not support discrete action-space.')
        else:
            return BEAR(**params)
    elif name == 'cql':
        if discrete:
            return DiscreteCQL(**params)
        else:
            return CQL(**params)
    elif name == 'ddpg':
        if discrete:
            raise ValueError('DDPG does not support discrete action-space.')
        else:
            return DDPG(**params)
    elif name == 'dqn':
        if discrete:
            return DQN(**params)
        else:
            raise ValueError('DQN does not support continuous action-space.')
    elif name == 'double_dqn':
        if discrete:
            return DoubleDQN(**params)
        else:
            raise ValueError(
                'DoubleDQN does not support continuous action-space.')
    elif name == 'plas':
        if discrete:
            raise ValueError('PLAS does not support discrete action-space.')
        else:
            return PLAS(**params)
    elif name == 'sac':
        if discrete:
            return DiscreteSAC(**params)
        else:
            return SAC(**params)
    elif name == 'td3':
        if discrete:
            raise ValueError('TD3 does not support discrete action-space.')
        else:
            return TD3(**params)
    raise ValueError('invalid algorithm name.')
