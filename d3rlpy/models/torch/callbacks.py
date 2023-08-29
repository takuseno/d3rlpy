from abc import ABCMeta, abstractmethod
from typing import Sequence

from ...algos import QLearningAlgoBase, QLearningAlgoImplBase
from ...constants import IMPL_NOT_INITIALIZED_ERROR


class QLearningCallback(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, algo: QLearningAlgoBase, epoch: int, total_step: int):
        pass


class ParameterReset(QLearningCallback):
    def __init__(self, replay_ratio: int, layer_reset:Sequence[bool], 
                 algo:QLearningAlgoBase=None) -> None:
        self._replay_ratio = replay_ratio
        self._layer_reset = layer_reset
        self._check = False
        if algo is not None:
            self._check_layer_resets(algo=algo)
            
    
    def _check_layer_resets(self, algo:QLearningAlgoBase):
        assert algo._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo._impl, QLearningAlgoImplBase)
        valid_layers = [
            hasattr(layer, 'reset_parameters') for lr, layer in zip(
                self._layer_reset, algo._impl.q_function) 
            if lr
            ]
        self._check = all(valid_layers)
        if not self._check:
            raise ValueError(
                "Some layer do not contain resettable parameters"
                )
    
    def __call__(self, algo: QLearningAlgoBase, epoch: int, total_step: int):
        if not self._check:
            self._check_layer_resets(algo=algo)
        assert isinstance(algo._impl, QLearningAlgoImplBase)
        if epoch % self._replay_ratio == 0:
            for lr, layer in enumerate(
                zip(self._layer_reset, algo._impl.q_function)
                ):
                if lr:
                    layer.reset_parameters()