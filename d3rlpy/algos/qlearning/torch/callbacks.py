from abc import ABCMeta, abstractmethod
from typing import Sequence, List
import torch.nn as nn

from ... import QLearningAlgoBase, QLearningAlgoImplBase
from ....constants import IMPL_NOT_INITIALIZED_ERROR

__all__ = [
    "ParameterReset"
]

class QLearningCallback(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, algo: QLearningAlgoBase, epoch: int, total_step: int):
        pass


class ParameterReset(QLearningCallback):
    def __init__(self, replay_ratio: int, encoder_reset:Sequence[bool],
                 output_reset:bool, algo:QLearningAlgoBase=None) -> None:
        self._replay_ratio = replay_ratio
        self._encoder_reset = encoder_reset
        self._output_reset = output_reset
        self._check = False
        if algo is not None:
            self._check_layer_resets(algo=algo)
            
    
    def _get_layers(self, q_func:nn.ModuleList)->List[nn.Module]:
        all_modules = {nm:module for (nm, module) in q_func.named_modules()}
        q_func_layers = [
            *all_modules["_encoder._layers"],
            all_modules["_fc"]
            ]
        return q_func_layers
    
    def _check_layer_resets(self, algo:QLearningAlgoBase):
        assert algo._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo._impl, QLearningAlgoImplBase)
        
        all_valid_layers = []
        for q_func in algo._impl.q_function:
            q_func_layers = self._get_layers(q_func)
            if len(self._encoder_reset) + 1 != len(q_func_layers):
                raise ValueError(
                    f"""
                    q_function layers: {q_func_layers};
                    specified encoder layers: {self._encoder_reset} 
                    """
                    )
            valid_layers = [
                hasattr(layer, 'reset_parameters') for lr, layer in zip(
                    self._encoder_reset, q_func_layers) 
                if lr
                ]
            all_valid_layers.append(all(valid_layers))
        self._check = all(all_valid_layers)
        if not self._check:
            raise ValueError(
                "Some layer do not contain resettable parameters"
                )
    
    def __call__(self, algo: QLearningAlgoBase, epoch: int, total_step: int):
        if not self._check:
            self._check_layer_resets(algo=algo)
        assert isinstance(algo._impl, QLearningAlgoImplBase)
        if epoch % self._replay_ratio == 0:
            reset_lst = [*self._encoder_reset, self._output_reset]
            for q_func in algo._impl.q_function:
                q_func_layers = self._get_layers(q_func)
                for lr, layer in zip(reset_lst, q_func_layers):
                    if lr:
                        layer.reset_parameters()