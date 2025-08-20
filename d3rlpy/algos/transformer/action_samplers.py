from typing import Protocol, Union

import numpy as np

from ...types import NDArray

__all__ = [
    "TransformerActionSampler",
    "IdentityTransformerActionSampler",
    "SoftmaxTransformerActionSampler",
    "GreedyTransformerActionSampler",
]


class TransformerActionSampler(Protocol):
    r"""Interface of TransformerActionSampler."""

    def __call__(self, transformer_output: NDArray) -> Union[NDArray, int]:
        r"""Returns sampled action from Transformer output.

        Args:
            transformer_output: Output of Transformer algorithms.

        Returns:
            Sampled action.
        """
        raise NotImplementedError


class IdentityTransformerActionSampler(TransformerActionSampler):
    r"""Identity action-sampler.

    This class implements identity function to process Transformer output.
    Sampled action is the exactly same as ``transformer_output``.
    """

    def __call__(self, transformer_output: NDArray) -> Union[NDArray, int]:
        return transformer_output


class SoftmaxTransformerActionSampler(TransformerActionSampler):
    r"""Softmax action-sampler.

    This class implements softmax function to sample action from discrete
    probability distribution.

    Args:
        temperature (int): Softmax temperature.
    """

    _temperature: float

    def __init__(self, temperature: float = 1.0):
        self._temperature = temperature

    def __call__(self, transformer_output: NDArray) -> Union[NDArray, int]:
        assert transformer_output.ndim == 1
        logits = transformer_output / self._temperature
        x = np.exp(logits - np.max(logits))
        probs = x / np.sum(x)
        action = np.random.choice(probs.shape[0], p=probs)
        return int(action)


class GreedyTransformerActionSampler(TransformerActionSampler):
    r"""Greedy action-sampler.

    This class implements greedy function to determine action from discrte
    probability distribution.
    """

    def __call__(self, transformer_output: NDArray) -> Union[NDArray, int]:
        assert transformer_output.ndim == 1
        return int(np.argmax(transformer_output))
