import numpy as np
from typing_extensions import Protocol, runtime_checkable

from ..types import Float32NDArray, Int32NDArray, NDArray
from .utils import mu_law_decode, mu_law_encode

__all__ = [
    "Tokenizer",
    "FloatTokenizer",
]


@runtime_checkable
class Tokenizer(Protocol):
    def __call__(self, x: NDArray) -> Int32NDArray:
        ...

    def decode(self, y: Int32NDArray) -> NDArray:
        ...


class FloatTokenizer(Tokenizer):
    _bins: Float32NDArray
    _use_mu_law_encode: bool
    _mu: float
    _basis: float
    _token_offset: int

    def __init__(
        self,
        num_bins: int,
        minimum: float = -1.0,
        maximum: float = 1.0,
        use_mu_law_encode: bool = True,
        mu: float = 100.0,
        basis: float = 256.0,
        token_offset: int = 0,
    ):
        self._bins = np.array(
            (maximum - minimum) * np.arange(num_bins) / num_bins + minimum,
            dtype=np.float32,
        )
        self._use_mu_law_encode = use_mu_law_encode
        self._mu = mu
        self._basis = basis
        self._token_offset = token_offset

    def __call__(self, x: NDArray) -> Int32NDArray:
        if self._use_mu_law_encode:
            x = mu_law_encode(x, self._mu, self._basis)
        return np.digitize(x, self._bins) - 1 + self._token_offset

    def decode(self, y: Int32NDArray) -> NDArray:
        x = self._bins[y - self._token_offset]
        if self._use_mu_law_encode:
            x = mu_law_decode(x, mu=self._mu, basis=self._basis)
        return x  # type: ignore
