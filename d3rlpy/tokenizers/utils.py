import numpy as np

from ..types import Float32NDArray, NDArray

__all__ = ["mu_law_encode", "mu_law_decode"]


def mu_law_encode(x: NDArray, mu: float, basis: float) -> Float32NDArray:
    x = np.array(x, dtype=np.float32)
    y = np.sign(x) * np.log(np.abs(x) * mu + 1.0) / np.log(basis * mu + 1.0)
    return y  # type: ignore


def mu_law_decode(y: Float32NDArray, mu: float, basis: float) -> Float32NDArray:
    x = np.sign(y) * (np.exp(np.log(basis * mu + 1.0) * np.abs(y)) - 1.0) / mu
    return x  # type: ignore
