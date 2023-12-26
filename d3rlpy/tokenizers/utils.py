import numpy as np

from ..types import Float32NDArray, NDArray

__all__ = ["mu_law_encode"]


def mu_law_encode(v: NDArray, mu: float, basis: float) -> Float32NDArray:
    v = np.array(v, dtype=np.float32)
    v = np.sign(v) * np.log(np.abs(v) * mu + 1.0) / np.log(basis * mu + 1.0)
    return v
