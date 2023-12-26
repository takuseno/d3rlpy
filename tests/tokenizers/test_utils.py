import numpy as np

from d3rlpy.tokenizers import mu_law_decode, mu_law_encode
from d3rlpy.types import NDArray


def test_mu_law_encode() -> None:
    v = np.arange(100) - 50
    encoded_v = mu_law_encode(v, mu=100, basis=256)
    assert np.all(encoded_v < 1)
    assert np.all(-1 < encoded_v)


def test_mu_law_decode() -> None:
    v: NDArray = np.array(np.arange(100) - 50, dtype=np.float32)
    decoded_v = mu_law_decode(
        mu_law_encode(v, mu=100, basis=256), mu=100, basis=256
    )
    assert np.allclose(decoded_v, v)
