import numpy as np

from d3rlpy.tokenizers import mu_law_encode


def test_mu_law_encode() -> None:
    v = np.arange(100) - 50
    encoded_v = mu_law_encode(v, mu=100, basis=256)
    assert np.all(encoded_v < 1)
    assert np.all(-1 < encoded_v)
