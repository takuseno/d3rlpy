import numpy as np

from d3rlpy.tokenizers import FloatTokenizer
from d3rlpy.types import NDArray


def test_float_tokenizer() -> None:
    tokenizer = FloatTokenizer(num_bins=100, use_mu_law_encode=False)
    v: NDArray = (2.0 * np.arange(100) / 100 - 1).astype(np.float32)
    tokenized_v = tokenizer(v)
    assert np.all(tokenized_v == np.arange(100))

    # check mu_law_encode
    tokenizer = FloatTokenizer(num_bins=100)
    v = np.arange(100) - 50
    tokenized_v = tokenizer(v)
    assert np.all(tokenized_v >= 0)
    assert np.all(tokenized_v < 100)

    # check token_offset
    tokenizer = FloatTokenizer(
        num_bins=100, use_mu_law_encode=False, token_offset=1
    )
    v = np.array([-1, 1])
    tokenized_v = tokenizer(v)
    assert tokenized_v[0] == 1
    assert tokenized_v[1] == 100

    # check decode
    tokenizer = FloatTokenizer(num_bins=1000000)
    v = np.arange(100) - 50
    decoded_v = tokenizer.decode(tokenizer(v))
    assert np.allclose(decoded_v, v, atol=1e-3)

    # check decode with multi-dimension
    v = np.reshape(v, [5, -1])
    decoded_v = tokenizer.decode(tokenizer(v))
    assert v.shape == decoded_v.shape
    assert np.allclose(decoded_v, v, atol=1e-3)
