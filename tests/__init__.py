import pytest
import os

from d3rlpy.encoders import PixelEncoderFactory, VectorEncoderFactory

is_skipping_performance_test = os.environ.get('TEST_PERFORMANCE') != "TRUE"
performance_test = pytest.mark.skipif(is_skipping_performance_test,
                                      reason='skip performance tests')


def create_encoder_factory(use_encoder_factory, observation_shape):
    if use_encoder_factory:
        if len(observation_shape) == 3:
            encoder_factory = PixelEncoderFactory()
        else:
            encoder_factory = VectorEncoderFactory()
    else:
        encoder_factory = None
    return encoder_factory
