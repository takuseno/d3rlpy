import numpy as np

from d3rlpy.experimental.server.message import pack_experience
from d3rlpy.experimental.server.message import unpack_experience


def test_pack_experience():
    observations = np.random.random((100, 32))
    actions = np.random.random((100, 2))
    rewards = np.random.random(100)
    terminals = np.random.randint(2, size=100)
    packed_data = pack_experience(observations, actions, rewards, terminals)

    unpacked_data = unpack_experience(packed_data)
    assert np.allclose(unpacked_data[0], observations)
    assert np.allclose(unpacked_data[1], actions)
    assert np.allclose(unpacked_data[2], rewards)
    assert np.allclose(unpacked_data[3], terminals)
