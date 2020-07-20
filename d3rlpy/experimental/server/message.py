import numpy as np
import msgpack


def pack_experience(observations, actions, rewards, terminals):
    package = {
        'observations': observations.astype('f4').tobytes('C'),
        'actions': actions.astype('f4').tobytes('C'),
        'rewards': rewards.astype('f4').tobytes('C'),
        'terminals': terminals.astype('f4').tobytes('C'),
        'observation_shape': observations.shape[1:],
        'data_size': observations.shape[0]
    }
    return msgpack.packb(package, use_bin_type=True)


def unpack_experience(binary_data):
    package = msgpack.unpackb(binary_data, raw=False)
    observations = np.frombuffer(package['observations'], dtype=np.float32)
    actions = np.frombuffer(package['actions'], dtype=np.float32)
    rewards = np.frombuffer(package['rewards'], dtype=np.float32)
    terminals = np.frombuffer(package['terminals'], dtype=np.float32)

    # reshape
    observation_shape = package['observation_shape']
    data_size = package['data_size']
    observations = np.reshape(observations, [data_size] + observation_shape)
    actions = np.reshape(actions, (data_size, -1))

    return observations, actions, rewards, terminals
