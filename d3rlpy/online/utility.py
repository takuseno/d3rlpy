def get_action_size_from_env(env):
    from gym.spaces import Discrete
    if isinstance(env.action_space, Discrete):
        return env.action_space.n
    return env.action_space.shape[0]
