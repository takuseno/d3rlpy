from gym.spaces import Discrete
from d3rlpy.dataset import MDPDataset


class SB3Wrapper:
    """ A wrapper for d3rlpy algorithms so they can be used with Stable-Baseline3 (SB3).

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm.

    Attributes:
        algo (d3rlpy.algos.base.AlgoBase): algorithm.

    """
    def __init__(self, algo):
        # Avoid infinite recursion due to override of setattr
        self.__dict__["algo"] = algo

    def predict(self, observation, state=None, mask=None, deterministic=True):
        """ Returns actions.

        Args:
            observation (np.ndarray): observation.
            state (np.ndarray): this argument is just ignored.
            mask (np.ndarray): this argument is just ignored.
            deterministic (bool): flag to return greedy actions.

        Returns:
            tuple: ``(actions, None)``.

        """
        if deterministic:
            return self.algo.predict(observation), None
        return self.algo.sample_action(observation), None

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.algo, attr)

    def __setattr__(self, attr_name, value):
        if attr_name != "algo":
            self.algo.__setattr__(attr_name, value)
        else:
            self.__dict__["algo"] = value


def to_mdp_dataset(replay_buffer):
    """ Returns d3rlpy's MDPDataset from SB3's ReplayBuffer

    Args:
        replay_buffer (stable_baselines3.common.buffers.ReplayBuffer):
            SB3's replay buffer.

    Returns:
        d3rlpy.dataset.MDPDataset: d3rlpy's MDPDataset.

    """
    pos = replay_buffer.size()
    discrete_action = isinstance(replay_buffer.action_space, Discrete)
    dataset = MDPDataset(observations=replay_buffer.observations[:pos, 0],
                         actions=replay_buffer.actions[:pos, 0],
                         rewards=replay_buffer.rewards[:pos, 0],
                         terminals=replay_buffer.dones[:pos, 0],
                         discrete_action=discrete_action)
    return dataset
