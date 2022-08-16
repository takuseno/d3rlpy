import numpy as np
import pytest
import stable_baselines3 as sb3

from d3rlpy.algos import SAC
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import DatasetInfo, ReplayBuffer
from d3rlpy.wrappers.sb3 import SB3Wrapper, to_mdp_dataset


@pytest.mark.parametrize("observation_shape", [(10,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [22])
def test_sb3_wrapper(observation_shape, action_size, batch_size):
    algo = SAC()
    algo.create_impl(observation_shape, action_size)

    sb3 = SB3Wrapper(algo)

    observations = np.random.random((batch_size,) + observation_shape)

    # check greedy action
    actions, state = sb3.predict(observations, deterministic=True)
    assert actions.shape == (batch_size, action_size)
    assert state is None

    # check sampling
    stochastic_actions, state = sb3.predict(observations, deterministic=False)
    assert stochastic_actions.shape == (batch_size, action_size)
    assert state is None
    assert not np.allclose(actions, stochastic_actions)


@pytest.mark.parametrize(
    "algo_env", [(sb3.SAC, "Pendulum-v0"), (sb3.DQN, "CartPole-v0")]
)
def test_to_mdp_dataset(algo_env):
    algo, env_name = algo_env
    model = algo(
        "MlpPolicy",
        env_name,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=2500,
    )
    model.learn(2500)

    env = model.get_env()
    dataset = to_mdp_dataset(model.replay_buffer)

    transition = dataset.sample_transition()
    dataset_info = DatasetInfo.from_episodes(dataset.episodes)

    assert isinstance(dataset, ReplayBuffer)
    assert transition.observation_shape == env.observation_space.shape
    if env_name == "CartPole-v0":
        assert dataset_info.action_size == env.action_space.n
        assert dataset_info.action_space == ActionSpace.DISCRETE
    elif env_name == "Pendulum-v0":
        assert dataset_info.action_size == env.action_space.shape[0]
        assert dataset_info.action_space == ActionSpace.CONTINUOUS
    assert len(dataset.episodes) == model.replay_buffer.dones.sum()
