import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl.gym_mujoco
import gym
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy
from d3rlpy.algos import CQL, IQL, QLearningAlgoBase
from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer
from d3rlpy.types import NDArray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v2")
    parser.add_argument("--context_size", type=int, default=20)
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument(
        "--q_learning_type", type=str, default="cql"
    )  # Q-learning algorithm ("cql" or "iql")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    # get timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # first fit Q-learning algorithm to the dataset
    if args.model_file is not None:
        q_algo = d3rlpy.load_learnable(args.model_file)
    else:
        if args.q_learning_type == "cql":
            q_algo = fit_cql(dataset, env, args.seed, args.gpu, timestamp)
        elif args.q_learning_type == "iql":
            q_algo = fit_iql(dataset, env, args.seed, args.gpu, timestamp)

    # relabel dataset RTGs with the learned value functions
    print("Relabeling dataset with RTGs...")
    relabel_dataset_rtg(
        dataset._buffer, q_algo, args.context_size, seed=args.seed
    )

    # fit decision transformer to the relabeled dataset
    fit_dt(dataset, env, args.context_size, args.seed, args.gpu, False, timestamp)


""" --------------------------------------------------------------------
    Aargument dataset
-------------------------------------------------------------------- """

def relabel_dataset_rtg(
    buffer: InfiniteBuffer, q_algo: Union["CQL", "IQL"], k: int, seed: int = 0
):
    """
    Relabel RTG (reward-to-go) to the given dataset using the given Q-function.

    Args:
        buffer (InfiniteBuffer): Buffer holding trajectory dataset.
        q_algo: Trained Q-learning algoirthm.
        k (int): Context length for DT.
        seed (int): The random seed.
        gpu (int, optional): The GPU device ID. Defaults to None.
        timestamp (str, optional): The timestamp for experiment name. Defaults to None.
    """
    # fix seed
    d3rlpy.seed(seed)

    prev_idx = -1
    for n in range(buffer.transition_count):
        episode, idx = buffer._transitions[-n - 1]  # get transitions backwards
        if idx > prev_idx:
            # get values for all observations in the episode
            values = []
            for _ in range(10):
                sampled_actions = q_algo.sample_action(episode.observations)
                values.append(
                    q_algo.predict_value(episode.observations, sampled_actions)
                )
            value = np.array(values).mean(axis=0)
            rewards = np.squeeze(episode.rewards, axis=1)
            rtg = 0
        else:
            start = max(0, idx - k + 1)
            rtg = rewards[idx] + np.maximum(rtg, value[idx + 1])  # relabel rtg
            relabelled_rewards = np.zeros_like(rewards)
            relabelled_rewards[idx] = rtg
            relabelled_rewards[start:idx] = rewards[start:idx]
            relabelled_episode = d3rlpy.dataset.components.Episode(
                observations=episode.observations,
                actions=episode.actions,
                rewards=np.expand_dims(relabelled_rewards, axis=1),
                terminated=episode.terminated,
            )
            buffer._transitions[-n - 1] = (relabelled_episode, idx)

        prev_idx = idx


""" -------------------------------------------------------------------- 
    Fit offline RL algorithms to the given dataset. 
-------------------------------------------------------------------- """


def fit_cql(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    seed: int = 1,
    gpu: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> "CQL":
    """
    Fit the CQL algorithm to the given dataset and environment.

    Args:
        dataset (MDPdataset): Dataset for the training.
        env (gym.Env): The environment instance.
        seed (int): The random seed.
        gpu (int, optional): The GPU device ID. Defaults to None.
        timestamp (str, optional): The timestamp for experiment name. Defaults to None.
    """
    # fix seed
    d3rlpy.seed(seed)
    d3rlpy.envs.seed_env(env, seed)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    if "medium-v0" in env.spec.id:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=conservative_weight,
    ).create(device=gpu)

    cql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=50,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=(
            f"CQL_{env.spec.id}_{seed}"
            if timestamp is None
            else f"CQL_{env.spec.id}_{seed}_{timestamp}"
        ),
        with_timestamp=False,
    )

    return cql


def fit_iql(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    seed: int = 1,
    gpu: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> "IQL":
    """
    Fit the IQL algorithm to the given dataset and environment.

    Args:
        dataset (MDPdataset): Dataset for the training.
        env (gym.Env): The environment instance.
        seed (int): The random seed.
        gpu (int, optional): The GPU device ID. Defaults to None.
        timestamp (str, optional): The timestamp for experiment name. Defaults to None.
    """
    # fix seed
    d3rlpy.seed(seed)
    d3rlpy.envs.seed_env(env, seed)

    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0
    )

    iql = d3rlpy.algos.IQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        weight_temp=3.0,
        max_weight=100.0,
        expectile=0.7,
        reward_scaler=reward_scaler,
    ).create(device=gpu)

    # workaround for learning scheduler
    iql.build_with_dataset(dataset)
    assert iql.impl
    scheduler = CosineAnnealingLR(
        iql.impl._modules.actor_optim,  # pylint: disable=protected-access
        500000,
    )

    def callback(algo: d3rlpy.algos.IQL, epoch: int, total_step: int) -> None:
        scheduler.step()

    iql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        callback=callback,
        evaluators={
            "environment": d3rlpy.metrics.EnvironmentEvaluator(env, n_trials=10)
        },
        experiment_name=(
            f"IQL_{env.spec.id}_{seed}"
            if timestamp is None
            else f"IQL_{env.spec.id}_{seed}_{timestamp}"
        ),
        with_timestamp=False,
    )

    return iql


def fit_dt(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    context_size: int = 20,
    seed: int = 1,
    gpu: Optional[int] = None,
    compile: bool = False,
    timestamp: Optional[str] = None,
) -> None:
    """
    Fit decisiton transformer to the given dataset and environment.

    Args:
        dataset (MDPdataset): Dataset for the training.
        env (gym.Env): The environment instance.
        seed (int): The random seed.
        gpu (int, optional): The GPU device ID. Defaults to None.
        timestamp (str, optional): The timestamp for experiment name. Defaults to None.
    """
    # fix seed
    d3rlpy.seed(seed)
    d3rlpy.envs.seed_env(env, seed)

    if "halfcheetah" in env.spec.id:
        target_return = 6000
    elif "hopper" in env.spec.id:
        target_return = 3600
    elif "walker" in env.spec.id:
        target_return = 5000
    else:
        raise ValueError("unsupported dataset")

    dt = d3rlpy.algos.DecisionTransformerConfig(
        batch_size=64,
        learning_rate=1e-4,
        optim_factory=d3rlpy.optimizers.AdamWFactory(
            weight_decay=1e-4,
            clip_grad_norm=0.25,
            lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
                warmup_steps=10000
            ),
        ),
        encoder_factory=d3rlpy.models.VectorEncoderFactory(
            [128],
            exclude_last_activation=True,
        ),
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.MultiplyRewardScaler(0.001),
        position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
        context_size=context_size,
        num_heads=1,
        num_layers=3,
        max_timestep=1000,
        compile_graph=compile,
    ).create(device=gpu)

    dt.fit(
        dataset,
        n_steps=100000,
        n_steps_per_epoch=1000,
        save_interval=10,
        eval_env=env,
        eval_target_return=target_return,
        experiment_name=(
            f"QDT_{env.spec.id}_{seed}"
            if timestamp is None
            else f"QDT_{env.spec.id}_{seed}_{timestamp}"
        ),
        with_timestamp=False,
    )


if __name__ == "__main__":
    main()
