import argparse

import gymnasium
from gymnasium.wrappers.flatten_observation import FlattenObservation

import d3rlpy

# You need to install DeepMind Control and Shimmy beforehands as follows:
# $ d3rlpy install dm_control

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper-stand")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    env_id = f"dm_control/{args.env}"
    env = FlattenObservation(gymnasium.make(env_id))
    eval_env = FlattenObservation(gymnasium.make(env_id))

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)
    d3rlpy.envs.seed_env(eval_env, args.seed)

    # setup algorithm
    sac = d3rlpy.algos.SACConfig(
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
    ).create(device=args.gpu)

    # replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=env)

    # start training
    sac.fit_online(
        env,
        buffer,
        eval_env=eval_env,
        n_steps=1000000,
        n_steps_per_epoch=10000,
        update_interval=1,
        update_start_step=1000,
    )


if __name__ == "__main__":
    main()
