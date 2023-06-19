import argparse
import copy

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=10.0,
    ).create(device=args.gpu)

    # pretraining
    cql.fit(
        dataset,
        n_steps=100000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"CQL_pretraining_{args.dataset}_{args.seed}",
    )

    sac = d3rlpy.algos.SACConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        batch_size=256,
    ).create(device=args.gpu)

    # copy pretrained models to SAC
    sac.build_with_env(env)
    sac.copy_policy_from(cql)  # type: ignore
    sac.copy_q_function_from(cql)  # type: ignore

    # prepare FIFO buffer filled with dataset episodes
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=100000,
        episodes=dataset.episodes,
    )

    # finetuning
    eval_env = copy.deepcopy(env)
    d3rlpy.envs.seed_env(eval_env, args.seed)
    sac.fit_online(
        env,
        buffer=buffer,
        eval_env=eval_env,
        experiment_name=f"SAC_finetuning_{args.dataset}_{args.seed}",
        n_steps=100000,
        n_steps_per_epoch=1000,
        save_interval=10,
    )


if __name__ == "__main__":
    main()
