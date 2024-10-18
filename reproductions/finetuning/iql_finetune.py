# pylint: disable=protected-access
import argparse
import copy

from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="antmaze-umaze-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_minari(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # for antmaze datasets
    reward_scaler = d3rlpy.preprocessing.ConstantShiftRewardScaler(shift=-1)

    iql = d3rlpy.algos.IQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        weight_temp=10.0,  # hyperparameter for antmaze
        max_weight=100.0,
        expectile=0.9,  # hyperparameter for antmaze
        reward_scaler=reward_scaler,
    ).create(device=args.gpu)

    # workaround for learning scheduler
    iql.build_with_dataset(dataset)
    assert iql.impl
    scheduler = CosineAnnealingLR(
        iql.impl._modules.actor_optim.optim,  # pylint: disable=protected-access
        1000000,
    )

    def callback(algo: d3rlpy.algos.IQL, epoch: int, total_step: int) -> None:
        scheduler.step()

    # pretraining
    iql.fit(
        dataset,
        n_steps=1000000,
        n_steps_per_epoch=100000,
        save_interval=10,
        callback=callback,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"IQL_pretraining_{args.dataset}_{args.seed}",
    )

    # reset learning rate
    for g in iql.impl._modules.actor_optim.optim.param_groups:
        g["lr"] = iql.config.actor_learning_rate

    # prepare FIFO buffer filled with dataset episodes
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(1000000)
    for episode in dataset.episodes:
        buffer.append_episode(episode)

    # finetuning
    eval_env = copy.deepcopy(env)
    d3rlpy.envs.seed_env(eval_env, args.seed)
    iql.fit_online(
        env,
        buffer=buffer,
        eval_env=eval_env,
        experiment_name=f"IQL_finetuning_{args.dataset}_{args.seed}",
        n_steps=1000000,
        n_steps_per_epoch=1000,
        save_interval=10,
    )


if __name__ == "__main__":
    main()
