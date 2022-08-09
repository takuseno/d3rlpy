import argparse
import d3rlpy
import gym
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    reward_scaler = d3rlpy.preprocessing.ConstantShiftRewardScaler(shift=-1)

    iql = d3rlpy.algos.IQL(actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           batch_size=256,
                           weight_temp=10.0,
                           max_weight=100.0,
                           expectile=0.9,
                           reward_scaler=reward_scaler,
                           use_gpu=args.gpu)

    # workaround for learning scheduler
    iql.create_impl(dataset.get_observation_shape(), dataset.get_action_size())
    scheduler = CosineAnnealingLR(iql.impl._actor_optim, 1000000)

    def callback(algo, epoch, total_step):
        scheduler.step()

    # pretraining
    iql.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=1000000,
            n_steps_per_epoch=100000,
            save_interval=10,
            callback=callback,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"IQL_pretraining_{args.dataset}_{args.seed}")

    # reset learning rate
    for g in iql.impl._actor_optim.param_groups:
        g["lr"] = iql._actor_learning_rate

    # finetuning
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000,
                                                episodes=dataset.episodes)
    eval_env = gym.make(env.unwrapped.spec.id)
    iql.fit_online(
        env,
        buffer=buffer,
        eval_env=eval_env,
        experiment_name=f"IQL_finetuning_{args.dataset}_{args.seed}",
        n_steps=1000000,
        n_steps_per_epoch=1000,
        save_interval=10)


if __name__ == '__main__':
    main()
