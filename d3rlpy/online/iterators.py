from tqdm import trange
from ..metrics.scorer import evaluate_on_environment
from ..logger import D3RLPyLogger
from .utility import get_action_size_from_env


def train(env,
          algo,
          buffer,
          explorer=None,
          n_steps_per_epoch=4000,
          n_updates_per_epoch=100,
          eval_env=None,
          eval_epsilon=0.05,
          experiment_name=None,
          with_timestamp=True,
          logdir='d3rlpy_logs',
          verbose=True,
          show_progress=True,
          tensorboard=True,
          save_interval=1):
    """ Start training loop of online deep reinforcement learning.

    Args:
        env (gym.Env): gym-like environment.
        algo (d3rlpy.algos.base.AlgoBase): algorithm.
        buffer (d3rlpy.online.buffers.Buffer): replay buffer.
        explorer (d3rlpy.online.explorers.Explorer): action explorer.
        n_steps_per_epoch (int): the number of steps per epoch.
        n_updates_per_epoch (int): the number of updates per epoch.
        eval_env (gym.Env): gym-like environment. If None, evaluation is
            skipped.
        eval_epsilon (float): :math:`\\epsilon`-greedy factor during
            evaluation.
        experiment_name (str): experiment name for logging. If not passed,
            the directory name will be `{class name}_online_{timestamp}`.
        with_timestamp (bool): flag to add timestamp string to the last of
            directory name.
        logdir (str): root directory name to save logs.
        verbose (bool): flag to show logged information on stdout.
        show_progress (bool): flag to show progress bar for iterations.
        tensorboard (bool): flag to save logged information in tensorboard
            (additional to the csv data)
        save_interval (int): interval to save parameters.

    """

    # setup logger
    if experiment_name is None:
        experiment_name = algo.__class__.__name__ + '_online'
    logger = D3RLPyLogger(experiment_name,
                          root_dir=logdir,
                          verbose=verbose,
                          tensorboard=tensorboard,
                          with_timestamp=with_timestamp)

    # setup algorithm
    if algo.impl is None:
        action_size = get_action_size_from_env(env)
        algo.create_impl(env.observation_space.shape, action_size)

    # save hyperparameters
    algo._save_params(logger)

    # switch based on show_progress flag
    xrange = trange if show_progress else range

    # setup evaluation scorer
    if eval_env:
        scorer = evaluate_on_environment(eval_env, epsilon=eval_epsilon)
    else:
        scorer = None

    # start training loop
    observation, reward, terminal = env.reset(), 0.0, False
    total_step = 0
    for epoch in range(algo.n_epochs):
        for step in range(n_steps_per_epoch):
            # sample exploration action
            if explorer:
                action = explorer.sample(algo, observation, total_step)
            else:
                action = algo.sample_action([observation])[0]

            # store observation
            buffer.append(observation, action, reward, terminal)

            # get next observation
            if terminal:
                observation, reward, terminal = env.reset(), 0.0, False
            else:
                observation, reward, terminal, _ = env.step(action)

            total_step += 1

        # update loop
        for i in xrange(n_updates_per_epoch):
            batch = buffer.sample(algo.batch_size)
            loss = algo.update(epoch, epoch * n_updates_per_epoch + i, batch)
            for name, val in zip(algo._get_loss_labels(), loss):
                if val:
                    logger.add_metric(name, val)

        # evaluation
        if scorer:
            logger.add_metric('evaluation', scorer(algo))

        # save metrics
        logger.commit(epoch, total_step)

        if epoch % save_interval == 0:
            logger.save_model(epoch, algo)
