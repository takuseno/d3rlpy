from tqdm import trange
from ..preprocessing.stack import StackedObservation
from ..metrics.scorer import evaluate_on_environment
from ..logger import D3RLPyLogger
from .utility import get_action_size_from_env


def train(env,
          algo,
          buffer,
          explorer=None,
          n_epochs=1000,
          n_steps_per_epoch=4000,
          n_updates_per_epoch=100,
          update_start_step=0,
          eval_env=None,
          eval_epsilon=0.05,
          eval_interval=1,
          save_metrics=True,
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
        n_epochs (int): the number of epochs to train.
        n_steps_per_epoch (int): the number of steps per epoch.
        n_updates_per_epoch (int): the number of updates per epoch.
        update_start_step (int): the steps before starting updates.
        eval_env (gym.Env): gym-like environment. If None, evaluation is
            skipped.
        eval_epsilon (float): :math:`\\epsilon`-greedy factor during
            evaluation.
        eval_interval (int): interval to perform evaluation.
        save_metrics (bool): flag to record metrics. If False, the log
            directory is not created and the model parameters are not saved.
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
                          save_metrics=save_metrics,
                          root_dir=logdir,
                          verbose=verbose,
                          tensorboard=tensorboard,
                          with_timestamp=with_timestamp)

    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    # prepare stacked observation
    if is_image:
        stacked_frame = StackedObservation(observation_shape, algo.n_frames)
        n_channels = observation_shape[0]
        image_size = observation_shape[1:]
        observation_shape = (n_channels * algo.n_frames, *image_size)

    # setup algorithm
    if algo.impl is None:
        action_size = get_action_size_from_env(env)
        algo.create_impl(observation_shape, action_size)

    # save hyperparameters
    algo._save_params(logger)

    # switch based on show_progress flag
    xrange = trange if show_progress else range

    # setup evaluation scorer
    if eval_env:
        eval_scorer = evaluate_on_environment(eval_env, epsilon=eval_epsilon)
    else:
        eval_scorer = None

    # start training loop
    observation, reward, terminal = env.reset(), 0.0, False
    total_step = 0
    for epoch in range(n_epochs):
        for _ in range(n_steps_per_epoch):
            # stack observation if necessary
            if is_image:
                stacked_frame.append(observation)
                fed_observation = stacked_frame.eval()
            else:
                observation = observation.astype('f4')
                fed_observation = observation

            # sample exploration action
            if explorer:
                action = explorer.sample(algo, fed_observation, total_step)
            else:
                action = algo.sample_action([fed_observation])[0]

            # store observation
            buffer.append(observation, action, reward, terminal)

            # get next observation
            if terminal:
                observation, reward, terminal = env.reset(), 0.0, False
                # for image observation
                if is_image:
                    stacked_frame.clear()
            else:
                observation, reward, terminal, _ = env.step(action)

            total_step += 1

        # update loop
        if total_step > update_start_step:
            for i in xrange(n_updates_per_epoch):
                batch = buffer.sample(algo.batch_size, algo.n_frames)
                update_count = epoch * n_updates_per_epoch + i
                loss = algo.update(epoch, update_count, batch)
                for name, val in zip(algo._get_loss_labels(), loss):
                    if val:
                        logger.add_metric(name, val)

        # evaluation
        if eval_scorer and epoch % eval_interval == 0:
            logger.add_metric('evaluation', eval_scorer(algo))

        # save metrics
        logger.commit(epoch, total_step)

        if epoch % save_interval == 0:
            logger.save_model(epoch, algo)
