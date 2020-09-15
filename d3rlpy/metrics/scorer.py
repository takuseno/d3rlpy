import numpy as np

from ..preprocessing.stack import StackedObservation
from ..dataset import TransitionMiniBatch


def _make_batches(episode, window_size, n_frames):
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, len(episode))
        transitions = episode.transitions[head_index:last_index]
        batch = TransitionMiniBatch(transitions, n_frames)
        yield batch


def td_error_scorer(algo, episodes, window_size=1024):
    """ Returns average TD error (in negative scale).

    This metics suggests how Q functions overfit to training sets.
    If the TD error is large, the Q functions are overfitting.

    .. math::

        \\mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \\sim D}
            [Q_\\theta (s_t, a_t)
             - (r_t + \\gamma \\max_a Q_\\theta (s_{t+1}, a))^2]

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: negative average TD error.

    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, algo.n_frames):
            # estimate values for current observations
            values = algo.predict_value(batch.observations, batch.actions)

            # estimate values for next observations
            next_actions = algo.predict(batch.next_observations)
            next_values = algo.predict_value(batch.next_observations,
                                             next_actions)

            # calculate td errors
            mask = (1.0 - np.asarray(batch.terminals)).reshape(-1)
            rewards = np.asarray(batch.next_rewards).reshape(-1)
            y = rewards + algo.gamma * next_values * mask
            total_errors += ((values - y)**2).tolist()

    # smaller is better
    return -np.mean(total_errors)


def discounted_sum_of_advantage_scorer(algo, episodes, window_size=1024):
    """ Returns average of discounted sum of advantage (in negative scale).

    This metrics suggests how the greedy-policy selects different actions in
    action-value space.
    If the sum of advantage is small, the policy selects actions with larger
    estimated action-values.

    .. math::

        \\mathbb{E}_{s_t, a_t \\sim D}
            [\\sum_{t' = t} \\gamma^{t' - t} A(s_{t'}, a_{t'})]

    where :math:`A(s_t, a_t) = Q_\\theta (s_t, a_t)
    - \\max_a Q_\\theta (s_t, a)`.

    References:
        * `Murphy., A generalization error for Q-Learning.
          <http://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: negative average of discounted sum of advantage.

    """
    total_sums = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, algo.n_frames):
            # estimate values for dataset actions
            dataset_values = algo.predict_value(batch.observations,
                                                batch.actions)

            # estimate values for the current policy
            actions = algo.predict(batch.observations)
            on_policy_values = algo.predict_value(batch.observations, actions)

            # calculate advantages
            advantages = (dataset_values - on_policy_values).tolist()

            # calculate discounted sum of advantages
            A = advantages[-1]
            sum_advantages = [A]
            for advantage in reversed(advantages[:-1]):
                A = advantage + algo.gamma * A
                sum_advantages.append(A)

            total_sums += sum_advantages

    # smaller is better
    return -np.mean(total_sums)


def average_value_estimation_scorer(algo, episodes, window_size=1024):
    """ Returns average value estimation (in negative scale).

    This metrics suggests the scale for estimation of Q functions.
    If average value estimation is too large, the Q functions overestimate
    action-values, which possibly makes training failed.

    .. math::

        \\mathbb{E}_{s_t \\sim D} [ \\max_a Q_\\theta (s_t, a)]

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: negative average value estimation.

    """
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, algo.n_frames):
            actions = algo.predict(batch.observations)
            values = algo.predict_value(batch.observations, actions)
            total_values += values.tolist()
    # smaller is better, maybe?
    return -np.mean(total_values)


def value_estimation_std_scorer(algo, episodes, window_size=1024):
    """ Returns standard deviation of value estimation (in negative scale).

    This metrics suggests how confident Q functions are for the given
    episodes.
    This metrics will be more accurate with `boostrap` enabled and the larger
    `n_critics` at algorithm.
    If standard deviation of value estimation is large, the Q functions are
    overfitting to the training set.

    .. math::

        \\mathbb{E}_{s_t \\sim D, a \\sim \\text{argmax}_a Q_\\theta(s_t, a)}
            [Q_{\\text{std}}(s_t, a)]

    where :math:`Q_{\\text{std}}(s, a)` is a standard deviation of action-value
    estimation over ensemble functions.

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: negative standard deviation.

    """
    total_stds = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, algo.n_frames):
            actions = algo.predict(batch.observations)
            _, stds = algo.predict_value(batch.observations, actions, True)
            total_stds += stds.tolist()
    # smaller is better
    return -np.mean(total_stds)


def initial_state_value_estimation_scorer(algo, episodes, window_size=1024):
    """ Returns mean estimated action-values at the initial states.

    This metrics suggests how much return the trained policy would get from
    the initial states by deploying the policy to the states.
    If the estimated value is large, the trained policy is expected to get
    higher returns.

    .. math::

        \\mathbb{E}_{s_0 \\sim D} [Q(s_0, \\pi(s_0))]

    References:
        * `Paine et al., Hyperparameter Selection for Offline Reinforcement
          Learning <https://arxiv.org/abs/2007.09055>`_

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: mean action-value estimation at the initial states.

    """
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, algo.n_frames):
            # estimate action-value in initial states
            actions = algo.predict([batch.observations[0]])
            values = algo.predict_value([batch.observations[0]], actions)
            total_values.append(values[0])
    return np.mean(total_values)


def soft_opc_scorer(return_threshold):
    """ Returns Soft Off-Policy Classification metrics.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer funciton is evaluating gaps of action-value
    estimation between the success episodes and the all episodes.
    If the learned Q-function is optimal, action-values in success episodes
    are expected to be higher than the others.
    The success episode is defined as an episode with a return above the given
    threshold.

    .. math::

        \\mathbb{E}_{s, a \\sim D_{success}} [Q(s, a)]
            - \\mathbb{E}_{s, a \\sim D} [Q(s, a)]

    .. code-block:: python

        from d3rlpy.datasets import get_cartpole
        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import soft_opc_scorer
        from sklearn.model_selection import train_test_split

        dataset, _ = get_cartpole()
        train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

        scorer = soft_opc_scorer(return_threshold=180)

        dqn = DQN()
        dqn.fit(train_episodes,
                eval_episodes=test_episodes,
                scorers={'soft_opc': scorer})

    References:
        * `Irpan et al., Off-Policy Evaluation via Off-Policy Classification.
          <https://arxiv.org/abs/1906.01624>`_

    Args:
        return_threshold (float): threshold of success episodes.

    Returns:
        callable: scorer function.

    """
    def scorer(algo, episodes, window_size=1024):
        success_values = []
        all_values = []
        for episode in episodes:
            is_success = episode.compute_return() >= return_threshold
            for batch in _make_batches(episode, window_size, algo.n_frames):
                values = algo.predict_value(batch.observations, batch.actions)
                all_values += values.reshape(-1).tolist()
                if is_success:
                    success_values += values.reshape(-1).tolist()
        return np.mean(success_values) - np.mean(all_values)

    return scorer


def continuous_action_diff_scorer(algo, episodes, window_size=1024):
    """ Returns squared difference of actions between algorithm and dataset.

    This metrics suggests how different the greedy-policy is from the given
    episodes in continuous action-space.
    If the given episodes are near-optimal, the small action difference would
    be better.

    .. math::

        \\mathbb{E}_{s_t, a_t \\sim D} [(a_t - \\pi_\\phi (s_t))^2]

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: negative squared action difference.

    """
    total_diffs = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, algo.n_frames):
            actions = algo.predict(batch.observations)
            diff = ((batch.actions - actions)**2).sum(axis=1).tolist()
            total_diffs += diff
    # smaller is better, sometimes?
    return -np.mean(total_diffs)


def discrete_action_match_scorer(algo, episodes, window_size=1024):
    """ Returns percentage of identical actions between algorithm and dataset.

    This metrics suggests how different the greedy-policy is from the given
    episodes in discrete action-space.
    If the given episdoes are near-optimal, the large percentage would be
    better.

    .. math::

        \\frac{1}{N} \\sum^N \\parallel
            \\{a_t = \\text{argmax}_a Q_\\theta (s_t, a)\\}

    Args:
        algo (d3rlpy.algos.base.AlgoBase): algorithm.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: percentage of identical actions.

    """
    total_matches = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, algo.n_frames):
            actions = algo.predict(batch.observations)
            match = (batch.actions.reshape(-1) == actions).tolist()
            total_matches += match
    return np.mean(total_matches)


def evaluate_on_environment(env, n_trials=10, epsilon=0.0, render=False):
    """ Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env (gym.Env): gym-styled environment.
        n_trials (int): the number of trials.
        epsilon (float): noise factor for epsilon-greedy policy.
        render (bool): flag to render environment.

    Returns:
        callable: scoerer function.


    """

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    def scorer(algo, *args):
        if is_image:
            stacked_observation = StackedObservation(observation_shape,
                                                     algo.n_frames)

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            while True:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if is_image:
                        action = algo.predict([stacked_observation.eval()])[0]
                    else:
                        action = algo.predict([observation])[0]

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards)

    return scorer


def dynamics_observation_prediction_error_scorer(dynamics,
                                                 episodes,
                                                 window_size=1024):
    """ Returns MSE of observation prediction (in negative scale).

    This metrics suggests how dynamics model is generalized to test sets.
    If the MSE is large, the dynamics model are overfitting.

    .. math::

        \\mathbb{E}_{s_t, a_t, s_{t+1} \\sim D} [(s_{t+1} - s')^2]

    where :math:`s' \\sim T(s_t, a_t)`.

    Args:
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: negative mean squared error.

    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, dynamics.n_frames):
            pred = dynamics.predict(batch.observations, batch.actions)
            errors = ((batch.next_observations - pred[0])**2).sum(axis=1)
            total_errors += errors.tolist()
    # smaller is better
    return -np.mean(total_errors)


def dynamics_reward_prediction_error_scorer(dynamics,
                                            episodes,
                                            window_size=1024):
    """ Returns MSE of reward prediction (in negative scale).

    This metrics suggests how dynamics model is generalized to test sets.
    If the MSE is large, the dynamics model are overfitting.

    .. math::

        \\mathbb{E}_{s_t, a_t, r_{t+1} \\sim D} [(r_{t+1} - r')^2]

    where :math:`r' \\sim T(s_t, a_t)`.

    Args:
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: negative mean squared error.

    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, dynamics.n_frames):
            pred = dynamics.predict(batch.observations, batch.actions)
            errors = ((batch.next_rewards - pred[1])**2).reshape(-1)
            total_errors += errors.tolist()
    # smaller is better
    return -np.mean(total_errors)


def dynamics_prediction_variance_scorer(dynamics, episodes, window_size=1024):
    """ Returns prediction variance of ensemble dynamics (in negative scale).

    This metrics suggests how dynamics model is confident of test sets.
    If the variance is large, the dynamics model has large uncertainty.

    Args:
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        episodes (list(d3rlpy.dataset.Episode)): list of episodes.
        window_size (int): mini-batch size to compute.

    Returns:
        float: negative variance.

    """
    total_variances = []
    for episode in episodes:
        for batch in _make_batches(episode, window_size, dynamics.n_frames):
            pred = dynamics.predict(batch.observations, batch.actions, True)
            total_variances += pred[2].tolist()
    # smaller is better
    return -np.mean(total_variances)


NEGATED_SCORER = [
    td_error_scorer,
    value_estimation_std_scorer,
    average_value_estimation_scorer,
    discounted_sum_of_advantage_scorer,
    continuous_action_diff_scorer,
    dynamics_observation_prediction_error_scorer,
    dynamics_reward_prediction_error_scorer,
    dynamics_prediction_variance_scorer,
]
