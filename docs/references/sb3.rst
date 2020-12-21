Stable-Baselines3 Wrapper
==========================

d3rlpy provides a minimal wrapper to use `Stable-Baselines3 (SB3) <https://github.com/DLR-RM/stable-baselines3>`_
features, like utility helpers or SB3 algorithms to create datasets.


.. note::

	This wrapper is far from complete, and only provide a minimal integration with SB3.


Convert SB3 replay buffer to d3rlpy dataset
-------------------------------------------

A replay buffer from Stable-Baselines3 can be easily converted to a :class:`d3rlpy.dataset.MDPDataset`
using ``to_mdp_dataset()`` utility function.


.. code-block:: python

    import stable_baselines3 as sb3

    from d3rlpy.algos import AWR
    from d3rlpy.wrappers.sb3 import to_mdp_dataset

    # Train an off-policy agent with SB3
    model = sb3.SAC("MlpPolicy", "Pendulum-v0", learning_rate=1e-3, verbose=1)
    model.learn(6000)

    # Convert to d3rlpy MDPDataset
    dataset = to_mdp_dataset(model.replay_buffer)
    # The dataset can then be used to train a d3rlpy model
    offline_model = AWR()
    offline_model.fit(dataset.episodes, n_epochs=100)


Convert d3rlpy to use SB3 helpers
----------------------------------

An agent from d3rlpy can be converted to use the SB3 interface (notably follow the interface of SB3 ``predict()``).
This allows to use SB3 helpers like ``evaluate_policy``.


.. code-block:: python

  import gym
  from stable_baselines3.common.evaluation import evaluate_policy

  from d3rlpy.algos import AWAC
  from d3rlpy.wrappers.sb3 import SB3Wrapper

  env = gym.make("Pendulum-v0")

  # Define an offline RL model
  offline_model = AWAC()
  # Train it using for instance a dataset created by a SB3 agent (see above)
  offline_model.fit(dataset.episodes, n_epochs=10)

  # Use SB3 wrapper (convert `predict()` method to follow SB3 API)
  # to have access to SB3 helpers
  # d3rlpy model is accessible via `wrapped_model.algo`
  wrapped_model = SB3Wrapper(offline_model)

  obs = env.reset()

  # We can now use SB3's predict style
  # it returns the action and the hidden states (for RNN policies)
  actions, _ = wrapped_model.predict(observations, deterministic=True)
  # The following is equivalent to offline_model.sample_action(obs)
  actions, _ = wrapped_model.predict(observations, deterministic=False)

  # Evaluate the trained model using SB3 helper
  mean_reward, std_reward = evaluate_policy(wrapped_model, env)

  print(f"mean_reward={mean_reward} +/- {std_reward}")

  # Call methods from the wrapped d3rlpy model
  wrapped_model.sample_action(obs)
  wrapped_model.fit(dataset.episodes, n_epochs=10)

  # Set attributes
  wrapped_model.n_epochs = 2
  # wrapped_model.n_epochs points to d3rlpy wrapped_model.algo.n_epochs
  assert wrapped_model.algo.n_epochs == 2
