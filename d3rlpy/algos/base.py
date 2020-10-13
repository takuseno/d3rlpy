from abc import ABCMeta, abstractmethod
from ..base import ImplBase, LearnableBase
from ..online.iterators import train


class AlgoImplBase(ImplBase):
    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def save_policy(self, fname, as_onnx):
        pass

    @abstractmethod
    def predict_best_action(self, x):
        pass

    @abstractmethod
    def predict_value(self, x, action, with_std):
        pass

    @abstractmethod
    def sample_action(self, x):
        pass


class AlgoBase(LearnableBase):
    def __init__(self, n_epochs, batch_size, n_frames, scaler, augmentation,
                 dynamics, use_gpu):
        super().__init__(n_epochs, batch_size, n_frames, scaler, augmentation,
                         use_gpu)
        self.dynamics = dynamics

    def save_policy(self, fname, as_onnx=False):
        """ Save the greedy-policy computational graph as TorchScript or ONNX.

        .. code-block:: python

            # save as TorchScript
            algo.save_policy('policy.pt')

            # save as ONNX
            algo.save_policy('policy.onnx', as_onnx=True)

        The artifacts saved with this method will work without d3rlpy.
        This method is especially useful to deploy the learned policy to
        production environments or embedding systems.

        See also

            * https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html (for Python).
            * https://pytorch.org/tutorials/advanced/cpp_export.html (for C++).
            * https://onnx.ai (for ONNX)

        Args:
            fname (str): destination file path.
            as_onnx (bool): flag to save as ONNX format.

        """
        self.impl.save_policy(fname, as_onnx)

    def predict(self, x):
        """ Returns greedy actions.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            actions = algo.predict(x)
            # actions.shape == (100, action size) for continuous control
            # actions.shape == (100,) for discrete control

        Args:
            x (numpy.ndarray): observations

        Returns:
            numpy.ndarray: greedy actions

        """
        return self.impl.predict_best_action(x)

    def predict_value(self, x, action, with_std=False):
        """ Returns predicted action-values.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            # for continuous control
            # 100 actions with shape of (2,)
            actions = np.random.random((100, 2))

            # for discrete control
            # 100 actions in integer values
            actions = np.random.randint(2, size=100)

            values = algo.predict_value(x, actions)
            # values.shape == (100,)

            values, stds = algo.predict_value(x, actions, with_std=True)
            # stds.shape  == (100,)

        Args:
            x (numpy.ndarray): observations
            action (numpy.ndarray): actions
            with_std (bool): flag to return standard deviation of ensemble
                estimation. This deviation reflects uncertainty for the given
                observations. This uncertainty will be more accurate if you
                enable `bootstrap` flag and increase `n_critics` value.

        Returns:
            numpy.ndarray: predicted action-values

        """
        return self.impl.predict_value(x, action, with_std)

    def sample_action(self, x):
        """ Returns sampled actions.

        The sampled actions are identical to the output of `predict` method if
        the policy is deterministic.

        Args:
            x (numpy.ndarray): observations.

        Returns:
            numpy.ndarray: sampled actions.

        """
        return self.impl.sample_action(x)

    def fit_online(self,
                   env,
                   buffer,
                   explorer=None,
                   n_epochs=None,
                   n_steps_per_epoch=4000,
                   n_updates_per_epoch=100,
                   update_start_step=0,
                   eval_env=None,
                   eval_epsilon=0.0,
                   experiment_name=None,
                   with_timestamp=True,
                   logdir='d3rlpy_logs',
                   verbose=True,
                   show_progress=True,
                   tensorboard=True,
                   save_interval=1):
        """ Start training loop of online deep reinforcement learning.

        This method is a convenient alias to ``d3rlpy.online.iterators.train``.

        Args:
            env (gym.Env): gym-like environment.
            buffer (d3rlpy.online.buffers.Buffer): replay buffer.
            explorer (d3rlpy.online.explorers.Explorer): action explorer.
            n_epochs (int): the number of epochs to train. If None is given,
                ``n_epochs`` of algorithm object will be used.
            n_steps_per_epoch (int): the number of steps per epoch.
            n_updates_per_epoch (int): the number of updates per epoch.
            update_start_step (int): the steps before starting updates.
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
        train(env=env,
              algo=self,
              buffer=buffer,
              explorer=explorer,
              n_epochs=n_epochs,
              n_steps_per_epoch=n_steps_per_epoch,
              n_updates_per_epoch=n_updates_per_epoch,
              update_start_step=update_start_step,
              eval_env=eval_env,
              eval_epsilon=eval_epsilon,
              experiment_name=experiment_name,
              with_timestamp=with_timestamp,
              logdir=logdir,
              verbose=verbose,
              show_progress=show_progress,
              tensorboard=tensorboard,
              save_interval=save_interval)

    def _generate_new_data(self, transitions):
        new_data = []
        if self.dynamics:
            new_data += self.dynamics.generate(self, transitions)
        return new_data
