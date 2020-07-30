from ..base import ImplBase, LearnableBase
from abc import ABCMeta, abstractmethod


class AlgoImplBase(ImplBase):
    @abstractmethod
    def save_policy(self, fname):
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
    def __init__(self, n_epochs, batch_size, scaler, dynamics, use_gpu):
        super().__init__(n_epochs, batch_size, scaler, use_gpu)
        self.dynamics = dynamics

    def save_policy(self, fname):
        """ Save the greedy-policy computational graph as TorchScript.

        .. code-block:: python

            algo.save_policy('policy.pt')

        The artifacts saved with this method will work without any dependencies
        except pytorch.
        This method is especially useful to deploy the learned policy to
        production environments or embedding systems.

        See also

            * https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html (for Python).
            * https://pytorch.org/tutorials/advanced/cpp_export.html (for C++).

        Args:
            fname (str): destination file path.

        """
        self.impl.save_policy(fname)

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

    def _generate_new_data(self, transitions):
        new_data = []
        if self.dynamics:
            new_data += self.dynamics.generate(self, transitions)
        return new_data
