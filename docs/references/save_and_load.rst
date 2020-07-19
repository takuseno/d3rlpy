Save and Load
=============

save_model and load_model
-------------------------

.. code-block:: python

    from d3rlpy.datasets import get_cartpole
    from d3rlpy.algos import DQN

    dataset, env = get_cartpole()

    dqn = DQN(n_epochs=1)
    dqn.fit(dataset.episodes)

    # save entire model parameters.
    dqn.save_model('model.pt')

    # load entire model parameters.
    dqn.load_model('model.pt')

`save_model` method saves all parameters including optimizer states, which is
useful when checking all the outputs or re-training from snapshots.


from_json
---------

It is very boring to set the same hyperparameters to initialize algorithms when
loading model parameters.
In d3rlpy, `params.json` is saved at the beggining of `fit` method, which
includes all hyperparameters within the algorithm object.
You can recreate algorithm objects from `params.json` via `from_json` method.

.. code-block:: python

    from d3rlpy.algos import DQN

    dqn = DQN.from_json('d3rlpy_logs/<path-to-json>/params.json')

    # ready to load
    dqn.load_model('model.pt')


save_policy
-----------

.. code-block:: python

    from d3rlpy.datasets import get_cartpole
    from d3rlpy.algos import DQN

    dataset, env = get_cartpole()

    dqn = DQN(n_epochs=1)
    dqn.fit(dataset.episodes)

    # save only greedy-policy
    dqn.save_policy('policy.pt')

`save_policy` method saves the only greedy-policy computation graph as
TorchSciprt.
TorchScript is a optimizable graph expression provided by PyTorch.
When `save_policy` method is called, the greedy-policy graph is constructed
and traced via `torch.jit.trace` function.
The saved policy can be loaded without any dependencies except PyTorch.

.. code-block:: python

    import torch

    # load greedy-policy only with PyTorch
    policy = torch.jit.load('policy.pt')

    # returns greedy actions
    actions = policy(torch.rand(32, 6))

This is especially useful when deploying the trained models to productions.
The computation can be faster and you don't need to install d3rlpy.
Moreover, TorchScript model can be easily loaded even with C++, which will
empower your robotics and embedding system projects.

.. code-block:: c++

    #include <torch/script.h>

    int main(int argc, char* argv[]) {
      torch::jit::script::Module module;
      try {
        module = torch::jit::load("policy.pt")
      } catch (const c10::Error& e) {
        return -1;
      }
      return 0;
    }

You can get more information about TorchScript
`here <https://pytorch.org/docs/stable/jit.html>`_.
