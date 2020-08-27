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

`save_policy` method saves the only greedy-policy computation graph as
TorchSciprt or ONNX.
When `save_policy` method is called, the greedy-policy graph is constructed
and traced via `torch.jit.trace` function.

.. code-block:: python

    from d3rlpy.datasets import get_cartpole
    from d3rlpy.algos import DQN

    dataset, env = get_cartpole()

    dqn = DQN(n_epochs=1)
    dqn.fit(dataset.episodes)

    # save greedy-policy as TorchScript
    dqn.save_policy('policy.pt')

    # save greedy-policy as ONNX
    dqn.save_policy('policy.onnx', as_onnx=True)

TorchScript
~~~~~~~~~~~

TorchScript is a optimizable graph expression provided by PyTorch.
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

ONNX
~~~~

ONNX is an open format built to represent machine learning models.
This is also useful when deploying the trained model to productions with
various programming languages including Python, C++, JavaScript and more.

The following example is written with
`onnxruntime <https://github.com/microsoft/onnxruntime>`_.

.. code-block:: python

  import onnxruntime

  # load ONNX policy via onnxruntime
  ort_session = ort.InferenceSession('policy.onnx')

  # observation
  observation = np.random.rand(1, 6).astype(np.float32)

  # returns greedy action
  action = ort_session.run(None, {'input_0': observation})[0]

You can get more information about ONNX `here <https://onnx.ai/>`_.
