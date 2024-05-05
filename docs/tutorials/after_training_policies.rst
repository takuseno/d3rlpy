***************************************
After Training Policies (Save and Load)
***************************************

This page provides answers to frequently asked questions about how to use the trained policies with your environment.

Prepare Pretrained Policies
--------------------------

.. code-block:: python

   import d3rlpy

   # prepare dataset and environment
   dataset, env = d3rlpy.datasets.get_dataset('pendulum-random')

   # setup algorithm
   cql_old = d3rlpy.algos.CQLConfig().create(device="cuda:0")

   # start offline training
   cql_old.fit(dataset, n_steps=100000)


Load Trained Policies
---------------------

.. code-block:: python

   # Option 1: Load d3 file

   # save d3 file
   cql_old.save("model.d3")
   # reconstruct full setup from a d3 file
   cql = d3rlpy.load_learnable("model.d3")


   # Option 2: Load pt file

   # save pt file
   cql_old.save_model("model.pt")
   # setup algorithm manually
   cql = d3rlpy.algos.CQLConfig().create()

   # choose one of three to build PyTorch models

   # if you have MDPDataset object
   cql.build_with_dataset(dataset)
   # or if you have Gym-styled environment object
   cql.build_with_env(env)
   # or manually set observation shape and action size
   cql.create_impl((3,), 1)

   # load pretrained model
   cql.load_model("model.pt")


Inference
---------

Now, you can use ``predict`` method to infer the actions. Please note that the observation MUST have the batch dimension.

.. code-block:: python

   import numpy as np

   # make sure that the observation has the batch dimension
   observation = np.random.random((1, 3))

   # infer the action
   action = cql.predict(observation)
   assert action.shape == (1, 1)


You can manually make the policy interact with the environment.

.. code-block:: python

   observation = env.reset()
   while True:
      action = cql.predict([observation])[0]
      observation, reward, done, _ = env.step(action)
      if done:
          break


Export Policies as TorchScript
------------------------------

Alternatively, you can export the trained policy as TorchScript format.
The advantage of the TorchScript format is that the exported policy can be used by not only Python programs, but also C++ programs, which would be useful for robotics integration.
Another merit is that the trained policy depends only on PyTorch so that you don't need to install d3rlpy at production.

.. code-block:: python

   # export as TorchScript
   cql.save_policy("policy.pt")


   import torch

   # load TorchScript policy
   policy = torch.jit.load("policy.pt")

   # infer the action
   action = policy(torch.rand(1, 3))
   assert action.shape == (1, 1)


If you train your policy with tuple observations, you can feed tuple observations as follows:

.. code-block:: python

   # load TorchScript policy
   policy = torch.jit.load("tuple_policy.pt")

   # infer the action
   tuple_observation = [torch.rand(1, 3), torch.rand(1, 5)]
   action = policy(tuple_observation[0], tuple_observation[1])


Export Policies as ONNX
-----------------------

Alternatively, you can also export the trained policy as ONNX.
ONNX is a widely used machine learning model format that is supported by numerous programming languages.

.. code-block:: python

   # export as ONNX
   cql.save_policy("policy.onnx")


   import onnxruntime as ort

   # load ONNX policy via onnxruntime
   ort_session = ort.InferenceSession('policy.onnx', providers=["CPUExecutionProvider"])

   # observation
   observation = np.random.rand(1, 3).astype(np.float32)

   # returns greedy action
   action = ort_session.run(None, {'input_0': observation})
   assert action.shape == (1, 1)


If you train your policy with tuple observations, you can feed tuple observations as follows:

.. code-block:: python

   # load ONNX policy via onnxruntime
   ort_session = ort.InferenceSession('tuple_policy.onnx', providers=["CPUExecutionProvider"])

   # infer the action
   tuple_observation = [np.random.rand(1, 3).astype(np.float32), np.random.rand(1, 5).astype(np.float32)]
   action = ort_session.run(None, {'input_0': tuple_observation[0], 'input_1': tuple_observation[1]})
