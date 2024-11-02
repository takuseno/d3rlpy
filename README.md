<p align="center"><img align="center" width="300px" src="assets/logo.png"></p>

# d3rlpy: An offline deep reinforcement learning library

![test](https://github.com/takuseno/d3rlpy/workflows/test/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/d3rlpy/badge/?version=latest)](https://d3rlpy.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/takuseno/d3rlpy/branch/master/graph/badge.svg?token=AQ02USKN6Y)](https://codecov.io/gh/takuseno/d3rlpy)
[![Maintainability](https://api.codeclimate.com/v1/badges/c9162eb736d0b0f612d8/maintainability)](https://codeclimate.com/github/takuseno/d3rlpy/maintainability)
![MIT](https://img.shields.io/badge/license-MIT-blue)

d3rlpy is an offline deep reinforcement learning library for practitioners and researchers.

```py
import d3rlpy

dataset, env = d3rlpy.datasets.get_dataset("hopper-medium-v0")

# prepare algorithm
sac = d3rlpy.algos.SACConfig().create(device="cuda:0")

# train offline
sac.fit(dataset, n_steps=1000000)

# train online
sac.fit_online(env, n_steps=1000000)

# ready to control
actions = sac.predict(x)
```

- Documentation: https://d3rlpy.readthedocs.io
- Paper: https://arxiv.org/abs/2111.03788

> [!IMPORTANT]
> v2.x.x introduces breaking changes. If you still stick to v1.x.x, please explicitly install previous versions (e.g. `pip install d3rlpy==1.1.1`).

## Key features

### :zap: Most Practical RL Library Ever
- **offline RL**: d3rlpy supports state-of-the-art offline RL algorithms. Offline RL is extremely powerful when the online interaction is not feasible during training (e.g. robotics, medical).
- **online RL**: d3rlpy also supports conventional state-of-the-art online training algorithms without any compromising, which means that you can solve any kinds of RL problems only with `d3rlpy`.

### :beginner: User-friendly API
- **zero-knowledge of DL library**: d3rlpy provides many state-of-the-art algorithms through intuitive APIs. You can become a RL engineer even without knowing how to use deep learning libraries.
- **extensive documentation**: d3rlpy is fully documented and accompanied with tutorials and reproduction scripts of the original papers.

### :rocket: Beyond State-of-the-art
- **distributional Q function**: d3rlpy is the first library that supports distributional Q functions in the all algorithms. The distributional Q function is known as the very powerful method to achieve the state-of-the-performance.
- **data-prallel distributed training**: d3rlpy is the first library that supports data-parallel distributed offline RL training, which allows you to scale up offline RL with multiple GPUs or nodes. See [example](examples/distributed_offline_training.py).


## Installation
d3rlpy supports Linux, macOS and Windows.

### Dependencies
Installing d3rlpy package will install or upgrade the following packages to satisfy requirements:
- torch>=2.5.0
- tqdm>=4.66.3
- gym>=0.26.0
- gymnasium>=1.0.0
- click
- colorama
- dataclasses-json
- h5py
- structlog
- typing-extensions

### PyPI (recommended)
[![PyPI version](https://badge.fury.io/py/d3rlpy.svg)](https://badge.fury.io/py/d3rlpy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/d3rlpy)
```
$ pip install d3rlpy
```
### Anaconda
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/d3rlpy/badges/version.svg)](https://anaconda.org/conda-forge/d3rlpy)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/d3rlpy/badges/downloads.svg)](https://anaconda.org/conda-forge/d3rlpy)
```
$ conda install conda-forge/noarch::d3rlpy
```

### Docker
![Docker Pulls](https://img.shields.io/docker/pulls/takuseno/d3rlpy)
```
$ docker run -it --gpus all --name d3rlpy takuseno/d3rlpy:latest bash
```

## Supported algorithms
| algorithm | discrete control | continuous control |
|:-|:-:|:-:|
| Behavior Cloning (supervised learning) | :white_check_mark: | :white_check_mark: |
| [Neural Fitted Q Iteration (NFQ)](https://link.springer.com/chapter/10.1007/11564096_32) | :white_check_mark: | :no_entry: |
| [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236) | :white_check_mark: | :no_entry: |
| [Double DQN](https://arxiv.org/abs/1509.06461) | :white_check_mark: | :no_entry: |
| [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | :no_entry: | :white_check_mark: |
| [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/abs/1802.09477) | :no_entry: | :white_check_mark: |
| [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905) | :white_check_mark: | :white_check_mark: |
| [Batch Constrained Q-learning (BCQ)](https://arxiv.org/abs/1812.02900) | :white_check_mark: | :white_check_mark: |
| [Bootstrapping Error Accumulation Reduction (BEAR)](https://arxiv.org/abs/1906.00949) | :no_entry: | :white_check_mark: |
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | :white_check_mark: | :white_check_mark: |
| [Advantage Weighted Actor-Critic (AWAC)](https://arxiv.org/abs/2006.09359) | :no_entry: | :white_check_mark: |
| [Critic Reguralized Regression (CRR)](https://arxiv.org/abs/2006.15134) | :no_entry: | :white_check_mark: |
| [Policy in Latent Action Space (PLAS)](https://arxiv.org/abs/2011.07213) | :no_entry: | :white_check_mark: |
| [TD3+BC](https://arxiv.org/abs/2106.06860) | :no_entry: | :white_check_mark: |
| [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169) | :no_entry: | :white_check_mark: |
| [Calibrated Q-Learning (Cal-QL)](https://arxiv.org/abs/2303.05479) | :no_entry: | :white_check_mark: |
| [ReBRAC](https://arxiv.org/abs/2305.09836) | :no_entry: | :white_check_mark: |
| [Decision Transformer](https://arxiv.org/abs/2106.01345) | :white_check_mark: | :white_check_mark: |
| [Gato](https://arxiv.org/abs/2205.06175) | :construction: | :construction: |

## Supported Q functions
- [x] standard Q function
- [x] [Quantile Regression](https://arxiv.org/abs/1710.10044)
- [x] [Implicit Quantile Network](https://arxiv.org/abs/1806.06923)

## Benchmark results
d3rlpy is benchmarked to ensure the implementation quality.
The benchmark scripts are available [reproductions](https://github.com/takuseno/d3rlpy/tree/master/reproductions) directory.
The benchmark results are available [d3rlpy-benchmarks](https://github.com/takuseno/d3rlpy-benchmarks) repository.

## Examples
### MuJoCo
<p align="center"><img align="center" width="160px" src="assets/mujoco_hopper.gif"></p>

```py
import d3rlpy

# prepare dataset
dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')

# prepare algorithm
cql = d3rlpy.algos.CQLConfig().create(device='cuda:0')

# train
cql.fit(
    dataset,
    n_steps=100000,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
)
```

See more datasets at [d4rl](https://github.com/rail-berkeley/d4rl).

### Atari 2600
<p align="center"><img align="center" width="160px" src="assets/breakout.gif"></p>

```py
import d3rlpy

# prepare dataset (1% dataset)
dataset, env = d3rlpy.datasets.get_atari_transitions(
    'breakout',
    fraction=0.01,
    num_stack=4,
)

# prepare algorithm
cql = d3rlpy.algos.DiscreteCQLConfig(
    observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
    reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
).create(device='cuda:0')

# start training
cql.fit(
    dataset,
    n_steps=1000000,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env, epsilon=0.001)},
)
```

See more Atari datasets at [d4rl-atari](https://github.com/takuseno/d4rl-atari).


### Online Training
```py
import d3rlpy
import gym

# prepare environment
env = gym.make('Hopper-v3')
eval_env = gym.make('Hopper-v3')

# prepare algorithm
sac = d3rlpy.algos.SACConfig().create(device='cuda:0')

# prepare replay buffer
buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=env)

# start training
sac.fit_online(env, buffer, n_steps=1000000, eval_env=eval_env)
```

## Tutorials
Try cartpole examples on Google Colaboratory!

- offline RL tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/cartpole.ipynb)
- online RL tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/online.ipynb)

More tutorial documentations are available [here](https://d3rlpy.readthedocs.io/en/stable/tutorials/index.html).

## Contributions
Any kind of contribution to d3rlpy would be highly appreciated!
Please check the [contribution guide](CONTRIBUTING.md).

## Community
| Channel | Link |
|:-|:-|
| Issues | [GitHub Issues](https://github.com/takuseno/d3rlpy/issues) |

> [!IMPORTANT]
> Please do NOT email to any contributors including the owner of this project to ask for technical support. Such emails will be ignored without replying to your message. Use GitHub Issues to report your problems.

## Projects using d3rlpy
| Project | Description |
|:-:|:-|
| [MINERVA](https://github.com/takuseno/minerva) | An out-of-the-box GUI tool for offline RL |
| [SCOPE-RL](https://github.com/hakuhodo-technologies/scope-rl) | An off-policy evaluation and selection library |

## Roadmap
The roadmap to the future release is available in [ROADMAP.md](ROADMAP.md).

## Citation
The paper is available [here](https://arxiv.org/abs/2111.03788).
```
@article{d3rlpy,
  author  = {Takuma Seno and Michita Imai},
  title   = {d3rlpy: An Offline Deep Reinforcement Learning Library},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {315},
  pages   = {1--20},
  url     = {http://jmlr.org/papers/v23/22-0017.html}
}
```

## Acknowledgement
This work started as a part of [Takuma Seno](https://github.com/takuseno)'s Ph.D project at Keio University in 2020.

This work is supported by Information-technology Promotion Agency, Japan
(IPA), Exploratory IT Human Resources Project (MITOU Program) in the fiscal
year 2020.
