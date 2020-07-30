![format check](https://github.com/takuseno/d3rlpy/workflows/format%20check/badge.svg)
![test](https://github.com/takuseno/d3rlpy/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/takuseno/d3rlpy/branch/master/graph/badge.svg?token=AQ02USKN6Y)](https://codecov.io/gh/takuseno/d3rlpy)
![MIT](https://img.shields.io/badge/license-MIT-blue)

# d3rlpy
Data-driven Deep Reinforcement Learning Library as a Tool.

```py
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import BEAR

# MDPDataset takes arrays of state transitions
dataset = MDPDataset(observations, actions, rewards, terminals)

# train data-driven deep RL
bear = BEAR()
bear.fit(dataset.episodes)

# ready to control
actions = bear.predict(x)
```

These are the design principles of d3rlpy:
- d3rlpy is designed for practical projects unlike the many other RL libraries.
- d3rlpy is not focusing on reproducing RL papers.
- d3rlpy is adding more techniques than the original implementations.

## scikit-learn compatibility
This library is designed as if born from scikit-learn.
You can fully utilize scikit-learn's utilities to increase your productivity.
```py
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import td_error_scorer

train_episodes, test_episodes = train_test_split(dataset)

bear.fit(train_episodes,
         eval_episodes=test_episodes,
         scorers={'td_error': td_error_scorer})
```

You can naturally perform cross-validation.
```py
from sklearn.model_selection import cross_validate

scores = cross_validate(bear, dataset, scoring={'td_error': td_error_scorer})
```

And more.
```py
from sklearn.model_selection import GridSearchCV

gscv = GridSearchCV(estimator=bear,
                    param_grid={'actor_learning_rate': np.arange(1, 10) * 1e-3},
                    scoring={'td_error': td_error_scorer},
                    refit=False)
gscv.fit(train_episodes)
```

## deploy
Machine learning models often require dependencies even after deployment.
d3rlpy provides more flexible options to solve this problem via torch
script so that the production environment never cares about which algorithms
and hyperparameters are used to train.

```py
# save the learned greedy policy as torch script
bear.save_policy('policy.pt')

# load the policy without any dependencies except pytorch
policy = torch.jit.load('policy.pt')
actions = policy(x)
```

even on C++
```c++
torch::jit::script::Module module;
try {
  module = torch::jit::load('policy.pt');
} catch (const c10::Error& e) {
  //
}
```

## supported algorithms
| algorithm | discrete control | continuous control | data-driven RL? |
|:-|:-:|:-:|:-:|
| Behavior Cloning (supervised learning) | :white_check_mark: | :white_check_mark: | |
| [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236) | :white_check_mark: | :no_entry: | |
| [Double DQN](https://arxiv.org/abs/1509.06461) | :white_check_mark: | :no_entry: | |
| [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | :no_entry: | :white_check_mark: | |
| [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/abs/1802.09477) | :no_entry: | :white_check_mark: | |
| [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905) | :no_entry: | :white_check_mark: | |
| [Random Ensemble Mixture (REM)](https://arxiv.org/abs/1907.04543) | :construction: | :no_entry: | :white_check_mark: |
| [Batch Constrained Q-learning (BCQ)](https://arxiv.org/abs/1812.02900) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Bootstrapping Error Accumulation Reduction (BEAR)](https://arxiv.org/abs/1906.00949) | :no_entry: | :white_check_mark: | :white_check_mark: |
| [Advantage-Weighted Regression (AWR)](https://arxiv.org/abs/1910.00177) | :construction: | :construction: | :white_check_mark: |
| [Advantage-weighted Behavior Model (ABM)](https://arxiv.org/abs/2002.08396) | :construction: | :construction: | :white_check_mark: |
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## supported Q functions
- [x] standard Q function
- [x] [Quantile Regression](https://arxiv.org/abs/1710.10044)
- [x] [Implicit Quantile Network](https://arxiv.org/abs/1806.06923)
- [x] [Fully parametrized Quantile Function](https://arxiv.org/abs/1911.02140) (experimental)

## other features
Basically, all features are available with every algorithm.

- [x] ensemble Q function with bootstrapping
- [x] [delayed policy updates](https://arxiv.org/abs/1802.09477)
- [x] parallel cross validation with multiple GPU
- [x] online training
- [x] [Model-based Offline Policy Optimization](https://arxiv.org/abs/2005.13239)
- [ ] automatic image augmentation


## contributions
### coding style
This library is fully formatted with [yapf](https://github.com/google/yapf).
You can format the entire scripts as follows:
```
$ ./scripts/format
```

### test
The unit tests are provided as much as possible.
This repository is using `pytest-cov` instead of `pytest`.
You can run the entire tests as follows:
```
$ ./scripts/test
```

If you give `-p` option, the performance tests with toy tasks are also run
(this will take minutes).
```
$ ./scripts/test -p
```

## acknowledgement
This work is supported by Information-technology Promotion Agency, Japan
(IPA), Exploratory IT Human Resources Project (MITOU Program) in the fiscal
year 2020.
