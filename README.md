![format check](https://github.com/takuseno/scikit-batch-rl/workflows/format%20check/badge.svg)
![test](https://github.com/takuseno/scikit-batch-rl/workflows/test/badge.svg)

# scikit-batch-rl
Data-driven Deep Reinforcement Learning library in scikit-learn style.
Unlike the other RL libraries, scikit-batch-rl is designed for practical projects rather than research ones.

```py
from skbrl.dataset import MDPDataset
from skbrl.algos import BEAR

# MDPDataset takes arrays of state transitions
dataset = MDPDataset(observations, actions, rewards, terminals)

# train data-driven deep RL
bear = BEAR()
bear.fit(dataset.episodes)

# ready to control
actions = bear.predict(x)
```

## scikit-learn compatibility
This library is designed as if born from scikit-learn.
You can fully utilize scikit-learn's utilities to increase your productivity.
```py
from sklearn.model_selection import train_test_split

train_episodes, test_episodes = train_test_split(dataset)

bear.fit(train_episodes)
```

## deploy to products
Machine learning models often require dependencies even after deployment.
scikit-batch-rl provides more flexible options to solve this problem via torch script.
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
### discrete control
- [x] DQN
- [x] Double DQN
- [ ] REM

### continuous control
- [x] DDPG
- [x] TD3
- [ ] SAC
- [ ] BCQ
- [ ] BEAR
- [ ] MOPO

## supported techniques
- [ ] Quantile Regression
- [ ] Implicit Quantile Network
- [ ] random network augmentation

## supported evaluation metrics
- [ ] Off-policy Classification
- [ ] Temporal-difference Error
- [ ] Discounted Sum of Advantages

## controbutions
### coding style
This library is fully formatted with [yapf](https://github.com/google/yapf).
You can format the entire scripts as follows:
```
$ ./format
```

### test
The unit tests are provided as much as possible, including performance tests with toy tasks.
You can run the entire tests as follows:
```
$ ./test
```
