import copy
import dataclasses
from typing import Dict, Sequence, cast

import gym
import torch
import torch.nn.functional as F
from torch import nn

import d3rlpy
from d3rlpy.torch_utility import hard_sync


class QFunction(nn.Module):  # type: ignore
    def __init__(self, observation_shape: Sequence[int], action_size: int):
        super().__init__()
        self._fc1 = nn.Linear(observation_shape[0], 256)
        self._fc2 = nn.Linear(256, 256)
        self._fc3 = nn.Linear(256, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self._fc1(x))
        h = torch.relu(self._fc2(h))
        return self._fc3(h)


@dataclasses.dataclass(frozen=True)
class CustomAlgoModules(d3rlpy.Modules):
    q_func: QFunction
    targ_q_func: QFunction
    optim: torch.optim.Optimizer


class CustomAlgoImpl(d3rlpy.algos.QLearningAlgoImplBase):
    _modules: CustomAlgoModules

    def __init__(
        self,
        observation_shape: d3rlpy.types.Shape,
        action_size: int,
        modules: CustomAlgoModules,
        target_update_interval: int,
        gamma: float,
        device: str,
    ):
        super().__init__(observation_shape, action_size, modules, device)
        self._target_update_interval = target_update_interval
        self._gamma = gamma

    def inner_update(
        self, batch: d3rlpy.TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        self._modules.optim.zero_grad()

        with torch.no_grad():
            # (N, 1)
            targ_q = (
                self._modules.targ_q_func(batch.next_observations)
                .max(dim=1, keepdims=True)
                .values
            )
            # compute target
            y = batch.rewards + self._gamma * targ_q * (1 - batch.terminals)

        # compute TD loss
        action_mask = F.one_hot(
            batch.actions.view(-1).long(), num_classes=self._action_size
        )
        q = (action_mask * self._modules.q_func(batch.observations)).sum(
            dim=1, keepdims=True
        )
        loss = ((q - y) ** 2).mean()

        # update parameters
        loss.backward()
        self._modules.optim.step()

        # update target
        if grad_step % self._target_update_interval == 0:
            hard_sync(self._modules.targ_q_func, self._modules.q_func)

        return {"loss": float(loss.detach().numpy())}

    def inner_predict_best_action(
        self, x: d3rlpy.types.TorchObservation
    ) -> torch.Tensor:
        q = self._modules.q_func(x)
        return q.argmax(dim=1)

    def inner_sample_action(
        self, x: d3rlpy.types.TorchObservation
    ) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(
        self, x: d3rlpy.types.TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        q = self._modules.q_func(x)
        flat_action = action.reshape(-1)
        return q[torch.arange(0, q.size(0)), flat_action].reshape(-1)


@dataclasses.dataclass()
class CustomAlgoConfig(d3rlpy.base.LearnableConfig):
    batch_size: int = 32
    learning_rate: float = 1e-3
    target_update_interval: int = 100
    gamma: float = 0.99

    def create(self, device: d3rlpy.base.DeviceArg = False) -> "CustomAlgo":
        return CustomAlgo(self, device)

    @staticmethod
    def get_type() -> str:
        return "custom"


class CustomAlgo(
    d3rlpy.algos.QLearningAlgoBase[CustomAlgoImpl, CustomAlgoConfig]
):
    def inner_create_impl(
        self, observation_shape: d3rlpy.types.Shape, action_size: int
    ) -> None:
        # create Q-functions
        q_func = QFunction(cast(Sequence[int], observation_shape), action_size)
        targ_q_func = copy.deepcopy(q_func)

        # move to device
        q_func.to(self._device)
        targ_q_func.to(self._device)

        # create optimizer
        optim = torch.optim.Adam(
            params=q_func.parameters(),
            lr=self._config.learning_rate,
        )

        # prepare Modules object
        modules = CustomAlgoModules(
            q_func=q_func,
            targ_q_func=targ_q_func,
            optim=optim,
        )

        # create CustomAlgoImpl object
        self._impl = CustomAlgoImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            target_update_interval=self._config.target_update_interval,
            gamma=self._config.gamma,
            device=self._device,
        )

    def get_action_type(self) -> d3rlpy.ActionSpace:
        return d3rlpy.ActionSpace.DISCRETE


def main() -> None:
    # prepare environments
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    # prepare custom algorithm
    explorer = d3rlpy.algos.ConstantEpsilonGreedy(epsilon=0.3)
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=1000000, env=env)
    algo = CustomAlgoConfig().create()

    # start training
    algo.fit_online(
        env=env,
        explorer=explorer,
        buffer=buffer,
        eval_env=eval_env,
        n_steps=100000,
        n_steps_per_epoch=100,
    )


if __name__ == "__main__":
    main()
