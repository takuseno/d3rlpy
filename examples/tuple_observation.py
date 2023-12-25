import argparse

import torch
from torch import nn

import d3rlpy
from d3rlpy.models.torch import Encoder, EncoderWithAction
from d3rlpy.types import Shape, TorchObservation


class TupleEncoder(Encoder):
    def __init__(self, observation_shape: Shape):
        super().__init__()
        shape1, shape2 = observation_shape
        assert isinstance(shape1, (tuple, list))
        assert isinstance(shape2, (tuple, list))
        self.fc1 = nn.Linear(shape1[0], 256)
        self.fc2 = nn.Linear(shape2[0], 256)
        self.shared = nn.Linear(256 * 2, 256)

    def forward(self, x: TorchObservation) -> torch.Tensor:
        h1 = self.fc1(x[0])
        h2 = self.fc2(x[1])
        return self.shared(torch.cat([h1, h2], dim=1))


class TupleEncoderWithAction(EncoderWithAction):
    def __init__(self, observation_shape: Shape, action_size: int):
        super().__init__()
        shape1, shape2 = observation_shape
        assert isinstance(shape1, (tuple, list))
        assert isinstance(shape2, (tuple, list))
        self.fc1 = nn.Linear(shape1[0], 256)
        self.fc2 = nn.Linear(shape2[0], 256)
        self.shared = nn.Linear(256 * 2 + action_size, 256)

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        h1 = self.fc1(x[0])
        h2 = self.fc2(x[1])
        return self.shared(torch.cat([h1, h2, action], dim=1))


class TupleEncoderFactory(d3rlpy.models.EncoderFactory):
    def create(self, observation_shape: Shape) -> Encoder:
        return TupleEncoder(observation_shape)

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> EncoderWithAction:
        return TupleEncoderWithAction(observation_shape, action_size)

    @staticmethod
    def get_type() -> str:
        return "custom"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_minari(
        "antmaze-umaze-v0", tuple_observation=True
    )

    cql = d3rlpy.algos.SACConfig(
        actor_encoder_factory=TupleEncoderFactory(),
        critic_encoder_factory=TupleEncoderFactory(),
        observation_scaler=d3rlpy.preprocessing.TupleObservationScaler(
            [
                d3rlpy.preprocessing.StandardObservationScaler(),
                d3rlpy.preprocessing.StandardObservationScaler(),
            ]
        ),
    ).create(device=args.gpu)

    # start training
    cql.fit(
        dataset,
        n_steps=100000,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        n_steps_per_epoch=1000,
    )


if __name__ == "__main__":
    main()
