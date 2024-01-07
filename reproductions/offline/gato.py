import argparse
import dataclasses
from typing import Any

import gym

import d3rlpy


@dataclasses.dataclass(frozen=True)
class Task:
    env: gym.Env[Any, Any]
    dataset: d3rlpy.algos.ReplayBufferWithEmbeddingKeys
    demonstration: d3rlpy.dataset.EpisodeBase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    task_names = ["hopper", "walker2d", "halfcheetah"]
    tasks = {}
    for name in task_names:
        dataset, env = d3rlpy.datasets.get_dataset(f"{name}-expert-v2")
        dataset_with_keys = d3rlpy.algos.ReplayBufferWithEmbeddingKeys(
            replay_buffer=dataset,
            observation_to_embedding_keys="normalized_float",
            action_embedding_key="float",
            task_id=name,
        )
        task = Task(
            env=env,
            dataset=dataset_with_keys,
            demonstration=dataset.episodes[0],
        )
        tasks[name] = task

    gato = d3rlpy.algos.GatoConfig(
        embedding_modules={
            "discrete": d3rlpy.models.DiscreteTokenEmbeddingModuleFactory(
                vocab_size=1024,
                embed_size=512,
            ),
        },
        token_embeddings={
            "normalized_float": d3rlpy.models.FloatTokenEmbeddingFactory(
                embedding_module_key="discrete",
                num_bins=1024,
                use_mu_law_encode=True,
                mu=100,
                basis=256,
            ),
            "float": d3rlpy.models.FloatTokenEmbeddingFactory(
                embedding_module_key="discrete",
                num_bins=1024,
                use_mu_law_encode=False,
            ),
        },
        layer_width=512,
        batch_size=128,
        clip_grad_norm=1.0,
        initial_learning_rate=1e-7,
        maximum_learning_rate=1e-4,
        warmup_steps=15000,
        final_steps=1000000,
        optim_factory=d3rlpy.models.GPTAdamWFactory(
            weight_decay=0.1, betas=(0.9, 0.95)
        ),
        action_vocab_size=1024,
        context_size=256,
        num_heads=4,
        num_layers=3,
        embed_activation_type="tanh",
    ).create(device=args.gpu)

    gato.fit(
        [task.dataset for task in tasks.values()],
        n_steps=1000000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={
            name: d3rlpy.algos.GatoEnvironmentEvaluator(
                env=task.env,
                return_integer=False,
                observation_to_embedding_keys="normalized_float",
                action_embedding_key="float",
                demonstration=task.demonstration,
            )
            for name, task in tasks.items()
        },
    )


if __name__ == "__main__":
    main()
