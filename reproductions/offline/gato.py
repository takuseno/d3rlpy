import argparse

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    dataset_with_keys = d3rlpy.algos.ReplayBufferWithEmbeddingKeys(
        replay_buffer=dataset,
        observation_to_embedding_keys="normalized_float",
        action_embedding_key="float",
        task_id="main",
    )

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    gato = d3rlpy.algos.GatoConfig(
        embedding_modules={
            "discrete": d3rlpy.models.DiscreteTokenEmbeddingModuleFactory(
                vocab_size=1024,
                embed_size=256,
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
        layer_width=256,
        batch_size=64,
        learning_rate=1e-4,
        optim_factory=d3rlpy.models.AdamWFactory(weight_decay=1e-4),
        action_vocab_size=1024,
        context_size=128,
        num_heads=1,
        num_layers=3,
    ).create(device=args.gpu)

    gato.fit(
        [dataset_with_keys],
        n_steps=100000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={
            "environment": d3rlpy.algos.GatoEnvironmentEvaluator(
                env=env,
                return_integer=False,
                observation_to_embedding_keys="normalized_float",
                action_embedding_key="float",
            )
        },
        experiment_name=f"Gato_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
