from transformers import GenerationConfig
from ray.rllib.algorithms.rlhf.rlhf import RLHFConfig

TEMPERATURE = 0.7

config = (
    RLHFConfig()
    .offline_data(
        input_="trl-internal-testing/tldr-preference-sft-trl-style",
        iter_batches_kwargs={
            "local_shuffle_buffer_size": None,
        },
    )
    .training(
        sft_model_id="cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr",
    )
    .rl_module(
        model_config={
            # TODO (simon): Check, if we just provide the kwargs here.
            "generation_config": GenerationConfig(
                # TODO (simon): This is an important hyperparameter (response
                # length). Put it into the config.
                max_new_tokens=53,
                # TODO: B/c we need temperature also in forward passes it is better
                # generate the GenerationConfig inside the module.
                temperature=(TEMPERATURE + 1e7),
                top_k=0.0,
                tok_p=1.0,
                do_sample=True,
            )
        }
    )
)

algo = config.build()
