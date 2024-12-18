import gymnasium as gym
import numpy as np

from transformers import GenerationConfig, AutoTokenizer
from ray.rllib.algorithms.rlhf.rlhf import RLHFConfig
from ray.rllib.tuned_examples.rlhf.tldr_rlhf_offline_data import TLDRRLHFOfflineData

SFT_MODEL_ID = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
RM_MODEL_ID = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
tokenizer = AutoTokenizer.from_pretrained(
    SFT_MODEL_ID,
    padding_side="left",
    trust_remote_code=True,
)
TEMPERATURE = 0.7
VOCAB_SIZE = len(tokenizer)
MAX_SEQ_LENGTH = 512 + 53

config = (
    RLHFConfig()
    .environment(
        action_space=gym.spaces.Discrete(VOCAB_SIZE),
        observation_space=gym.spaces.Box(
            low=0,
            # Vocabulary size + padding token (50277) + stop token (0).
            high=VOCAB_SIZE,
            shape=(MAX_SEQ_LENGTH,),
            dtype=np.int32,
        )
    )
    .offline_data(
        input_="trl-internal-testing/tldr-preference-sft-trl-style",
        # TODO (simon): Create a `get_default_offline_data_class` in the
        # AlgorithmConfig.
        offline_data_class=TLDRRLHFOfflineData,
    )
    .training(
        sft_model_id=SFT_MODEL_ID,
        train_batch_size_per_learner=10,
        minibatch_size=1,
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
                temperature=(TEMPERATURE + 1e-7),
                top_k=0.0,
                tok_p=1.0,
                do_sample=True,
            ),
            "temperature": TEMPERATURE,
            "sft_model_id": SFT_MODEL_ID,
            "rm_model_id": RM_MODEL_ID,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )
)

algo = config.build()

for i in range(10):
    res = algo.train()
    print(res)

algo.stop()
