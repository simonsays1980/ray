from ray.rllib.algorithms.rlhf.rlhf import RLHFConfig
from ray.rllib.tuned_examples.rlhf.tldr_rlhf_offline_data import TLDRRLHFOfflineData

config = (
    RLHFConfig()
    .offline_data(
        input_="trl-internal-testing/tldr-preference-sft-trl-style",
    )
    .training(
        sft_model_id="cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr",
    )
)

offline_data = TLDRRLHFOfflineData(config)

for i in range(10):
    batch = offline_data.sample(num_samples=4)
